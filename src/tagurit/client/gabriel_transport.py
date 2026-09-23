"""
Send scheduler-selected images through Gabriel and recover from connection failures

Unresolved images retain their identity and contents across connection attempts
Only a matching receipt releases the scheduler's in-flight frame
"""

import asyncio
import time
from collections.abc import Callable
from typing import Any

from gabriel_client.gabriel_client import InputProducer
from gabriel_client.websocket_client import WebsocketClient
from gabriel_protocol.v1 import gabriel_pb2
from websockets.exceptions import WebSocketException

from tagurit.client.config import (
    CONNECTION_TIMEOUT_SECONDS,
    GABRIEL_ENDPOINT,
    GABRIEL_ENGINE_ID,
    GABRIEL_PRODUCER_NAME,
    LOOP_INTERVAL_SECONDS,
    RECEIPT_TIMEOUT_SECONDS,
    RECONNECT_INTERVAL_SECONDS,
    SEND_INTERVAL_SECONDS
)
from tagurit.client.frame_scheduler import FrameScheduler
from tagurit.protocol import ImageFrame
from tagurit.shared.image_protocol import (
    ImageMessage,
    decode_receipt,
    encode_image,
    make_image_message,
    validate_identity
)


# Separate invalid receipts from connection failures that can be retried
class ReceiptValidationError(ValueError):
    """
    Report a receipt that failed validation

    This error ends the transport instead of treating invalid data as an outage
    """


# Keep unfinished images while reconnecting Gabriel as needed
class GabrielTransport:
    """
    Connect the scheduler to one Gabriel image producer

    Connection failures and timeouts trigger retries without discarding work
    The unresolved image is retried before selecting another frame
    Calls MUST run on the same event loop as the scheduler's other operations

    Parameters:
        - scheduler (FrameScheduler): Owner of waiting and unresolved frames
        - session_id (str): Stable UUID for this client run
        - report (Callable[[str, str], None]): Callback receiving event and detail
    """

    # Keep application state separate from each network connection
    def __init__(self, scheduler: FrameScheduler, session_id: str, report: Callable[[str, str], None]) -> None:
        validate_identity(session_id, 1)

        self._scheduler = scheduler
        self._session_id = session_id
        self._report = report
        self._accepting = True
        self._link_allowed = True
        self._ready = False

        # Retain application work across connection attempts
        self._pending_frame: ImageFrame | None = None
        self._pending_message: ImageMessage | None = None
        self._submitted_at: float | None = None
        self._last_submit = -float("inf")

        # Track connection attempts and receipt validation failures
        self._receipt_error: Exception | None = None
        self._attempt_number = 0
        self._active_attempt: int | None = None
        self.completed = 0

        # No connection is available until Gabriel finishes registration
        self._scheduler.set_connected(False)

    # Apply an optional simulation gate without overriding connection failures
    def set_link_allowed(self, available: bool) -> None:
        """
        Allow or pause submissions independently of actual connection readiness

        True still requires Gabriel to be ready
        False pauses submissions but does not close the connection

        Parameters:
            - available (bool): Whether the application permits submissions

        Return:
            void
        """
        self._link_allowed = available
        self._update_scheduler_connection()

    # Allow sending only when the application gate and Gabriel both permit it
    def _update_scheduler_connection(self) -> None:
        previous_state = self._scheduler.state
        self._scheduler.set_connected(self._link_allowed and self._ready)

        if previous_state != self._scheduler.state:
            self._report(
                "STATE",
                f"{previous_state.value} -> {self._scheduler.state.value}"
            )

    # Stop selecting new work but still allow an unresolved image to retry
    def stop_accepting(self) -> None:
        """
        Stop reserving new frames while allowing the unresolved frame to finish

        This does not drain waiting queues or stop the connection loop

        Return:
            void
        """
        self._accepting = False

    # Supply new work or retry the unresolved image on a fresh connection
    async def _produce(self, attempt: int) -> gabriel_pb2.InputFrame:

        # Gabriel calls the producer after registration and token availability
        if not self._ready:
            self._ready = True
            self._report("LINK", "Gabriel registered and ready")
            self._update_scheduler_connection()

        while self._active_attempt == attempt:
            now = time.monotonic()

            # Wait for availability, pacing and any outstanding receipt
            pacing_ready = now - self._last_submit >= SEND_INTERVAL_SECONDS

            if self._link_allowed and self._submitted_at is None and pacing_ready:
                retrying = self._pending_frame is not None

                # Existing unresolved work always takes priority over new work
                if not retrying and self._accepting:
                    frame = self._scheduler.reserve_next_frame()

                    if frame is not None:
                        self._pending_frame = frame
                        self._pending_message = make_image_message(
                            self._session_id,
                            frame.frame_id,
                            frame.image_bytes
                        )

                # Resend the retained message or submit the newly selected one
                if self._pending_frame is not None:
                    self._submitted_at = now
                    self._last_submit = now
                    event = "RESEND" if retrying else "SUBMIT"

                    self._report(
                        event,
                        f"Frame {self._pending_frame.frame_id} "
                        f"from {self._scheduler.inflight_lane()}"
                    )

                    return gabriel_pb2.InputFrame(
                        payload_type=gabriel_pb2.PayloadType.IMAGE,
                        byte_payload=encode_image(self._pending_message)
                    )

            await asyncio.sleep(LOOP_INTERVAL_SECONDS)

        raise asyncio.CancelledError()

    # Complete an image only after checking a receipt from the active connection
    def _receive_result(self, result: Any, attempt: int) -> None:

        # Ignore callbacks belonging to a connection that has been abandoned
        if attempt != self._active_attempt:
            return

        try:
            frame = self._pending_frame
            expected = self._pending_message

            if frame is None or expected is None:
                raise ValueError("Receipt arrived without an outstanding image")

            receipt = decode_receipt(result.string_result)

            # Match the receipt to the application identity
            same_session = receipt.session_id == expected.session_id
            same_frame = receipt.frame_id == expected.frame_id

            if not same_session or not same_frame:
                raise ValueError("Receipt identifies a different application image")

            # Match the receipt to the submitted image contents
            same_hash = receipt.sha256 == expected.sha256
            same_size = receipt.byte_count == len(frame.image_bytes)

            if not same_hash or not same_size:
                raise ValueError("Receipt does not match the submitted image bytes")

            # Release retained work only after every receipt check passes
            previous_state = self._scheduler.state
            self._scheduler.acknowledge_frame(frame.frame_id)
            self._pending_frame = None
            self._pending_message = None
            self._submitted_at = None
            self.completed += 1

            self._report(
                "ACK",
                f"Frame {frame.frame_id} received and retained by server"
            )

            if previous_state != self._scheduler.state:
                self._report(
                    "STATE",
                    f"{previous_state.value} -> {self._scheduler.state.value}"
                )

        except (ValueError, TypeError, AttributeError) as error:
            self._receipt_error = error

    # Build a new Gabriel client with fresh connection and token state
    def _create_client(self, attempt: int) -> WebsocketClient:

        # Associate producer calls with this connection attempt
        async def produce() -> gabriel_pb2.InputFrame:
            return await self._produce(attempt)

        # Associate returned results with this connection attempt
        def consume(result: Any) -> None:
            self._receive_result(result, attempt)

        producer = InputProducer(
            producer=produce,
            target_engine_ids=[GABRIEL_ENGINE_ID],
            producer_name=GABRIEL_PRODUCER_NAME
        )

        return WebsocketClient(
            server_endpoint=GABRIEL_ENDPOINT,
            input_producers=[producer],
            consumer=consume
        )

    # Watch one connection for termination or missing receipts
    async def _watch_connection(self, client_task: asyncio.Task[Any], started: float) -> None:
        while True:

            # Invalid receipts are surfaced as application errors
            if self._receipt_error is not None:
                raise ReceiptValidationError("Receipt validation failed") from self._receipt_error

            # Treat a completed connection task as a lost connection
            if client_task.done():
                client_task.result()
                raise ConnectionError("Gabriel connection stopped")

            now = time.monotonic()

            # Require Gabriel registration within the startup timeout
            if not self._ready and now - started > CONNECTION_TIMEOUT_SECONDS:
                raise TimeoutError("Gabriel did not become ready")

            # Require a matching receipt within the transmission timeout
            if self._submitted_at is not None:
                if now - self._submitted_at > RECEIPT_TIMEOUT_SECONDS:
                    raise TimeoutError("No matching receipt arrived")

            await asyncio.sleep(LOOP_INTERVAL_SECONDS)

    # Reconnect after connection failures without clearing retained work
    async def run(self) -> None:
        """
        Run connection attempts until cancelled or an unexpected error occurs

        Network failures and timeouts mark the scheduler disconnected
        A fresh Gabriel connection retries unresolved work with the same identity
        Invalid receipts RAISE ReceiptValidationError

        Return:
            void
        """
        while True:
            self._attempt_number += 1
            self._active_attempt = self._attempt_number
            self._ready = False
            self._submitted_at = None
            self._receipt_error = None

            # Create fresh network state while keeping the application session
            client = self._create_client(self._attempt_number)
            self._report("CONNECT", f"Attempt {self._attempt_number}")
            client_task = asyncio.create_task(client.launch_async())

            try:
                await self._watch_connection(client_task, time.monotonic())

            except (OSError, WebSocketException) as error:
                self._report("LINK", f"Connection unavailable: {error}")

            finally:

                # Cancel the old connection before starting another attempt
                self._active_attempt = None
                self._ready = False
                self._update_scheduler_connection()

                client_task.cancel()
                await asyncio.gather(client_task, return_exceptions=True)
                self._submitted_at = None

            # TimeoutError and ConnectionError are also subclasses of OSError
            self._report(
                "RETRY",
                f"Next connection attempt in {RECONNECT_INTERVAL_SECONDS:g}s"
            )

            await asyncio.sleep(RECONNECT_INTERVAL_SECONDS)
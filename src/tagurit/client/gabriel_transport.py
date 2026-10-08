"""
Send scheduler-selected images through Gabriel and recover from connection failures

Unresolved images retain their identity and contents across connection attempts
Only a matching receipt releases the scheduler's in-flight frame
"""

import asyncio
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
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
    IMAGE_CODEC,
    LOOP_INTERVAL_SECONDS,
    RECEIPT_TIMEOUT_SECONDS,
    RECONNECT_INTERVAL_SECONDS,
    SEND_INTERVAL_SECONDS,
    VIDEO_CRF,
)
from tagurit.client.frame_scheduler import FrameScheduler
from tagurit.protocol import ImageFrame
from tagurit.shared.image_codec import encode_video
from tagurit.shared.image_protocol import (
    ImageMessage,
    decode_receipt,
    encode_image,
    make_image_message,
    validate_identity,
)
from tagurit.shared.telemetry import EventLog


@dataclass
class PendingTransmission:
    """
    Retain encoded contents and the submission time for one unresolved frame.

    This mutable record belongs to the transport and stays on its event loop.
    """

    frame: ImageFrame
    message: ImageMessage | None = None
    wire: bytes | None = None
    submitted_at: float | None = None
    attempt: int | None = None


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
    Unresolved images are retried before selecting new frames
    Calls MUST run on the same event loop as the scheduler's other operations

    Parameters:
        - scheduler (FrameScheduler): Owner of waiting and unresolved frames
        - session_id (str): Stable UUID for this client run
        - report (Callable[[str, str], None]): Callback receiving event and detail
        - codec (str): Image wire codec
        - video_crf (int): Software encoder quality setting
        - send_interval (float | None): Minimum seconds between encoded frame submissions
        - endpoint (str): Gabriel WebSocket URL
        - event_log (EventLog | None): Optional caller-owned CSV recorder
    """

    # Keep application state separate from each network connection
    def __init__(
        self,
        scheduler: FrameScheduler,
        session_id: str,
        report: Callable[[str, str], None],
        codec: str = IMAGE_CODEC,
        video_crf: int = VIDEO_CRF,
        *,
        send_interval: float | None = None,
        endpoint: str = GABRIEL_ENDPOINT,
        event_log: EventLog | None = None,
    ) -> None:
        validate_identity(session_id, 1)
        if send_interval is None:
            send_interval = SEND_INTERVAL_SECONDS
        if not math.isfinite(send_interval) or send_interval < 0:
            raise ValueError("Send interval must be finite and nonnegative")
        self._send_interval = send_interval
        self._endpoint = endpoint
        self._event_log = event_log
        if codec not in ("jpeg", "h264", "h265"):
            raise ValueError("Unsupported transport image codec")
        if type(video_crf) is not int or not 0 <= video_crf <= 51:
            raise ValueError("Video CRF must be an integer from 0 to 51")

        self._scheduler = scheduler
        self._session_id = session_id
        self._report = report
        self._codec = codec
        self._video_crf = video_crf
        self._accepting = True
        self._link_allowed = True
        self._ready = False

        # Retain application work across connection attempts
        self._pending: dict[int, PendingTransmission] = {}
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
            self._report("STATE", f"{previous_state.value} -> {self._scheduler.state.value}")

    # Stop selecting new work but still allow an unresolved image to retry
    def stop_accepting(self) -> None:
        """
        Stop reserving new frames while allowing the unresolved frame to finish

        This does not drain waiting queues or stop the connection loop

        Return:
            void
        """
        self._accepting = False

    # Gabriel acquires a connection token before asking for another payload.
    async def _produce(self, attempt: int) -> gabriel_pb2.InputFrame:
        if attempt != self._active_attempt:
            raise asyncio.CancelledError()
        if not self._ready:
            self._ready = True
            self._report("LINK", f"Gabriel ready | client window={self._scheduler.window_size}")
            self._update_scheduler_connection()

        while self._active_attempt == attempt:
            if self._receipt_error is not None:
                raise ReceiptValidationError("Receipt validation failed") from self._receipt_error
            pacing_ready = time.monotonic() - self._last_submit >= self._send_interval
            if self._link_allowed and pacing_ready:
                # Replay retained entries in reservation order before selecting new work.
                pending = next((p for p in self._pending.values() if p.attempt != attempt), None)
                retrying = pending is not None and pending.wire is not None
                if pending is None and self._accepting:
                    frame = self._scheduler.reserve_next_frame()
                    if frame is not None:
                        pending = PendingTransmission(frame)
                        self._pending[frame.frame_id] = pending
                        self._record("selected", pending)
                if pending is not None:
                    frame = pending.frame
                    if pending.message is None:
                        self._record("encode_start", pending)
                        if self._codec in ("h264", "h265"):
                            encoded, width, height = await asyncio.to_thread(
                                encode_video, frame.image_bytes, self._codec, self._video_crf
                            )
                            pending.message = make_image_message(
                                self._session_id,
                                frame.frame_id,
                                encoded,
                                self._codec,
                                width,
                                height,
                            )
                        else:
                            pending.message = make_image_message(
                                self._session_id, frame.frame_id, frame.image_bytes
                            )
                        self._record("encode_end", pending)
                    if pending.wire is None:
                        pending.wire = encode_image(pending.message)
                    if attempt != self._active_attempt:
                        raise asyncio.CancelledError()
                    pending.submitted_at = time.monotonic()
                    pending.attempt = attempt
                    self._last_submit = pending.submitted_at
                    self._record("submitted", pending, at=pending.submitted_at)
                    self._report(
                        "RESEND" if retrying else "SUBMIT",
                        f"Frame {frame.frame_id} "
                        f"from {self._scheduler.inflight_lane(frame.frame_id)} | "
                        f"{self._codec} | JPEG={len(frame.image_bytes)}B "
                        f"wire={len(pending.wire)}B | "
                        f"inflight={self._scheduler.inflight_count()}/{self._scheduler.window_size}",
                    )
                    return gabriel_pb2.InputFrame(
                        payload_type=gabriel_pb2.PayloadType.IMAGE, byte_payload=pending.wire
                    )
            await asyncio.sleep(LOOP_INTERVAL_SECONDS)
        raise asyncio.CancelledError()

    # Complete an image only after checking a receipt from the active connection
    def _receive_result(self, result: Any, attempt: int) -> None:

        # Ignore callbacks belonging to a connection that has been abandoned
        if attempt != self._active_attempt:
            return

        try:
            receipt = decode_receipt(result.string_result)
            if receipt.session_id != self._session_id:
                raise ValueError("Receipt identifies a different application session")
            pending = self._pending.get(receipt.frame_id)
            if pending is None or pending.message is None or pending.attempt != attempt:
                raise ValueError("Receipt arrived without an outstanding submission")
            frame = pending.frame
            expected = pending.message

            # Match the receipt to the application identity
            same_session = receipt.session_id == expected.session_id
            same_frame = receipt.frame_id == expected.frame_id

            if not same_session or not same_frame:
                raise ValueError("Receipt identifies a different application image")

            # Match the receipt to the submitted image contents
            same_hash = receipt.sha256 == expected.sha256
            same_size = receipt.byte_count == len(expected.image_bytes)
            same_dimensions = expected.codec == "jpeg" or (
                receipt.width == expected.width and receipt.height == expected.height
            )

            if not same_hash or not same_size or not same_dimensions:
                raise ValueError("Receipt does not match the submitted image bytes")

            # Record only validated receipts, before discarding retained bytes.
            self._record("ack", pending)

            # Release retained work only after every receipt check passes
            previous_state = self._scheduler.state
            self._scheduler.acknowledge_frame(frame.frame_id)
            del self._pending[frame.frame_id]
            self.completed += 1

            self._report("ACK", f"Frame {frame.frame_id} received and retained by server")

            if previous_state != self._scheduler.state:
                self._report("STATE", f"{previous_state.value} -> {self._scheduler.state.value}")

        except (ValueError, TypeError, AttributeError) as error:
            self._receipt_error = error

    def _record(self, event: str, pending: PendingTransmission, *, at: float | None = None) -> None:
        if self._event_log is None:
            return
        self._event_log.record(
            event,
            self._session_id,
            pending.frame.frame_id,
            at=at,
            live_count=self._scheduler.live_count(),
            bank_count=self._scheduler.stored_count(),
            inflight_count=self._scheduler.inflight_count(),
            codec=self._codec,
            jpeg_bytes=len(pending.frame.image_bytes),
            wire_bytes=len(pending.wire) if pending.wire is not None else 0,
            attempt=self._active_attempt or 0,
            submitted_count=sum(p.attempt == self._active_attempt for p in self._pending.values()),
        )

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
            producer_name=GABRIEL_PRODUCER_NAME,
        )

        return WebsocketClient(
            server_endpoint=self._endpoint, input_producers=[producer], consumer=consume
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
            for pending in self._pending.values():
                if pending.submitted_at is not None:
                    if now - pending.submitted_at > RECEIPT_TIMEOUT_SECONDS:
                        raise TimeoutError(
                            f"No matching receipt for frame {pending.frame.frame_id}"
                        )

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
                for pending in self._pending.values():
                    pending.submitted_at = None
                    pending.attempt = None

            # TimeoutError and ConnectionError are also subclasses of OSError
            self._report("RETRY", f"Next connection attempt in {RECONNECT_INTERVAL_SECONDS:g}s")

            await asyncio.sleep(RECONNECT_INTERVAL_SECONDS)

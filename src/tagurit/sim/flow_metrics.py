"""
Summarize client event logs without subtracting clocks from different machines.

Queue delay ends at scheduler selection. Send-to-receipt starts at the handoff
of encoded bytes to Gabriel; it includes its send scheduling and server work.
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path


def summarize_events(path: Path) -> tuple[list[dict], dict]:
    """
    Build one row per arrival, retaining incomplete frames and retry counts.

    Parameters:
        - path (Path): CSV written by the client's EventLog

    Return: Frame rows and aggregate measurements, using milliseconds for delays
    """
    with path.open(newline="") as stream:
        events = list(csv.DictReader(stream))
    grouped = defaultdict(list)
    for event in events:
        if int(event["frame_id"]) > 0:
            grouped[(event["session_id"], int(event["frame_id"]))].append(event)
    rows = []
    for (session, frame_id), records in grouped.items():
        by_kind = defaultdict(list)
        for record in records:
            by_kind[record["event"]].append(record)
        arrival = by_kind["arrive"][0]
        selected = next(iter(by_kind["selected"]), None)
        submitted = next(iter(by_kind["submitted"]), None)
        last_submit = by_kind["submitted"][-1] if submitted else None
        ack = next(iter(by_kind["ack"]), None)

        def duration(start, end):
            if start is None or end is None:
                return None
            return 1000 * (float(end["monotonic_s"]) - float(start["monotonic_s"]))

        # Keep interrupted encoding attempts out of the measured encoding total.
        starts = {r["attempt"]: r for r in by_kind["encode_start"]}
        encode_ms = sum(duration(starts[r["attempt"]], r) for r in by_kind["encode_end"])
        rows.append(
            {
                "session_id": session,
                "frame_id": frame_id,
                "status": "acknowledged" if ack else "unfinished",
                "arrival_monotonic_s": float(arrival["monotonic_s"]),
                "first_submit_monotonic_s": float(submitted["monotonic_s"]) if submitted else None,
                "ack_monotonic_s": float(ack["monotonic_s"]) if ack else None,
                "queue_ms": duration(arrival, selected),
                "encode_ms": encode_ms if by_kind["encode_end"] else None,
                "arrival_to_submit_ms": duration(arrival, submitted),
                "send_to_receipt_ms": duration(last_submit, ack),
                "first_submit_to_receipt_ms": duration(submitted, ack),
                "arrival_to_receipt_ms": duration(arrival, ack),
                "attempts": len(by_kind["submitted"]),
                "jpeg_bytes": int(arrival["jpeg_bytes"]),
                "wire_bytes": int(submitted["wire_bytes"]) if submitted else 0,
            }
        )
    completed = [row for row in rows if row["status"] == "acknowledged"]
    clean = [row for row in completed if row["attempts"] == 1]
    elapsed = (
        (
            max(row["ack_monotonic_s"] for row in completed)
            - min(row["arrival_monotonic_s"] for row in rows)
        )
        if completed
        else 0
    )
    summary = {
        "arrived": len(rows),
        "acknowledged": len(completed),
        "unfinished": len(rows) - len(completed),
        "max_submitted_frames": max(
            (int(e.get("submitted_count") or 0) for e in events), default=0
        ),
        "retried_frames": sum(row["attempts"] > 1 for row in rows),
        "duration_s": elapsed,
        "acknowledged_frames_per_s": len(completed) / elapsed if elapsed else 0,
        "acknowledged_wire_bytes_per_s": (
            sum(row["wire_bytes"] for row in completed) / elapsed if elapsed else 0
        ),
        "jpeg_bytes": sum(row["jpeg_bytes"] for row in completed),
        "wire_bytes": sum(row["wire_bytes"] for row in completed),
        "max_waiting_frames": max(
            (int(e["live_count"] or 0) + int(e["bank_count"] or 0) for e in events), default=0
        ),
    }
    for name in ("queue_ms", "encode_ms", "send_to_receipt_ms", "arrival_to_receipt_ms"):
        # Retried responses may correspond to an earlier attempt, so report
        # unambiguous send-to-receipt percentiles only for single-attempt frames.
        sample = clean if name == "send_to_receipt_ms" else completed
        values = sorted(row[name] for row in sample if row[name] is not None)
        index = (len(values) - 1) * 0.95
        low = int(index)
        p95 = (
            (values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low))
            if values
            else None
        )
        summary[name] = {
            "count": len(values),
            "median": statistics.median(values) if values else None,
            "p95": p95,
            "max": max(values) if values else None,
        }
    return rows, summary

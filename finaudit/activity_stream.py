"""Live Research Activity Stream and Event Hub for FinAudit AI.

Provides real-time event dispatching and SSE streaming for live source transparency,
tool execution progress, document extractions, and evidence linking with full request isolation.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)


class ResearchEventHub:
    """Thread-safe event dispatcher and streaming queue manager for research sessions."""

    def __init__(self):
        self._lock = threading.Lock()
        self._queues: Dict[str, queue.Queue] = {}

    def _channel_key(self, session_id: str, request_id: str) -> str:
        return f"{session_id}::{request_id}"

    def register_stream(self, session_id: str, request_id: str) -> queue.Queue:
        """Register a queue for SSE streaming for a specific session and request ID."""
        key = self._channel_key(session_id, request_id)
        with self._lock:
            q: queue.Queue = queue.Queue(maxsize=1000)
            self._queues[key] = q
            return q

    def emit(
        self,
        event_type: str,
        session_id: str,
        request_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Emit a structured event to the registered listener if active."""
        key = self._channel_key(session_id, request_id)
        with self._lock:
            q = self._queues.get(key)
        if q is not None:
            event_obj = {
                "type": event_type,
                "session_id": session_id,
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": data,
            }
            try:
                q.put_nowait(event_obj)
            except queue.Full:
                logger.warning("Event queue full for %s, dropping event", key)

    def close_stream(self, session_id: str, request_id: str) -> None:
        """Send termination signal and remove the queue."""
        key = self._channel_key(session_id, request_id)
        with self._lock:
            q = self._queues.pop(key, None)
        if q is not None:
            try:
                q.put_nowait({"type": "done", "session_id": session_id, "request_id": request_id})
            except Exception:
                pass

    def stream_sse(
        self,
        session_id: str,
        request_id: str,
        timeout: float = 90.0,
    ) -> Generator[str, None, None]:
        """Yield Server-Sent Events formatted data chunks until 'done' or timeout."""
        q = self.register_stream(session_id, request_id)
        start_time = time.time()
        try:
            # Yield initial connection heartbeat
            yield f"data: {json.dumps({'type': 'connected', 'session_id': session_id, 'request_id': request_id})}\n\n"

            while True:
                if time.time() - start_time > timeout:
                    yield f"data: {json.dumps({'type': 'timeout', 'message': 'Research stream timed out'})}\n\n"
                    break
                try:
                    event = q.get(timeout=0.5)
                except queue.Empty:
                    # Keep-alive heartbeat ping
                    yield ": ping\n\n"
                    continue

                if event.get("type") == "done":
                    yield f"data: {json.dumps(event)}\n\n"
                    break

                yield f"data: {json.dumps(event)}\n\n"
        finally:
            self.close_stream(session_id, request_id)


_global_event_hub: Optional[ResearchEventHub] = None
_hub_lock = threading.Lock()


def get_event_hub() -> ResearchEventHub:
    """Retrieve singleton ResearchEventHub."""
    global _global_event_hub
    with _hub_lock:
        if _global_event_hub is None:
            _global_event_hub = ResearchEventHub()
        return _global_event_hub


event_hub = get_event_hub()


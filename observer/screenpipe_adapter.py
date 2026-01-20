import time
import hashlib
import requests
from typing import Dict, Optional


class ScreenpipeAdapter:
    """
    Read-only adapter for Screenpipe outputs.

    HARD GUARANTEES:
    - Never starts Screenpipe
    - Never installs Screenpipe
    - Never controls OS input
    - Never blocks the main loop
    - Never raises to caller
    """

    SCREENPIPE_URL = "http://127.0.0.1:3030/latest"  # standard local endpoint
    STALE_AFTER_SECONDS = 5.0
    REQUEST_TIMEOUT = 0.3  # non-blocking

    def __init__(self):
        self.last_read_ts: Optional[float] = None
        self.last_frame_ts: Optional[float] = None

        self.state: Dict[str, object] = {
            "available": False,
            "frame_ts": None,
            "screen_text_hash": None,
            "stale": True,
        }

        print("[SCREENPIPE] Adapter initialized (read-only)")

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(
            text.encode("utf-8", errors="ignore")
        ).hexdigest()

    def read(self) -> Dict[str, object]:
        """
        Read latest observation snapshot from Screenpipe if available.
        """
        try:
            now = time.time()
            self.last_read_ts = now

            resp = requests.get(
                self.SCREENPIPE_URL,
                timeout=self.REQUEST_TIMEOUT,
            )

            if resp.status_code != 200:
                raise RuntimeError("Screenpipe not ready")

            payload = resp.json()

            text = payload.get("text", "")
            frame_ts = payload.get("timestamp")

            if not frame_ts:
                raise RuntimeError("No frame timestamp")

            self.last_frame_ts = frame_ts

            self.state.update(
                {
                    "available": True,
                    "frame_ts": frame_ts,
                    "screen_text_hash": self._hash_text(text),
                    "stale": (now - frame_ts) > self.STALE_AFTER_SECONDS,
                }
            )

        except Exception:
            # Honest degradation
            self.state.update(
                {
                    "available": False,
                    "frame_ts": None,
                    "screen_text_hash": None,
                    "stale": True,
                }
            )

        return dict(self.state)

    def is_available(self) -> bool:
        return bool(self.state.get("available", False))

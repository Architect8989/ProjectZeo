import time
import hashlib
from typing import Dict, Optional


class ScreenpipeAdapter:
    """
    Read-only adapter for Screenpipe outputs.

    HARD GUARANTEES:
    - Never starts Screenpipe
    - Never controls Screenpipe
    - Never touches OS input
    - Never blocks the main loop
    - Never raises to caller
    """

    STALE_AFTER_SECONDS = 5.0

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
        return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

    def read(self) -> Dict[str, object]:
        """
        Read latest observation snapshot.

        NOTE:
        Real Screenpipe integration will replace the placeholder section.
        """
        try:
            now = time.time()

            # ---- PLACEHOLDER (Phase-2 compliant) ----
            # No live Screenpipe feed yet; this simulates "no data"
            text = ""
            frame_ts = None
            # -----------------------------------------

            if frame_ts is None:
                self.state.update(
                    {
                        "available": False,
                        "frame_ts": None,
                        "screen_text_hash": None,
                        "stale": True,
                    }
                )
            else:
                self.state.update(
                    {
                        "available": True,
                        "frame_ts": frame_ts,
                        "screen_text_hash": self._hash_text(text),
                        "stale": (now - frame_ts) > self.STALE_AFTER_SECONDS,
                    }
                )

            self.last_read_ts = now
            self.last_frame_ts = frame_ts

        except Exception:
            # Absolute containment: observer must survive everything
            self.state["available"] = False
            self.state["stale"] = True

        return dict(self.state)

    def is_available(self) -> bool:
        return bool(self.state.get("available", False))

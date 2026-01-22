import time
import hashlib
import requests
from typing import Dict, Optional


class ScreenpipeBlindnessError(RuntimeError):
    """Raised when screen input is provably blind or frozen."""


class ScreenpipeAdapter:
    """
    Read-only adapter for Screenpipe outputs.

    HARD GUARANTEES:
    - Never starts Screenpipe
    - Never installs Screenpipe
    - Never controls OS input
    - Never blocks the main loop
    """

    SCREENPIPE_URL = "http://127.0.0.1:3030/latest"
    REQUEST_TIMEOUT = 0.3

    MAX_FRAME_AGE_SECONDS = 0.5
    MAX_SAME_HASH_FRAMES = 10

    def __init__(self):
        self.last_read_mono: Optional[float] = None
        self.last_frame_ts: Optional[float] = None
        self.last_hash: Optional[str] = None
        self.same_hash_count = 0
        self.blind = False

        self.state: Dict[str, object] = {
            "available": False,
            "frame_ts": None,
            "screen_text_hash": None,
            "stale": True,
            "blind": False,
        }

        print("[SCREENPIPE] Adapter initialized (read-only)")

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(
            text.encode("utf-8", errors="ignore")
        ).hexdigest()

    def read(self) -> Dict[str, object]:
        if self.blind:
            raise ScreenpipeBlindnessError("Screenpipe already marked blind")

        try:
            now_mono = time.monotonic()
            now_wall = time.time()
            self.last_read_mono = now_mono

            resp = requests.get(
                self.SCREENPIPE_URL,
                timeout=self.REQUEST_TIMEOUT,
            )

            if resp.status_code != 200:
                raise RuntimeError("Screenpipe HTTP failure")

            payload = resp.json()

            frame_ts = payload.get("timestamp")
            text = payload.get("text", "")

            if frame_ts is None:
                raise RuntimeError("Missing frame timestamp")

            age = now_wall - frame_ts
            if age > self.MAX_FRAME_AGE_SECONDS:
                raise ScreenpipeBlindnessError("Frame too old")

            text_hash = self._hash_text(text)

            if text_hash == self.last_hash:
                self.same_hash_count += 1
                if self.same_hash_count >= self.MAX_SAME_HASH_FRAMES:
                    raise ScreenpipeBlindnessError("Frozen screen detected")
            else:
                self.same_hash_count = 0

            self.last_hash = text_hash
            self.last_frame_ts = frame_ts

            self.state.update(
                {
                    "available": True,
                    "frame_ts": frame_ts,
                    "screen_text_hash": text_hash,
                    "stale": False,
                    "blind": False,
                }
            )

        except ScreenpipeBlindnessError:
            self.blind = True
            self.state.update(
                {
                    "available": False,
                    "frame_ts": None,
                    "screen_text_hash": None,
                    "stale": True,
                    "blind": True,
                }
            )
            raise

        except Exception:
            self.state.update(
                {
                    "available": False,
                    "frame_ts": None,
                    "screen_text_hash": None,
                    "stale": True,
                    "blind": False,
                }
            )

        return dict(self.state)

    def is_available(self) -> bool:
        return bool(self.state.get("available", False))

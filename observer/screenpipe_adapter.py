import time
import hashlib
import requests
import threading
from typing import Dict, Optional


class ScreenpipeBlindnessError(RuntimeError):
    """Raised when screen capture is provably unavailable."""


class ScreenpipeAdapter:
    """
    Read-only adapter for Screenpipe outputs.

    Guarantees:
    - Static screens are valid
    - Blindness only on capture failure
    - Timestamp sanity enforced
    """

    SCREENPIPE_URL = "http://127.0.0.1:3030/latest"
    REQUEST_TIMEOUT = 0.3

    MAX_FRAME_AGE_SECONDS = 0.5

    def __init__(self):
        self.last_read_mono: Optional[float] = None
        self.last_frame_ts: Optional[float] = None
        self.last_hash: Optional[str] = None

        self.first_seen_mono: Optional[float] = None
        self.frame_counter = 0

        self.blind = False
        self.blind_reason: Optional[str] = None

        self._lock = threading.Lock()

        self._state: Dict[str, object] = {
            "available": False,
            "frame_ts": None,
            "screen_text_hash": None,
            "stale": True,
            "blind": False,
        }

        print("[SCREENPIPE] Adapter initialized (read-only)")

    # -------------------------------------------------

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(
            text.encode("utf-8", errors="ignore")
        ).hexdigest()

    def _normalize_timestamp(self, raw_ts: float) -> float:
        ts = float(raw_ts)
        # milliseconds → seconds
        if ts > 1e12:
            ts /= 1000.0
        return ts

    def _mark_blind(self, reason: str) -> None:
        with self._lock:
            if self.blind:
                return
            self.blind = True
            self.blind_reason = reason
            self._state = {
                "available": False,
                "frame_ts": None,
                "screen_text_hash": None,
                "stale": True,
                "blind": True,
            }

    # -------------------------------------------------

    def read(self) -> Dict[str, object]:
        with self._lock:
            if self.blind:
                raise ScreenpipeBlindnessError(
                    f"Screenpipe blind: {self.blind_reason}"
                )

        # ---- fetch (no lock) ----
        try:
            resp = requests.get(
                self.SCREENPIPE_URL,
                timeout=self.REQUEST_TIMEOUT,
            )
        except Exception:
            return self._mark_unavailable()

        # ---- parse ----
        try:
            if resp.status_code != 200:
                return self._mark_unavailable()

            payload = resp.json()
            raw_ts = payload.get("timestamp")
            text = payload.get("text", "")

            if raw_ts is None:
                return self._mark_unavailable()

            frame_ts = self._normalize_timestamp(raw_ts)

            now_wall = time.time()
            age = now_wall - frame_ts

            # Reject future or stale frames strictly
            if age < 0 or age > self.MAX_FRAME_AGE_SECONDS:
                return self._mark_unavailable()

            text_hash = self._hash_text(text)

            now_mono = time.monotonic()

            with self._lock:
                if self.first_seen_mono is None:
                    self.first_seen_mono = now_mono

                self.last_read_mono = now_mono
                self.last_frame_ts = frame_ts
                self.last_hash = text_hash
                self.frame_counter += 1

                self._state = {
                    "available": True,
                    "frame_ts": frame_ts,
                    "screen_text_hash": text_hash,
                    "stale": False,
                    "blind": False,
                }

                return dict(self._state)

        except ScreenpipeBlindnessError:
            raise

        except Exception:
            return self._mark_unavailable()

    # -------------------------------------------------

    def _mark_unavailable(self) -> Dict[str, object]:
        with self._lock:
            self._state = {
                "available": False,
                "frame_ts": None,
                "screen_text_hash": None,
                "stale": True,
                "blind": False,
            }
            return dict(self._state)

    # -------------------------------------------------

    def is_available(self) -> bool:
        with self._lock:
            return bool(self._state.get("available"))

    def get_health_snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "blind": self.blind,
                "blind_reason": self.blind_reason,
                "frame_counter": self.frame_counter,
                "last_frame_ts": self.last_frame_ts,
                "last_read_mono": self.last_read_mono,
                "uptime_seconds": (
                    time.monotonic() - self.first_seen_mono
                    if self.first_seen_mono
                    else None
                ),
            }

    def self_test(self) -> bool:
        try:
            self.read()
            return True
        except Exception:
            return False

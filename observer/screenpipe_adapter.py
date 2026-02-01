import time
import hashlib
import requests
import threading
from typing import Dict, Optional


class ScreenpipeBlindnessError(RuntimeError):
    """Raised when screen input is provably blind or frozen."""


class ScreenpipeAdapter:
    """
    Read-only adapter for Screenpipe outputs.
    """

    SCREENPIPE_URL = "http://127.0.0.1:3030/latest"
    REQUEST_TIMEOUT = 0.3

    MAX_FRAME_AGE_SECONDS = 0.5
    MAX_SAME_HASH_FRAMES = 10
    MAX_HASH_STALL_SECONDS = 1.5

    def __init__(self):
        self.last_read_mono: Optional[float] = None
        self.last_frame_ts: Optional[float] = None
        self.last_hash: Optional[str] = None
        self.same_hash_count = 0

        self.first_seen_mono: Optional[float] = None
        self.last_change_mono: Optional[float] = None
        self.frame_counter = 0

        self.blind = False
        self.blind_reason: Optional[str] = None

        self._lock = threading.Lock()

        self.state: Dict[str, object] = {
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

    def _normalize_timestamp(self, frame_ts: float) -> float:
        """
        Normalize timestamp to seconds since epoch.
        Handles millisecond timestamps safely.
        """
        ts = float(frame_ts)

        # Heuristic: > year 33658 in seconds → milliseconds
        if ts > 1e12:
            ts /= 1000.0

        return ts

    def _mark_blind(self, reason: str) -> None:
        with self._lock:
            if self.blind:
                return
            self.blind = True
            self.blind_reason = reason
            self.state = {
                "available": False,
                "frame_ts": None,
                "screen_text_hash": None,
                "stale": True,
                "blind": True,
            }

    # -------------------------------------------------

    def read(self) -> Dict[str, object]:
        # ---- fast pre-check ----
        with self._lock:
            if self.blind:
                raise ScreenpipeBlindnessError(
                    f"Screenpipe marked blind: {self.blind_reason}"
                )

        # ---- network call (NO LOCK) ----
        try:
            resp = requests.get(
                self.SCREENPIPE_URL,
                timeout=self.REQUEST_TIMEOUT,
            )
        except Exception:
            with self._lock:
                self.state = {
                    "available": False,
                    "frame_ts": None,
                    "screen_text_hash": None,
                    "stale": True,
                    "blind": False,
                }
                return dict(self.state)

        # ---- response processing ----
        try:
            if resp.status_code != 200:
                raise RuntimeError("Screenpipe HTTP failure")

            payload = resp.json()
            raw_ts = payload.get("timestamp")
            text = payload.get("text", "")

            if raw_ts is None:
                raise RuntimeError("Missing frame timestamp")

            now_mono = time.monotonic()
            now_wall = time.time()

            frame_ts = self._normalize_timestamp(raw_ts)
            age = now_wall - frame_ts

            # Reject timestamps from the future or far past
            if (
                age > self.MAX_FRAME_AGE_SECONDS
                or age < -self.MAX_FRAME_AGE_SECONDS
            ):
                raise ScreenpipeBlindnessError(
                    f"Invalid frame timestamp (age={age:.2f}s)"
                )

            text_hash = self._hash_text(text)

            with self._lock:
                self.last_read_mono = now_mono

                if self.first_seen_mono is None:
                    self.first_seen_mono = now_mono

                if text_hash == self.last_hash:
                    self.same_hash_count += 1
                    if self.last_change_mono is None:
                        self.last_change_mono = now_mono

                    stall = now_mono - self.last_change_mono
                    if (
                        self.same_hash_count >= self.MAX_SAME_HASH_FRAMES
                        or stall >= self.MAX_HASH_STALL_SECONDS
                    ):
                        raise ScreenpipeBlindnessError(
                            "Frozen screen detected"
                        )
                else:
                    self.same_hash_count = 0
                    self.last_change_mono = now_mono

                self.last_hash = text_hash
                self.last_frame_ts = frame_ts
                self.frame_counter += 1

                self.state = {
                    "available": True,
                    "frame_ts": frame_ts,
                    "screen_text_hash": text_hash,
                    "stale": False,
                    "blind": False,
                }

                return dict(self.state)

        except ScreenpipeBlindnessError as e:
            self._mark_blind(str(e))
            raise

        except Exception:
            with self._lock:
                self.state = {
                    "available": False,
                    "frame_ts": None,
                    "screen_text_hash": None,
                    "stale": True,
                    "blind": False,
                }
                return dict(self.state)

    # -------------------------------------------------

    def is_available(self) -> bool:
        with self._lock:
            return bool(self.state.get("available"))

    def get_health_snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "blind": self.blind,
                "blind_reason": self.blind_reason,
                "frame_counter": self.frame_counter,
                "last_frame_ts": self.last_frame_ts,
                "last_read_mono": self.last_read_mono,
                "same_hash_count": self.same_hash_count,
                "uptime_seconds": (
                    time.monotonic() - self.first_seen_mono
                    if self.first_seen_mono
                    else None
                ),
            }

    def self_test(self) -> bool:
        try:
            _ = self.read()
            return True
        except Exception:
            return False

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

    Properties:
    - No permanent blindness
    - Failure reasons preserved
    - Bounded staleness tolerance
    - Deterministic hashing
    """

    SCREENPIPE_URL = "http://127.0.0.1:3030/latest"
    REQUEST_TIMEOUT = 0.3

    MAX_FRAME_AGE_SECONDS = 1.5        # relaxed (audit fix)
    MAX_CONSECUTIVE_FAILURES = 5       # before blindness

    def __init__(self):
        self.last_read_mono: Optional[float] = None
        self.last_frame_ts: Optional[float] = None
        self.last_hash: Optional[str] = None

        self.first_seen_mono: Optional[float] = None
        self.frame_counter = 0

        self.failure_count = 0
        self.blind = False
        self.blind_reason: Optional[str] = None

        self._lock = threading.Lock()

        self._state: Dict[str, object] = {
            "available": False,
            "frame_ts": None,
            "screen_hash": None,
            "stale": True,
            "blind": False,
            "reason": None,
        }

        print("[SCREENPIPE] Adapter initialized")

        # -------------------------------------------------
        # Check if Screenpipe service is available on initialization
        if not self._is_screenpipe_running():
            raise ScreenpipeBlindnessError("Screenpipe service is not available.")

    # -------------------------------------------------
    def _is_screenpipe_running(self) -> bool:
        """Checks if the Screenpipe service is running."""
        try:
            response = requests.get(self.SCREENPIPE_URL, timeout=self.REQUEST_TIMEOUT)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            return False
        return False

    def _hash_payload(self, text: str, ts: float) -> str:
        h = hashlib.sha256()
        h.update(text.encode("utf-8", errors="ignore"))
        h.update(str(int(ts)).encode())
        return h.hexdigest()

    def _normalize_timestamp(self, raw_ts: float) -> float:
        ts = float(raw_ts)
        if ts > 1e12:  # ms → sec
            ts /= 1000.0
        return ts

    def _set_state(
        self,
        *,
        available: bool,
        frame_ts=None,
        screen_hash=None,
        stale=True,
        blind=False,
        reason=None,
    ):
        self._state = {
            "available": available,
            "frame_ts": frame_ts,
            "screen_hash": screen_hash,
            "stale": stale,
            "blind": blind,
            "reason": reason,
        }
        return dict(self._state)

    # -------------------------------------------------

    def read(self) -> Dict[str, object]:
        with self._lock:
            if self.blind:
                raise ScreenpipeBlindnessError(
                    f"Screenpipe blind: {self.blind_reason}"
                )

        try:
            resp = requests.get(
                self.SCREENPIPE_URL,
                timeout=self.REQUEST_TIMEOUT,
            )
        except Exception as e:
            return self._mark_failure(f"network_error:{e}")

        if resp.status_code != 200:
            return self._mark_failure(f"http_{resp.status_code}")

        try:
            payload = resp.json()
        except Exception:
            return self._mark_failure("invalid_json")

        raw_ts = payload.get("timestamp")
        text = payload.get("text", "")

        if raw_ts is None:
            return self._mark_failure("missing_timestamp")

        try:
            frame_ts = self._normalize_timestamp(raw_ts)
        except Exception:
            return self._mark_failure("bad_timestamp")

        age = time.time() - frame_ts
        if age < -0.1 or age > self.MAX_FRAME_AGE_SECONDS:
            return self._mark_failure(f"stale_frame:{age:.2f}s")

        screen_hash = self._hash_payload(text, frame_ts)
        now_mono = time.monotonic()

        with self._lock:
            if self.first_seen_mono is None:
                self.first_seen_mono = now_mono

            self.last_read_mono = now_mono
            self.last_frame_ts = frame_ts
            self.last_hash = screen_hash
            self.frame_counter += 1
            self.failure_count = 0

            return self._set_state(
                available=True,
                frame_ts=frame_ts,
                screen_hash=screen_hash,
                stale=False,
                blind=False,
            )

    # -------------------------------------------------

    def _mark_failure(self, reason: str) -> Dict[str, object]:
        with self._lock:
            self.failure_count += 1

            if self.failure_count >= self.MAX_CONSECUTIVE_FAILURES:
                self.blind = True
                self.blind_reason = reason

            return self._set_state(
                available=False,
                stale=True,
                blind=self.blind,
                reason=reason,
            )

    # -------------------------------------------------

    def is_available(self) -> bool:
        with self._lock:
            return bool(self._state.get("available"))

    def get_health_snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "blind": self.blind,
                "blind_reason": self.blind_reason,
                "failure_count": self.failure_count,
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
            state = self.read()
            return bool(state.get("available"))
        except Exception:
            return False

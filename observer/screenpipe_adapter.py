import time
import hashlib
import requests
import threading
from typing import Dict, Optional
from enum import Enum


class ScreenpipeBlindnessError(RuntimeError):
    """Raised when screen capture is unavailable."""


class ScreenpipeState(Enum):
    INIT = "init"
    READY = "ready"
    TEMP_UNAVAILABLE = "temporary_unavailable"
    FATAL = "fatal"


class ScreenpipeAdapter:
    """
    Authoritative, read-only adapter for Screenpipe.

    HARD GUARANTEES:
    - read() returns a valid frame OR raises ScreenpipeBlindnessError
    - Temporary failures are recoverable
    - Fatal blindness is explicit and terminal
    - Adapter owns all readiness & health logic
    """

    SCREENPIPE_URL = "http://127.0.0.1:3030/latest"
    REQUEST_TIMEOUT = 0.3

    MAX_FRAME_AGE_SECONDS = 1.5
    MAX_CONSECUTIVE_FAILURES = 5
    RECOVERY_COOLDOWN_SECONDS = 2.0

    # -------------------------------------------------

    def __init__(self):
        self._lock = threading.Lock()

        self.state = ScreenpipeState.INIT

        self.last_read_mono: Optional[float] = None
        self.last_frame_ts: Optional[float] = None
        self.last_hash: Optional[str] = None

        self.first_seen_mono: Optional[float] = None
        self.frame_counter = 0

        self.failure_count = 0
        self.last_failure_mono: Optional[float] = None
        self.blind_reason: Optional[str] = None

        self._last_snapshot: Dict[str, object] = {
            "available": False,
            "frame_ts": None,
            "screen_hash": None,
            "text": "",
        }

        print("[SCREENPIPE] Adapter initialized")

    # =================================================
    # Internal helpers
    # =================================================

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

    # =================================================
    # Failure & recovery logic
    # =================================================

    def _record_failure(self, reason: str) -> None:
        self.failure_count += 1
        self.last_failure_mono = time.monotonic()
        self.blind_reason = reason

        if self.failure_count >= self.MAX_CONSECUTIVE_FAILURES:
            self.state = ScreenpipeState.TEMP_UNAVAILABLE

    def _maybe_recover(self) -> None:
        if self.state != ScreenpipeState.TEMP_UNAVAILABLE:
            return

        if self.last_failure_mono is None:
            return

        if (
            time.monotonic() - self.last_failure_mono
            < self.RECOVERY_COOLDOWN_SECONDS
        ):
            return

        # allow retry
        self.failure_count = 0
        self.blind_reason = None
        self.state = ScreenpipeState.INIT

    # =================================================
    # Public API
    # =================================================

    def read(self) -> Dict[str, object]:
        with self._lock:
            if self.state == ScreenpipeState.FATAL:
                raise ScreenpipeBlindnessError(
                    "Screenpipe entered fatal state"
                )

            if self.state == ScreenpipeState.TEMP_UNAVAILABLE:
                self._maybe_recover()
                if self.state == ScreenpipeState.TEMP_UNAVAILABLE:
                    raise ScreenpipeBlindnessError(
                        f"Screenpipe temporarily unavailable: {self.blind_reason}"
                    )

        # -------------------------------------------------
        # Attempt read (no lock held)
        try:
            resp = requests.get(
                self.SCREENPIPE_URL,
                timeout=self.REQUEST_TIMEOUT,
            )
        except Exception as e:
            with self._lock:
                self._record_failure(f"network_error:{e}")
            raise ScreenpipeBlindnessError("Screenpipe network failure")

        if resp.status_code != 200:
            with self._lock:
                self._record_failure(f"http_{resp.status_code}")
            raise ScreenpipeBlindnessError(
                f"Screenpipe HTTP {resp.status_code}"
            )

        try:
            payload = resp.json()
        except Exception:
            with self._lock:
                self._record_failure("invalid_json")
            raise ScreenpipeBlindnessError("Invalid Screenpipe JSON")

        raw_ts = payload.get("timestamp")
        text = payload.get("text", "")

        if raw_ts is None:
            with self._lock:
                self._record_failure("missing_timestamp")
            raise ScreenpipeBlindnessError("Missing frame timestamp")

        try:
            frame_ts = self._normalize_timestamp(raw_ts)
        except Exception:
            with self._lock:
                self._record_failure("bad_timestamp")
            raise ScreenpipeBlindnessError("Bad frame timestamp")

        age = time.time() - frame_ts
        if age < -0.1 or age > self.MAX_FRAME_AGE_SECONDS:
            with self._lock:
                self._record_failure(f"stale_frame:{age:.2f}s")
            raise ScreenpipeBlindnessError("Stale screen frame")

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
            self.blind_reason = None
            self.state = ScreenpipeState.READY

            self._last_snapshot = {
                "available": True,
                "frame_ts": frame_ts,
                "screen_hash": screen_hash,
                "text": text,
            }

            return dict(self._last_snapshot)

    # =================================================
    # Introspection
    # =================================================

    def is_available(self) -> bool:
        with self._lock:
            return self.state == ScreenpipeState.READY

    def get_health_snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "state": self.state.value,
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
            frame = self.read()
            return bool(frame.get("available"))
        except ScreenpipeBlindnessError:
            return False

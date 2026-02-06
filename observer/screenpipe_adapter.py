import time
import hashlib
import requests
import threading
from typing import Dict, Optional
from enum import Enum


class ScreenpipeBlindnessError(RuntimeError):
    """Raised when screen capture is temporarily unavailable."""


class ScreenpipeState(Enum):
    INIT = "init"
    READY = "ready"
    TEMP_UNAVAILABLE = "temporary_unavailable"
    FATAL = "fatal"


class ScreenpipeAdapter:
    """
    Authoritative, read-only adapter for Screenpipe.

    HARD GUARANTEES:
    - No permanent blindness unless explicitly fatal
    - Temporary failures are recoverable
    - read() is deterministic: valid frame OR typed exception
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

        self._state_snapshot: Dict[str, object] = {
            "available": False,
            "frame_ts": None,
            "screen_text_hash": None,
            "stale": True,
            "blind": False,
            "reason": None,
        }

        print("[SCREENPIPE] Adapter initialized (non-fatal init)")

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

    def _set_snapshot(
        self,
        *,
        available: bool,
        frame_ts=None,
        screen_text_hash=None,
        stale=True,
        blind=False,
        reason=None,
    ) -> Dict[str, object]:
        self._state_snapshot = {
            "available": available,
            "frame_ts": frame_ts,
            "screen_text_hash": screen_text_hash,
            "stale": stale,
            "blind": blind,
            "reason": reason,
        }
        return dict(self._state_snapshot)

    # =================================================
    # Failure & recovery logic
    # =================================================

    def _mark_failure(self, reason: str) -> Dict[str, object]:
        with self._lock:
            self.failure_count += 1
            self.last_failure_mono = time.monotonic()
            self.blind_reason = reason

            if self.failure_count >= self.MAX_CONSECUTIVE_FAILURES:
                self.state = ScreenpipeState.TEMP_UNAVAILABLE

            return self._set_snapshot(
                available=False,
                stale=True,
                blind=self.state != ScreenpipeState.READY,
                reason=reason,
            )

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
                raise ScreenpipeBlindnessError("Screenpipe entered fatal state")

            if self.state == ScreenpipeState.TEMP_UNAVAILABLE:
                self._maybe_recover()
                if self.state == ScreenpipeState.TEMP_UNAVAILABLE:
                    raise ScreenpipeBlindnessError(
                        f"Screenpipe temporarily unavailable: {self.blind_reason}"
                    )

        # -------------------------------------------------
        # Attempt read
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
            self.blind_reason = None
            self.state = ScreenpipeState.READY

            return self._set_snapshot(
                available=True,
                frame_ts=frame_ts,
                screen_text_hash=screen_hash,
                stale=False,
                blind=False,
            )

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
            state = self.read()
            return bool(state.get("available"))
        except ScreenpipeBlindnessError:
            return False

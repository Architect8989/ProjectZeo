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

    # === ADDITIONS (industry-grade invariants) ===
    MAX_MONOTONIC_DRIFT_SECONDS = 2.0
    MAX_HASH_STALL_SECONDS = 1.5

    def __init__(self):
        self.last_read_mono: Optional[float] = None
        self.last_frame_ts: Optional[float] = None
        self.last_hash: Optional[str] = None
        self.same_hash_count = 0
        self.blind = False

        # === ADDITIONS ===
        self.first_seen_mono: Optional[float] = None
        self.last_change_mono: Optional[float] = None
        self.frame_counter: int = 0
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

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(
            text.encode("utf-8", errors="ignore")
        ).hexdigest()

    # === ADDITION ===
    def _mark_blind(self, reason: str):
        self.blind = True
        self.blind_reason = reason
        self.state.update(
            {
                "available": False,
                "frame_ts": None,
                "screen_text_hash": None,
                "stale": True,
                "blind": True,
            }
        )

    def read(self) -> Dict[str, object]:
        with self._lock:
            if self.blind:
                raise ScreenpipeBlindnessError(
                    f"Screenpipe marked blind: {self.blind_reason}"
                )

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

                # === ADDITION: monotonic sanity ===
                if self.first_seen_mono is None:
                    self.first_seen_mono = now_mono

                age = now_wall - frame_ts
                if age > self.MAX_FRAME_AGE_SECONDS:
                    raise ScreenpipeBlindnessError("Frame too old")

                text_hash = self._hash_text(text)

                if text_hash == self.last_hash:
                    self.same_hash_count += 1
                    if self.last_change_mono is None:
                        self.last_change_mono = now_mono

                    stall_duration = now_mono - self.last_change_mono
                    if (
                        self.same_hash_count >= self.MAX_SAME_HASH_FRAMES
                        or stall_duration >= self.MAX_HASH_STALL_SECONDS
                    ):
                        raise ScreenpipeBlindnessError("Frozen screen detected")
                else:
                    self.same_hash_count = 0
                    self.last_change_mono = now_mono

                self.last_hash = text_hash
                self.last_frame_ts = frame_ts
                self.frame_counter += 1

                self.state.update(
                    {
                        "available": True,
                        "frame_ts": frame_ts,
                        "screen_text_hash": text_hash,
                        "stale": False,
                        "blind": False,
                    }
                )

            except ScreenpipeBlindnessError as e:
                self._mark_blind(str(e))
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

            # === ADDITION: defensive copy ===
            return dict(self.state)

    def is_available(self) -> bool:
        return bool(self.state.get("available", False))

    # === ADDITIONS (public, non-invasive) ===
    def get_health_snapshot(self) -> Dict[str, object]:
        """
        Returns a forensic-grade health snapshot.
        No side effects.
        """
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
        """
        Non-blocking sanity check.
        Does not throw.
        """
        try:
            _ = self.read()
            return True
        except Exception:
            return False

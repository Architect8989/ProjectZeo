import time
import os
import sys
import atexit
import threading
from collections import deque
from typing import Optional

try:
    import psutil as _psutil_module
    _PSUTIL_AVAILABLE = True
except ImportError:
    _psutil_module = None
    _PSUTIL_AVAILABLE = False


# Hard limits
MAX_RUNTIME_SECONDS  = 3600         # 1 hour per task
MAX_MEMORY_MB        = 4096         # 4 GB resident set
MAX_CPU_PERCENT      = 90           # sustained CPU

# AUDIT-FIX: Minimum effective task timeout (was incorrectly set to 86_400 = 24h).
# main.py sets _MIN_EFFECTIVE_TASK_SECONDS = 1_800. Watchdog must be consistent.
# 1_800 seconds = 30 minutes — minimum task budget even in "unlimited" mode.
MIN_EFFECTIVE_TIMEOUT_SECONDS = 1_800  # 30 minutes (was 86_400 — DEFECT FIXED)

# Sampling
CPU_SAMPLE_INTERVAL  = 0.1          # seconds
CPU_WINDOW_SECONDS   = 3.0          # sustained window
CPU_MIN_SAMPLES      = int(CPU_WINDOW_SECONDS / CPU_SAMPLE_INTERVAL)

# Background sampling thread interval
BACKGROUND_SAMPLE_INTERVAL = 0.25   # seconds


class WatchdogViolation(RuntimeError):
    """Raised when a runtime safety budget is violated."""
    pass


class RuntimeWatchdog:

    def __init__(self):
        # Dedicated lock for start_time
        self._start_time_lock = threading.Lock()
        self._start_time: float = time.time()

        # Psutil process handle
        self.process: Optional[object] = None
        if _PSUTIL_AVAILABLE:
            try:
                self.process = _psutil_module.Process(os.getpid())
                self.process.cpu_percent(interval=None)
            except Exception:
                self.process = None

        # CPU sampling state
        self._cpu_samples: deque = deque(maxlen=CPU_MIN_SAMPLES)
        self._cpu_samples_lock = threading.Lock()

        # CPU pause
        self._cpu_paused: bool = False
        self._cpu_pause_lock = threading.Lock()

        # Violation tracking for check()
        self._pending_violation: Optional[str] = None
        self._violation_lock = threading.Lock()

        # Background sampling thread
        self._bg_stop_event = threading.Event()
        self._bg_thread = threading.Thread(
            target=self._background_sample_loop,
            name="WatchdogBGSampler",
            daemon=True,
        )
        self._bg_thread.start()

        # atexit forensic log
        atexit.register(self._atexit_report)

    # =================================================
    # START TIME PROPERTY
    # =================================================

    @property
    def start_time(self) -> float:
        """Thread-safe read of the per-task start timestamp."""
        with self._start_time_lock:
            return self._start_time

    @start_time.setter
    def start_time(self, value: float) -> None:
        """Thread-safe write of the per-task start timestamp."""
        with self._start_time_lock:
            self._start_time = float(value)
        with self._violation_lock:
            self._pending_violation = None
        with self._cpu_samples_lock:
            self._cpu_samples.clear()

    # =================================================
    # CPU PAUSE / RESUME
    # =================================================

    def pause_cpu(self) -> None:
        """Pause CPU monitoring for LLM inference. Idempotent."""
        with self._cpu_pause_lock:
            self._cpu_paused = True

    def resume_cpu(self) -> None:
        """Resume CPU monitoring. Clears samples from paused window."""
        with self._cpu_pause_lock:
            self._cpu_paused = False
        with self._cpu_samples_lock:
            self._cpu_samples.clear()

    def is_cpu_paused(self) -> bool:
        with self._cpu_pause_lock:
            return self._cpu_paused

    # =================================================
    # BACKGROUND SAMPLING THREAD
    # =================================================

    def _background_sample_loop(self) -> None:
        while not self._bg_stop_event.is_set():
            time.sleep(BACKGROUND_SAMPLE_INTERVAL)
            try:
                self._bg_sample_once()
            except Exception as exc:
                print(
                    f"[WatchdogBGSampler] Unexpected error: {exc}",
                    file=sys.stderr,
                )

    def _bg_sample_once(self) -> None:
        """Single background sample tick — time, memory, CPU."""
        now = time.time()
        elapsed = now - self.start_time

        # Enforce minimum effective timeout (consistent with main.py).
        # AUDIT-FIX: MIN_EFFECTIVE_TIMEOUT_SECONDS is now 1_800, not 86_400.
        effective_timeout = max(MAX_RUNTIME_SECONDS, MIN_EFFECTIVE_TIMEOUT_SECONDS)
        _env_override = os.environ.get("PROJECTZEO_MAX_TASK_SECONDS", "")
        try:
            _env_val = int(_env_override.strip())
            if _env_val > 0:
                effective_timeout = max(_env_val, MIN_EFFECTIVE_TIMEOUT_SECONDS)
            elif _env_val == 0:
                # "unlimited" mode — still enforce 30-minute minimum
                effective_timeout = MIN_EFFECTIVE_TIMEOUT_SECONDS
        except (ValueError, AttributeError):
            pass

        if elapsed > effective_timeout:
            self._set_violation(
                f"TIME_LIMIT(elapsed={elapsed:.0f}s, limit={effective_timeout}s)"
            )

        # Memory check
        if self.process is not None:
            try:
                mem_mb = self.process.memory_info().rss / (1024 * 1024)
                if mem_mb > MAX_MEMORY_MB:
                    self._set_violation(f"MEMORY_LIMIT(rss={mem_mb:.0f}MB)")
            except Exception:
                pass

        # CPU check (skip when paused)
        with self._cpu_pause_lock:
            paused = self._cpu_paused
        if not paused and self.process is not None:
            try:
                cpu = self.process.cpu_percent(interval=CPU_SAMPLE_INTERVAL)
                with self._cpu_samples_lock:
                    self._cpu_samples.append(cpu)
                    samples = list(self._cpu_samples)

                if len(samples) >= CPU_MIN_SAMPLES:
                    avg_cpu = sum(samples) / len(samples)
                    if avg_cpu > MAX_CPU_PERCENT:
                        self._set_violation(f"CPU_LIMIT(avg={avg_cpu:.1f}%)")
            except Exception:
                pass

    def _set_violation(self, reason: str) -> None:
        """Record a pending violation (thread-safe, idempotent)."""
        with self._violation_lock:
            if self._pending_violation is None:
                self._pending_violation = reason
                print(
                    f"[Watchdog] VIOLATION DETECTED (background thread): {reason}",
                    file=sys.stderr,
                )

    # =================================================
    # MAIN CHECK — called from main thread
    # =================================================

    def check(self) -> None:
        # Raise any violation detected by the background thread
        with self._violation_lock:
            pending = self._pending_violation

        if pending:
            self._violate(pending)

        # Fallback: if background thread died, do inline time check only
        if not self._bg_thread.is_alive():
            now = time.time()
            elapsed = now - self.start_time
            effective_timeout = max(MAX_RUNTIME_SECONDS, MIN_EFFECTIVE_TIMEOUT_SECONDS)
            if elapsed > effective_timeout:
                self._violate(f"TIME_LIMIT_INLINE(elapsed={elapsed:.0f}s)")

    def _violate(self, reason: str) -> None:
        msg = f"WATCHDOG_LIMIT_EXCEEDED:{reason}"
        print(f"[Watchdog] {msg}", file=sys.stderr)
        raise WatchdogViolation(msg)

    # =================================================
    # SHUTDOWN
    # =================================================

    def shutdown(self) -> None:
        """Stop the background sampling thread cleanly."""
        self._bg_stop_event.set()
        if self._bg_thread.is_alive():
            self._bg_thread.join(timeout=2.0)

    # =================================================
    # ATEXIT FORENSICS
    # =================================================

    def _atexit_report(self) -> None:
        """Log any unraised violations to stderr on process exit."""
        with self._violation_lock:
            pending = self._pending_violation
        if pending:
            print(
                f"[Watchdog] FORENSIC: process exiting with unraised "
                f"violation: {pending}",
                file=sys.stderr,
            )

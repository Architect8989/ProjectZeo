import psutil
import time
import os
import threading
from collections import deque


MAX_RUNTIME_SECONDS = 3600          # 1 hour per task
MAX_MEMORY_MB = 4096                # 4GB resident set
MAX_CPU_PERCENT = 90                # sustained CPU

CPU_SAMPLE_INTERVAL = 0.1           # seconds
CPU_WINDOW_SECONDS = 3.0            # sustained window
CPU_MIN_SAMPLES = int(
    CPU_WINDOW_SECONDS / CPU_SAMPLE_INTERVAL
)


class WatchdogViolation(RuntimeError):
    """
    Raised when a runtime safety budget is violated.
    Kernel must catch and transition to ERROR.
    """
    pass


class RuntimeWatchdog:
    """
    Hard runtime guard.

    GUARANTEES:
    - No false positives on short CPU spikes
    - Best-effort only (cannot interrupt blocking ops)
    - Deterministic violation semantics

    FIX RB-4: CPU checking can be paused during LLM inference.

    On CPU-only Ollama inference (40-90s per call), sustained CPU usage above
    90% is expected and legitimate — the model is actively computing. Without
    pausing, the 3-second sustained CPU window triggers WatchdogViolation mid-
    inference and forces _force_safe_shutdown(), aborting every task on CPU-only
    hardware.

    Usage pattern in main.py / run.py:
        watchdog.pause_cpu()
        try:
            result = llm_adapter.get_next_action(...)
        finally:
            watchdog.resume_cpu()

    This preserves CPU monitoring for all non-inference phases (UI operations,
    command execution, restoration) while suppressing false positives during
    legitimate heavy computation.
    """

    def __init__(self):
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())

        # Sliding window of CPU samples
        self._cpu_samples = deque(maxlen=CPU_MIN_SAMPLES)

        # FIX RB-4: CPU pause flag — thread-safe.
        # When True, cpu_percent() is NOT called and samples accumulated during
        # the paused window are discarded on resume.
        self._cpu_paused: bool = False
        self._cpu_pause_lock = threading.Lock()

        # Prime cpu_percent so first real read is meaningful
        try:
            self.process.cpu_percent(interval=None)
        except Exception:
            pass

    # =================================================
    # CPU PAUSE / RESUME  [FIX RB-4]
    # =================================================

    def pause_cpu(self) -> None:
        """
        FIX RB-4: Pause CPU monitoring for the duration of an LLM inference call.

        Call before submitting to the LLM adapter. Must be paired with resume_cpu()
        in a try/finally block. Nested pause_cpu() calls are idempotent.
        """
        with self._cpu_pause_lock:
            self._cpu_paused = True

    def resume_cpu(self) -> None:
        """
        FIX RB-4: Resume CPU monitoring after LLM inference completes.

        Clears accumulated samples from the paused window so a legitimate CPU
        spike during inference does not poison the next monitoring window.
        """
        with self._cpu_pause_lock:
            self._cpu_paused = False
            self._cpu_samples.clear()  # discard samples from paused window

    def is_cpu_paused(self) -> bool:
        """Return True if CPU monitoring is currently paused."""
        with self._cpu_pause_lock:
            return self._cpu_paused

    # =================================================
    # MAIN CHECK
    # =================================================

    def check(self):
        now = time.time()
        elapsed = now - self.start_time

        # --- TIME LIMIT ---
        if elapsed > MAX_RUNTIME_SECONDS:
            self._violate("TIME_LIMIT")

        # --- MEMORY LIMIT ---
        try:
            mem_mb = self.process.memory_info().rss / (1024 * 1024)
            if mem_mb > MAX_MEMORY_MB:
                self._violate("MEMORY_LIMIT")
        except Exception:
            # Fail-open on telemetry failure
            pass

        # --- CPU LIMIT (SUSTAINED) ---
        # FIX RB-4: Skip entirely when paused. cpu_percent(interval=0.1) is a
        # blocking syscall — calling it during LLM inference adds 100ms to every
        # 250ms heartbeat iteration and accumulates samples that would immediately
        # trigger WatchdogViolation on resume (90% CPU from inference).
        with self._cpu_pause_lock:
            _paused = self._cpu_paused

        if not _paused:
            try:
                cpu = self.process.cpu_percent(interval=CPU_SAMPLE_INTERVAL)
                self._cpu_samples.append(cpu)

                # Only evaluate after window is populated
                if len(self._cpu_samples) >= CPU_MIN_SAMPLES:
                    avg_cpu = sum(self._cpu_samples) / len(self._cpu_samples)

                    if avg_cpu > MAX_CPU_PERCENT:
                        self._violate(
                            f"CPU_LIMIT(avg={avg_cpu:.1f}%)"
                        )

            except Exception:
                # Telemetry failure must never abort execution
                pass

    def _violate(self, reason: str):
        msg = f"WATCHDOG_LIMIT_EXCEEDED:{reason}"
        print(f"[WATCHDOG] {msg}")
        raise WatchdogViolation(msg)

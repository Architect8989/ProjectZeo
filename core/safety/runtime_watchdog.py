import psutil
import time
import os
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
    """

    def __init__(self):
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())

        # Sliding window of CPU samples
        self._cpu_samples = deque(maxlen=CPU_MIN_SAMPLES)

        # Prime cpu_percent so first real read is meaningful
        try:
            self.process.cpu_percent(interval=None)
        except Exception:
            pass

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

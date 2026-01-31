# core/safety/runtime_watchdog.py

import psutil
import time
import os


MAX_RUNTIME_SECONDS = 3600          # 1 hour per task
MAX_MEMORY_MB = 4096                # 4GB resident set
MAX_CPU_PERCENT = 90                # sustained CPU


class WatchdogViolation(RuntimeError):
    """
    Raised when a runtime safety budget is violated.
    Kernel must catch and transition to ERROR.
    """
    pass


class RuntimeWatchdog:
    """
    Hard runtime guard.

    - Never kills process
    - Never exits Python
    - Raises deterministic exception
    - Allows restoration path to run
    """

    def __init__(self):
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())

    def check(self):
        elapsed = time.time() - self.start_time

        mem_mb = self.process.memory_info().rss / (1024 * 1024)
        cpu = self.process.cpu_percent(interval=0.1)

        if elapsed > MAX_RUNTIME_SECONDS:
            self._violate("TIME_LIMIT")

        if mem_mb > MAX_MEMORY_MB:
            self._violate("MEMORY_LIMIT")

        if cpu > MAX_CPU_PERCENT:
            self._violate("CPU_LIMIT")

    def _violate(self, reason: str):
        msg = f"WATCHDOG_LIMIT_EXCEEDED:{reason}"
        print(f"[WATCHDOG] {msg}")
        raise WatchdogViolation(msg)

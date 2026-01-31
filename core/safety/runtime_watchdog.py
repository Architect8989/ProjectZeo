# core/safety/runtime_watchdog.py

import psutil
import time
import os
import signal

MAX_RUNTIME_SECONDS = 3600          # 1 hour per task
MAX_MEMORY_MB = 4096                # 4GB
MAX_CPU_PERCENT = 90                # sustained

class RuntimeWatchdog:

    def __init__(self):
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())

    def check(self):
        elapsed = time.time() - self.start_time

        mem_mb = self.process.memory_info().rss / (1024 * 1024)
        cpu = self.process.cpu_percent(interval=0.1)

        if elapsed > MAX_RUNTIME_SECONDS:
            self._kill("TIME_LIMIT")

        if mem_mb > MAX_MEMORY_MB:
            self._kill("MEMORY_LIMIT")

        if cpu > MAX_CPU_PERCENT:
            self._kill("CPU_LIMIT")

    def _kill(self, reason):
        print(f"[WATCHDOG] TERMINATING: {reason}")
        os.kill(os.getpid(), signal.SIGKILL)

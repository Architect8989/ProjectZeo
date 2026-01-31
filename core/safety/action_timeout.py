# core/safety/action_timeout.py

import signal
from contextlib import contextmanager

class ActionTimeout(Exception):
    pass


@contextmanager
def action_timeout(seconds: int):
    def handler(signum, frame):
        raise ActionTimeout("Action timeout exceeded")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)

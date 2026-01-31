import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import threading

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "system.log")

_logger_lock = threading.Lock()

logger = logging.getLogger("SOC")
logger.setLevel(logging.INFO)
logger.propagate = False

# -------------------------------------------------
# Handler Setup (idempotent)
# -------------------------------------------------

if not logger.handlers:

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,   # 10MB
        backupCount=5
    )

    console_handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# -------------------------------------------------
# Safe wrappers
# -------------------------------------------------

def _safe_log(fn, msg: str):
    try:
        with _logger_lock:
            fn(msg)
    except Exception:
        pass


def log_info(msg: str):
    _safe_log(logger.info, msg)


def log_warn(msg: str):
    _safe_log(logger.warning, msg)


def log_error(msg: str):
    _safe_log(logger.error, msg)

import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import threading
import pathlib

# FIX: Absolute path anchored to project root prevents divergence when
# the process is launched from a different working directory.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
LOG_DIR = str(_PROJECT_ROOT / "logs")
try:
    os.makedirs(LOG_DIR, exist_ok=True)
except (PermissionError, OSError):
    # Read-only filesystem: fall back to stderr-only logging.
    LOG_DIR = None

LOG_FILE = os.path.join(LOG_DIR, "system.log") if LOG_DIR else None

_logger_lock = threading.Lock()

logger = logging.getLogger("SOC")
logger.setLevel(logging.INFO)
logger.propagate = False

# -------------------------------------------------
# Handler Setup (idempotent)
# -------------------------------------------------

if not logger.handlers:

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # Only add file handler when LOG_FILE is available (not read-only FS).
    if LOG_FILE is not None:
        try:
            file_handler = RotatingFileHandler(
                LOG_FILE,
                maxBytes=10 * 1024 * 1024,   # 10MB
                backupCount=5
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (PermissionError, OSError):
            pass  # Fall through to console-only logging.

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
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

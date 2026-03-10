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


def log_debug(msg: str):
    _safe_log(logger.debug, msg)


def log_critical(msg: str):
    _safe_log(logger.critical, msg)


def log_event(event: str, data: dict = None, level: str = "info") -> None:
    """
    Structured event logging. Formats event + JSON data into a single line.
    Used by GII components for machine-parseable audit trails.
    
    Example:
        log_event("llamaguard_block", {"categories": ["S1"], "iteration": 42})
        → '2026-03-10 12:00:00 | INFO | EVENT:llamaguard_block | {"categories": ["S1"], "iteration": 42}'
    """
    import json as _json
    try:
        data_str = _json.dumps(data or {}, separators=(",", ":"))
        msg = f"EVENT:{event} | {data_str}"
        fn_map = {
            "debug": logger.debug,
            "info": logger.info,
            "warning": logger.warning,
            "warn": logger.warning,
            "error": logger.error,
            "critical": logger.critical,
        }
        fn = fn_map.get(str(level).lower(), logger.info)
        _safe_log(fn, msg)
    except Exception:
        pass


def get_logger(name: str) -> logging.Logger:
    """Return a child logger namespaced under the SOC root logger."""
    return logging.getLogger(f"SOC.{name}")

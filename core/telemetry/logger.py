# core/telemetry/logger.py

import logging
from logging.handlers import RotatingFileHandler
import os

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "system.log")

logger = logging.getLogger("SOC")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=10 * 1024 * 1024,   # 10MB
    backupCount=5
)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

handler.setFormatter(formatter)
logger.addHandler(handler)


def log_info(msg: str):
    logger.info(msg)


def log_warn(msg: str):
    logger.warning(msg)


def log_error(msg: str):
    logger.error(msg)

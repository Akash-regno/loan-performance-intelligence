"""
src/utils/logger.py
-------------------
Structured, coloured console logger used across all modules.
Provides a single `get_logger(name)` factory so every module gets
a properly named logger without boilerplate.

Usage:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Pipeline started")
    log.warning("Missing values detected: %d rows", n)
    log.error("Schema validation failed")
"""

import logging
import sys


# ANSI color codes for coloured output in terminals that support it
_COLORS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    """Formatter that prepends ANSI color codes to the level name."""

    FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATEFMT = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        color = _COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{_RESET}"
        formatter = logging.Formatter(self.FMT, datefmt=self.DATEFMT)
        return formatter.format(record)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with coloured console output.

    Multiple calls with the same *name* return the same logger instance
    (standard Python logging behaviour).
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured — avoid adding duplicate handlers
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ColorFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger

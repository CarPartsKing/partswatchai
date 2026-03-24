"""
utils/logging_config.py — Shared logging setup for partswatch-ai.

Call `get_logger(__name__)` in any module to get a pre-configured logger
with timestamps and consistent formatting across all Repls.
"""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with timestamp + level formatting.

    Args:
        level: Logging level string — DEBUG, INFO, WARNING, ERROR, CRITICAL.
               Defaults to INFO.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logging is called more than once
    if not root.handlers:
        root.addHandler(handler)
    else:
        root.handlers.clear()
        root.addHandler(handler)

    # Suppress noisy third-party loggers in production
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("hpack").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring the root logger is configured.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A configured logging.Logger instance.
    """
    # Import here to avoid circular imports at module load time
    try:
        from config import LOG_LEVEL
    except EnvironmentError:
        LOG_LEVEL = "INFO"

    setup_logging(LOG_LEVEL)
    return logging.getLogger(name)

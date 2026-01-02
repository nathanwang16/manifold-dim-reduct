"""
Logger Module

Centralized logging with colored console output, file rotation, and metrics support.

Usage:
    from logger import get_logger, LogTimer, log_metrics, configure_logging

    # Configure once at startup (optional, has sensible defaults)
    configure_logging(log_dir="logs", console_level=logging.INFO)

    # Get logger in any module
    logger = get_logger(__name__)

    # Basic logging
    logger.info("Processing started")
    logger.debug("Debug details")
    logger.warning("Something unexpected")
    logger.error("An error occurred")

    # Timed operations
    with LogTimer(logger, "Feature extraction"):
        extract_features()

    # Metrics logging
    log_metrics(logger, {"accuracy": 0.95, "f1": 0.92})

    # Function timing decorator
    @timed(logger)
    def expensive_function():
        pass
"""

from .logger import (
    configure_logging,
    get_logger,
    LogTimer,
    log_metrics,
    log_progress,
    log_context,
    timed,
    ProgressLogger,
    setup_exception_logging,
    Colors,
    LoggerConfig,
)

__all__ = [
    "configure_logging",
    "get_logger",
    "LogTimer",
    "log_metrics",
    "log_progress",
    "log_context",
    "timed",
    "ProgressLogger",
    "setup_exception_logging",
    "Colors",
    "LoggerConfig",
]

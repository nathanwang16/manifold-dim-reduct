"""
Comprehensive Logging Module

Provides centralized logging with:
- Console output with color coding
- Rotating file handlers
- Timing decorators and context managers
- Structured metrics logging
- Easy module integration

Usage:
    from logger import get_logger, LogTimer, log_metrics

    logger = get_logger(__name__)
    logger.info("Processing started")

    with LogTimer(logger, "Feature extraction"):
        # expensive operation
        pass

    log_metrics(logger, {"accuracy": 0.95, "loss": 0.05})
"""

import logging
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from functools import wraps
from typing import Optional, Dict, Any, Callable
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager


# ANSI color codes for console output
class Colors:
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


# Log level to color mapping
LEVEL_COLORS = {
    logging.DEBUG: Colors.DIM + Colors.WHITE,
    logging.INFO: Colors.GREEN,
    logging.WARNING: Colors.YELLOW,
    logging.ERROR: Colors.RED,
    logging.CRITICAL: Colors.BOLD + Colors.RED,
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console."""

    def __init__(self, fmt: str, datefmt: str = None, use_colors: bool = True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors and sys.stdout.isatty():
            color = LEVEL_COLORS.get(record.levelno, Colors.WHITE)
            record.levelname = f"{color}{record.levelname}{Colors.RESET}"
            record.name = f"{Colors.CYAN}{record.name}{Colors.RESET}"
            record.msg = f"{color}{record.msg}{Colors.RESET}"
        return super().format(record)


class MetricsFormatter(logging.Formatter):
    """JSON formatter for structured metrics logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }

        # Include extra fields if present
        if hasattr(record, "metrics"):
            log_data["metrics"] = record.metrics
        if hasattr(record, "duration"):
            log_data["duration_seconds"] = record.duration
        if hasattr(record, "context"):
            log_data["context"] = record.context

        return json.dumps(log_data)


class LoggerConfig:
    """Configuration container for logger setup."""

    def __init__(
        self,
        log_dir: str = "logs",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        use_colors: bool = True,
        log_metrics_file: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.console_level = console_level
        self.file_level = file_level
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.use_colors = use_colors
        self.log_metrics_file = log_metrics_file

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)


# Global configuration instance
_config: Optional[LoggerConfig] = None
_initialized_loggers: Dict[str, logging.Logger] = {}


def configure_logging(
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    **kwargs
) -> None:
    """
    Configure global logging settings. Call once at application startup.

    Args:
        log_dir: Directory for log files
        console_level: Minimum level for console output
        file_level: Minimum level for file output
        **kwargs: Additional config options
    """
    global _config
    _config = LoggerConfig(
        log_dir=log_dir,
        console_level=console_level,
        file_level=file_level,
        **kwargs
    )


def get_logger(
    name: str,
    log_file: Optional[str] = None,
    propagate: bool = False
) -> logging.Logger:
    """
    Get or create a configured logger instance.

    Args:
        name: Logger name (typically __name__)
        log_file: Optional specific log file name
        propagate: Whether to propagate to parent loggers

    Returns:
        Configured logger instance
    """
    global _config, _initialized_loggers

    # Initialize default config if not configured
    if _config is None:
        configure_logging()

    # Return existing logger if already initialized
    if name in _initialized_loggers:
        return _initialized_loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = propagate

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(_config.console_level)
    console_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console_handler.setFormatter(
        ColoredFormatter(console_format, datefmt="%H:%M:%S", use_colors=_config.use_colors)
    )
    logger.addHandler(console_handler)

    # File handler with rotation
    if log_file is None:
        # Use date-based log file
        log_file = f"{datetime.now().strftime('%Y%m%d')}_phase2.log"

    file_path = _config.log_dir / log_file
    file_handler = RotatingFileHandler(
        file_path,
        maxBytes=_config.max_bytes,
        backupCount=_config.backup_count,
    )
    file_handler.setLevel(_config.file_level)
    file_format = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    file_handler.setFormatter(logging.Formatter(file_format))
    logger.addHandler(file_handler)

    # Metrics file handler (JSON format)
    if _config.log_metrics_file:
        metrics_path = _config.log_dir / f"{datetime.now().strftime('%Y%m%d')}_metrics.jsonl"
        metrics_handler = RotatingFileHandler(
            metrics_path,
            maxBytes=_config.max_bytes,
            backupCount=_config.backup_count,
        )
        metrics_handler.setLevel(logging.INFO)
        metrics_handler.setFormatter(MetricsFormatter())
        metrics_handler.addFilter(lambda r: hasattr(r, "metrics"))
        logger.addHandler(metrics_handler)

    _initialized_loggers[name] = logger
    return logger


class LogTimer:
    """
    Context manager for timing code blocks with automatic logging.

    Usage:
        with LogTimer(logger, "Feature extraction"):
            extract_features()
    """

    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        level: int = logging.INFO
    ):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time = None
        self.duration = None

    def __enter__(self) -> "LogTimer":
        self.start_time = time.perf_counter()
        self.logger.log(self.level, f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.duration = time.perf_counter() - self.start_time

        if exc_type is not None:
            self.logger.error(
                f"Failed: {self.operation} after {self.duration:.2f}s - {exc_type.__name__}: {exc_val}"
            )
            return False

        self.logger.log(self.level, f"Completed: {self.operation} in {self.duration:.2f}s")
        return False


@contextmanager
def log_context(logger: logging.Logger, context: Dict[str, Any]):
    """
    Context manager that adds context to all log messages within the block.

    Usage:
        with log_context(logger, {"phase": "feature_extraction", "k": 5}):
            logger.info("Processing...")  # Will include context
    """
    # Store original factory
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.context = context
        return record

    logging.setLogRecordFactory(record_factory)
    try:
        yield
    finally:
        logging.setLogRecordFactory(old_factory)


def log_metrics(
    logger: logging.Logger,
    metrics: Dict[str, Any],
    message: str = "Metrics",
    level: int = logging.INFO
) -> None:
    """
    Log structured metrics data.

    Args:
        logger: Logger instance
        metrics: Dictionary of metric values
        message: Log message
        level: Log level
    """
    record = logger.makeRecord(
        logger.name, level, "", 0, message, (), None
    )
    record.metrics = metrics
    logger.handle(record)

    # Also log human-readable version
    metrics_str = " | ".join(f"{k}={v}" for k, v in metrics.items())
    logger.log(level, f"{message}: {metrics_str}")


def timed(logger: logging.Logger = None, level: int = logging.INFO):
    """
    Decorator for timing function execution.

    Usage:
        @timed(logger)
        def expensive_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            start = time.perf_counter()
            logger.log(level, f"Calling {func.__name__}")

            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start
                logger.log(level, f"{func.__name__} completed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.perf_counter() - start
                logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
                raise

        return wrapper
    return decorator


def log_progress(
    logger: logging.Logger,
    iterable,
    desc: str = "Processing",
    log_every: int = 1000
):
    """
    Generator that logs progress for iterables.

    Usage:
        for item in log_progress(logger, items, "Processing items"):
            process(item)
    """
    total = len(iterable) if hasattr(iterable, "__len__") else None
    start_time = time.perf_counter()

    for i, item in enumerate(iterable):
        if i > 0 and i % log_every == 0:
            elapsed = time.perf_counter() - start_time
            rate = i / elapsed if elapsed > 0 else 0

            if total:
                pct = 100 * i / total
                eta = (total - i) / rate if rate > 0 else 0
                logger.info(f"{desc}: {i}/{total} ({pct:.1f}%) - {rate:.1f}/s - ETA: {eta:.1f}s")
            else:
                logger.info(f"{desc}: {i} items - {rate:.1f}/s")

        yield item

    elapsed = time.perf_counter() - start_time
    final_count = i + 1 if "i" in dir() else 0
    logger.info(f"{desc}: Completed {final_count} items in {elapsed:.2f}s")


class ProgressLogger:
    """
    Class-based progress logger for more control.

    Usage:
        progress = ProgressLogger(logger, total=1000, desc="Processing")
        for item in items:
            process(item)
            progress.update()
        progress.finish()
    """

    def __init__(
        self,
        logger: logging.Logger,
        total: Optional[int] = None,
        desc: str = "Processing",
        log_every: int = 1000
    ):
        self.logger = logger
        self.total = total
        self.desc = desc
        self.log_every = log_every
        self.count = 0
        self.start_time = time.perf_counter()

    def update(self, n: int = 1) -> None:
        """Update progress counter."""
        self.count += n
        if self.count % self.log_every == 0:
            self._log_progress()

    def _log_progress(self) -> None:
        elapsed = time.perf_counter() - self.start_time
        rate = self.count / elapsed if elapsed > 0 else 0

        if self.total:
            pct = 100 * self.count / self.total
            eta = (self.total - self.count) / rate if rate > 0 else 0
            self.logger.info(
                f"{self.desc}: {self.count}/{self.total} ({pct:.1f}%) - {rate:.1f}/s - ETA: {eta:.1f}s"
            )
        else:
            self.logger.info(f"{self.desc}: {self.count} items - {rate:.1f}/s")

    def finish(self) -> None:
        """Log final progress summary."""
        elapsed = time.perf_counter() - self.start_time
        rate = self.count / elapsed if elapsed > 0 else 0
        self.logger.info(f"{self.desc}: Completed {self.count} items in {elapsed:.2f}s ({rate:.1f}/s)")


def setup_exception_logging(logger: logging.Logger) -> None:
    """
    Configure global exception handler to log uncaught exceptions.

    Args:
        logger: Logger to use for exception logging
    """
    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = exception_handler

"""Logging helpers for Kubera runs."""

from __future__ import annotations

import logging
import sys

from kubera.utils.run_context import RunContext


FILE_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"


def configure_logging(
    run_context: RunContext,
    log_level: str,
    *,
    logger_name: str = "kubera",
) -> logging.Logger:
    """Configure the shared Kubera logger for console and file output."""

    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level))
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    run_context.log_file_path.parent.mkdir(parents=True, exist_ok=True)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    file_handler = logging.FileHandler(run_context.log_file_path, encoding="utf-8")
    file_handler.setLevel(getattr(logging, log_level))
    file_handler.setFormatter(logging.Formatter(FILE_LOG_FORMAT))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

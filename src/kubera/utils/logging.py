"""Logging helpers for Kubera runs."""

from __future__ import annotations

import logging
import re
import sys

from kubera.utils.run_context import RunContext


FILE_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
REDACTED_LOG_VALUE = "[redacted]"
SENSITIVE_QUERY_PARAM_PATTERN = re.compile(
    r"((?:api[_-]?key|api[_-]?token|token|access[_-]?token|auth)=)([^&\s]+)",
    re.IGNORECASE,
)
SENSITIVE_HEADER_PATTERN = re.compile(
    r"((?:x-goog-api-key|authorization|api[_-]?key|api[_-]?token|access[_-]?token|token)\s*[:=]\s*)([^\s,;]+)",
    re.IGNORECASE,
)
BEARER_TOKEN_PATTERN = re.compile(r"(Bearer\s+)([A-Za-z0-9._\-+/=]+)")


class RedactingFormatter(logging.Formatter):
    """Sanitize log text before it is written to any sink."""

    def format(self, record: logging.LogRecord) -> str:
        return sanitize_log_text(super().format(record))


def sanitize_log_text(message: str) -> str:
    """Redact common secret-bearing fragments from log output."""

    sanitized = BEARER_TOKEN_PATTERN.sub(
        rf"\1{REDACTED_LOG_VALUE}",
        message,
    )
    sanitized = SENSITIVE_QUERY_PARAM_PATTERN.sub(
        rf"\1{REDACTED_LOG_VALUE}",
        sanitized,
    )
    sanitized = SENSITIVE_HEADER_PATTERN.sub(
        rf"\1{REDACTED_LOG_VALUE}",
        sanitized,
    )
    return sanitized


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
    console_handler.setFormatter(RedactingFormatter("%(message)s"))

    file_handler = logging.FileHandler(run_context.log_file_path, encoding="utf-8")
    file_handler.setLevel(getattr(logging, log_level))
    file_handler.setFormatter(RedactingFormatter(FILE_LOG_FORMAT))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

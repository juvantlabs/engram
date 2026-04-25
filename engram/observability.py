"""Structured logging setup for Engram.

All Engram modules obtain loggers via :func:`get_logger`. Applications can
call :func:`configure_logging` once on startup to install a JSON renderer
and set the log level; if they don't, ``structlog``'s defaults apply and
events are rendered as human-readable key-value pairs to stderr.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.stdlib import BoundLogger

_configured: bool = False


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Install the Engram log configuration.

    Idempotent: repeated calls replace the previous configuration.

    :param level: Standard Python log level (``DEBUG``/``INFO``/...).
    :param json_output: If ``True`` (default), emit one-line JSON events; if
        ``False``, use structlog's ``ConsoleRenderer`` for humans.
    """
    global _configured

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=level.upper(),
        force=True,
    )

    renderer: Any
    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level.upper())
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> BoundLogger:
    """Return a structlog logger bound to ``name``."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]

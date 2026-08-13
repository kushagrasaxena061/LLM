# 3. utils/logger.py

# Where it goes: Inside the utils/ folder.
# Why it exists: Standard print() statements are terrible for deep learning. 
# You can't search them easily, they don't have timestamps, 
# and they don't format dictionaries well. We use structlog to create a beautiful, 
# professional terminal output.

# utils/logger.py
"""Structured logging framework using structlog and rich visual rendering."""

import logging
import sys

import structlog


def get_logger(name: str = "llm_platform"):
    """
    Returns a pre-configured logger that prints beautiful, color-coded terminal messages.
    """
    # Configure the standard Python logging library to output to the terminal (sys.stdout)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

    # Configure 'structlog', which acts as a wrapper to make logs look highly professional
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,      # Allows passing variables to logs
            structlog.processors.add_log_level,           # Adds [INFO], [ERROR], etc.
            structlog.processors.StackInfoRenderer(),     # Formats error stack traces
            structlog.dev.set_exc_info,                   # Formats exceptions
            structlog.processors.TimeStamper(fmt="iso"),  # Adds an ISO-8601 timestamp to every log
            structlog.dev.ConsoleRenderer(colors=True),   # Adds beautiful terminal colors
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Return the customized logger object
    return structlog.get_logger(name)

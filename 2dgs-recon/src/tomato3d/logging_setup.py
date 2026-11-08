"""The single logging configuration used by every entry point."""

from __future__ import annotations

import logging

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once, idempotently.

    Args:
        level: Logging level name, e.g. ``"INFO"``. Unknown names fall back to ``INFO``
            with a warning rather than raising, so that a typo in ``.env`` never stops a
            capture session from being processed.
    """
    resolved = logging.getLevelNamesMapping().get(level.upper())
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(resolved if resolved is not None else logging.INFO)
    if resolved is None:
        root.warning("unknown log level %r, falling back to INFO", level)

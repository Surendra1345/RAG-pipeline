"""Logging setup. Logs go to stdout, which is where Docker and hosting platforms collect them.

Call setup_logging() once at start-up (the command-line tools and, later, the API do this).
The level comes from LOG_LEVEL in backend/.env (default INFO; use DEBUG to see more).
"""
import logging
import sys

from app.core.config import settings

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_NOISY = ("httpx", "httpx2", "httpcore", "urllib3", "sentence_transformers", "groq", "google", "filelock", "huggingface_hub")


def setup_logging(level: str | None = None) -> None:
    root = logging.getLogger()
    root.setLevel((level or settings.log_level).upper())
    if not any(getattr(h, "_rag_handler", False) for h in root.handlers):  # safe to call twice
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
        handler._rag_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)

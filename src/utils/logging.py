"""Console logging shared by scripts, training code and the API."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Install a single stdout handler on the root logger.

    Idempotent: repeated calls (for example from a script that also imports the
    API) do not duplicate handlers or produce doubled log lines.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    # These libraries are chatty at INFO and drown out the project's own output.
    for noisy in ("urllib3", "filelock", "matplotlib", "PIL", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)

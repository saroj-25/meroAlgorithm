"""Consistent logging for scripts and the Streamlit app."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_configured = False


def get_logger(name: str = "algosathi", log_file: str | Path | None = None,
               level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger (idempotent: safe to call many times)."""
    global _configured
    root = logging.getLogger("algosathi")
    if not _configured:
        root.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
        root.addHandler(handler)
        _configured = True
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FMT))
        root.addHandler(fh)
    return root if name in ("algosathi", None) else root.getChild(name)

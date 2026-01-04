#!/usr/bin/env python3
"""Compatibility shim for model config lookup."""

from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fv.model_config import get_model_config  # noqa: E402

__all__ = ["get_model_config"]

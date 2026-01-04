"""Local import shim to expose src/fv as a top-level package."""

from pathlib import Path

_src_pkg = Path(__file__).resolve().parent.parent / "src" / "fv"
if _src_pkg.is_dir():
    __path__.append(str(_src_pkg))

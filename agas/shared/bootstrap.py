"""Import `shared.bootstrap` by file path instead of by package name.

`agas/` is not a Python package and is not on `sys.path` as a package root, so
`import shared.bootstrap` cannot resolve. The previous form worked only for the
handful of scripts that happened to have `agas/` itself on `sys.path`, which is
why 22 of 31 scripts broke.

The fix is to load the helper by absolute file location with `importlib`, which
needs nothing on `sys.path`. One line, and it works from any directory depth.

Call it as `_add_shared()` in the script header; it returns the directory it
added.
"""

import importlib.util
import sys
from pathlib import Path

_MARKER = "_agas_shared_bootstrap"


def _bootstrap_path():
    here = Path(__file__).resolve().parent
    for parent in here.parents:
        candidate = parent / "shared" / "bootstrap.py"
        if candidate.exists():
            return candidate, parent
    raise RuntimeError(f"shared/bootstrap.py not found above {here}")


def add_shared(start=None):
    """Put `agas/shared` on sys.path and return it. Safe to call repeatedly."""
    if _MARKER in sys.modules:
        return sys.modules[_MARKER]._dir
    src, shared = _bootstrap_path()
    spec = importlib.util.spec_from_file_location(_MARKER, src)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MARKER] = module
    spec.loader.exec_module(module)
    module._dir = shared
    if str(shared) not in sys.path:
        sys.path.insert(0, str(shared))
    return shared

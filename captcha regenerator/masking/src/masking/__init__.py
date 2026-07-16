"""Pin HF/torch caches before torch is imported, then expose top-level API."""

import os as _os
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parent.parent.parent
_LOCAL_CACHE = _REPO_ROOT / ".cache"
_os.environ.setdefault("TORCH_HOME", str(_LOCAL_CACHE / "torch"))
_os.environ.setdefault("HF_HOME", str(_LOCAL_CACHE / "hf"))
_os.environ.setdefault("XDG_CACHE_HOME", str(_LOCAL_CACHE))

__version__ = "0.1.0"

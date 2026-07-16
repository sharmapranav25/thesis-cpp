"""Phase 1: CAPTCHA character extraction. See README.md for usage."""

import os as _os
from pathlib import Path as _Path

# Pin model/torch-hub caches to a local .cache/ on the same drive as the venv,
# so installs never spill to C: even when third-party libs fall back to
# torch.hub or hf_hub. Set BEFORE any torch import.
_REPO_ROOT = _Path(__file__).resolve().parent.parent.parent
_LOCAL_CACHE = _REPO_ROOT / ".cache"
_os.environ.setdefault("TORCH_HOME", str(_LOCAL_CACHE / "torch"))
_os.environ.setdefault("HF_HOME", str(_LOCAL_CACHE / "hf"))
_os.environ.setdefault("XDG_CACHE_HOME", str(_LOCAL_CACHE))

from char_extraction.api import extract  # noqa: E402
from char_extraction.types import CharBox, ExtractionResult  # noqa: E402

__version__ = "0.1.0"
__all__ = ["extract", "ExtractionResult", "CharBox", "__version__"]

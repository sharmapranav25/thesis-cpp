"""Config-driven dispatch.

Resolves a `captcha_type` string (or None) to a configured `BaseMethod`
instance by:
1. Loading `configs/_default.yaml` (universal-baseline fallback).
2. Loading `configs/<captcha_type>.yaml` if present and shallow-merging on top.
3. Reading `method:` from the merged config.
4. Looking up that method class in the registry.
5. Instantiating with the merged config.

If no per-type config exists OR the type config has no `method:`, the
universal baseline from `_default.yaml` is used. This is the graceful-
degradation path for unseen types.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

# Side-effect imports register methods.
from char_extraction import methods as _methods_pkg  # noqa: F401
from char_extraction.methods import get_method
from char_extraction.methods.base import BaseMethod


def _safe_load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config {path} must be a mapping at the top level, got {type(loaded).__name__}")
    return loaded


def load_config(captcha_type: Optional[str], config_dir: Union[str, Path]) -> Dict[str, Any]:
    cfg_dir = Path(config_dir)
    default = _safe_load(cfg_dir / "_default.yaml")
    if captcha_type is None:
        return default
    type_path = cfg_dir / f"{captcha_type}.yaml"
    overlay = _safe_load(type_path)
    merged = dict(default)
    merged.update(overlay)
    return merged


def route(
    captcha_type: Optional[str],
    config_dir: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
) -> BaseMethod:
    cfg = load_config(captcha_type, config_dir)
    if overrides:
        cfg.update(overrides)
    method_name = cfg.get("method")
    if not method_name:
        raise ValueError(
            f"No method resolvable for captcha_type={captcha_type!r}. "
            f"_default.yaml must define a 'method' key."
        )
    method_cls = get_method(method_name)
    return method_cls(cfg)

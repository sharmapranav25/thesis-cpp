"""Method registry. Methods auto-register via @register_method.

Side-effect imports below trigger the decorators. To add a new method,
import it here so its `@register_method` runs at package-load time.

Heavy methods (universal_sam, universal_craft) are imported lazily to avoid
loading torch on every CLI invocation. They register only when first
referenced via _ensure_loaded().
"""

from typing import Dict

from char_extraction.methods.base import BaseMethod, get_method, list_methods, register_method

# Eager: zero-dep classical methods.
from char_extraction.methods import otsu_cc as _otsu_cc  # noqa: F401

# Lazy: each entry is the dotted module path that, when imported, triggers
# the @register_method decorator. We import on demand from get_method().
_LAZY_METHODS: Dict[str, str] = {
    "universal_sam": "char_extraction.methods.universal_sam",
    "universal_craft": "char_extraction.methods.universal_craft",
    "universal_psstrnet": "char_extraction.methods.psstrnet",
    "universal_swt": "char_extraction.methods.swt",
}


def _ensure_loaded(method_name: str) -> None:
    if method_name in _LAZY_METHODS:
        import importlib
        importlib.import_module(_LAZY_METHODS[method_name])


# Wrap base get_method to trigger lazy import first.
_base_get_method = get_method


def get_method(name: str):  # type: ignore[no-redef]
    _ensure_loaded(name)
    return _base_get_method(name)


__all__ = ["BaseMethod", "get_method", "list_methods", "register_method"]

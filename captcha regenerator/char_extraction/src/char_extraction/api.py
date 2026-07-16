"""Public API: `extract(image, captcha_type, ...) -> ExtractionResult`.

Thin facade; orchestrates load -> route -> run -> (optionally) write.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from char_extraction.io_utils import derive_captcha_id, load_image, write_extraction
from char_extraction.router import route
from char_extraction.types import ExtractionResult

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def extract(
    image: Union[str, Path, np.ndarray],
    captcha_type: Optional[str] = None,
    config_dir: Optional[Union[str, Path]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    captcha_id: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> ExtractionResult:
    """Run the extraction pipeline on one image.

    Args:
        image: path to an image file, or an RGB uint8 ndarray.
        captcha_type: type name. None -> use _default.yaml universal baseline.
        config_dir: where configs live. Defaults to <repo>/configs.
        out_dir: if given, write the spec's per-image output layout there.
        captcha_id: explicit ID; required if `image` is an ndarray.
        overrides: optional dict shallow-merged onto the resolved config.
    """
    cfg_dir = Path(config_dir) if config_dir else _DEFAULT_CONFIG_DIR
    if isinstance(image, (str, Path)):
        img = load_image(image)
    else:
        img = image
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
    cid = derive_captcha_id(image, captcha_id)
    method = route(captcha_type, cfg_dir, overrides=overrides)
    result = method.run(img, captcha_id=cid, captcha_type=captcha_type)
    if out_dir is not None:
        write_extraction(result, img, out_dir)
    return result

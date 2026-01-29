from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from PIL import Image

try:
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
except Exception:
    pillow_heif = None

try:
    import piexif  # type: ignore
except Exception:
    piexif = None


def _extract_exif(path: str) -> Dict:
    if piexif is None:
        return {}
    try:
        exif_dict = piexif.load(path)
    except Exception:
        return {}
    clean = {}
    for ifd_name, ifd in exif_dict.items():
        if isinstance(ifd, dict):
            for tag, value in ifd.items():
                try:
                    clean[str(tag)] = value
                except Exception:
                    continue
    return clean


def load_heic_with_exif(path: str) -> Tuple[np.ndarray, Dict]:
    img = Image.open(path)
    img = img.convert("RGB")
    img_np = np.asarray(img).astype(np.float32) / 255.0
    exif = _extract_exif(path)
    return img_np, exif


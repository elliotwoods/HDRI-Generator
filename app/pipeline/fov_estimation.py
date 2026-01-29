from __future__ import annotations

from math import atan, degrees
from typing import Dict, Tuple

from config import DEFAULT_FOV_H_DEG, DEFAULT_FOV_V_DEG

FOCAL_LENGTH_35MM_TAG = "41989"


def _fov_from_focal_length_35mm(focal_length_mm: float) -> Tuple[float, float]:
    if focal_length_mm <= 0.0:
        return DEFAULT_FOV_H_DEG, DEFAULT_FOV_V_DEG
    fov_h = 2.0 * degrees(atan(36.0 / (2.0 * focal_length_mm)))
    fov_v = 2.0 * degrees(atan(24.0 / (2.0 * focal_length_mm)))
    return fov_h, fov_v


def estimate_fov(exif: Dict) -> Tuple[float, float]:
    focal = None
    try:
        raw = exif.get(FOCAL_LENGTH_35MM_TAG)
        if isinstance(raw, (int, float)):
            focal = float(raw)
        elif isinstance(raw, (bytes, bytearray)):
            focal = float(raw.decode(errors="ignore"))
    except Exception:
        focal = None
    if focal is None:
        return DEFAULT_FOV_H_DEG, DEFAULT_FOV_V_DEG
    return _fov_from_focal_length_35mm(focal)


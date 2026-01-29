from __future__ import annotations

from typing import Dict

import numpy as np


def infer_scene_type(exif: Dict, img_srgb: np.ndarray) -> str:
    avg_luma = float(np.mean(0.2126 * img_srgb[..., 0] + 0.7152 * img_srgb[..., 1] + 0.0722 * img_srgb[..., 2]))
    if avg_luma < 0.12:
        return "night"
    if avg_luma < 0.35:
        return "indoor"
    return "outdoor"


def build_internal_prompt(scene_type: str) -> str:
    if scene_type == "night":
        return "a realistic night environment with natural lighting"
    if scene_type == "indoor":
        return "a realistic indoor environment with walls and ceiling"
    return "a realistic outdoor environment with natural sky and ground"


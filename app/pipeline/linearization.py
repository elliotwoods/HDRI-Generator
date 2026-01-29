from __future__ import annotations

import numpy as np


def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    cutoff = 0.04045
    low = img / 12.92
    high = ((img + 0.055) / 1.055) ** 2.4
    return np.where(img <= cutoff, low, high).astype(np.float32)


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    cutoff = 0.0031308
    low = img * 12.92
    high = 1.055 * (img ** (1.0 / 2.4)) - 0.055
    return np.where(img <= cutoff, low, high).astype(np.float32)


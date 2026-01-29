from __future__ import annotations

import numpy as np

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def compute_highlight_mask(img_lin: np.ndarray, threshold: float = 0.98) -> np.ndarray:
    luma = 0.2126 * img_lin[..., 0] + 0.7152 * img_lin[..., 1] + 0.0722 * img_lin[..., 2]
    mask = luma >= threshold
    if cv2 is None:
        return mask
    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((7, 7), dtype=np.uint8)
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    return dilated > 0


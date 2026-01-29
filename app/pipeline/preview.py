from __future__ import annotations

import numpy as np

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

from .linearization import linear_to_srgb


def tone_map_for_preview(hdr: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        tonemap = cv2.createTonemapReinhard(gamma=1.0)
        ldr = tonemap.process(hdr.astype(np.float32))
        return np.clip(ldr, 0.0, 1.0)
    hdr = np.clip(hdr, 0.0, None)
    ldr = hdr / (1.0 + hdr)
    return np.clip(linear_to_srgb(ldr), 0.0, 1.0)


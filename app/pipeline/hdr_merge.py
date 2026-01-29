from __future__ import annotations

from typing import List

import numpy as np

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def _resize_to(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    if img.shape[0] == h and img.shape[1] == w:
        return img
    if cv2 is not None:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    from PIL import Image

    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img_u8)
    pil = pil.resize((w, h), Image.BILINEAR)
    return np.asarray(pil).astype(np.float32) / 255.0


def merge_hdr_from_panos(panos_lin: List[np.ndarray], ev_offsets: List[float]) -> np.ndarray:
    if cv2 is None:
        base_h, base_w = panos_lin[0].shape[:2]
        resized = [_resize_to(p, (base_h, base_w)) for p in panos_lin]
        return np.mean(np.stack(resized, axis=0), axis=0).astype(np.float32)
    times = np.array([2.0 ** ev for ev in ev_offsets], dtype=np.float32)
    base_h, base_w = panos_lin[0].shape[:2]
    resized = [_resize_to(p, (base_h, base_w)) for p in panos_lin]
    ldrs = [np.clip(pano, 0.0, 1.0) for pano in resized]
    ldrs_u8 = [np.clip(p * 255.0, 0, 255).astype(np.uint8) for p in ldrs]
    merge = cv2.createMergeDebevec()
    try:
        hdr = merge.process(ldrs_u8, times=times)
    except Exception as exc:
        shapes = [p.shape for p in panos_lin]
        resized_shapes = [p.shape for p in resized]
        raise RuntimeError(
            f"MergeDebevec failed. pano_shapes={shapes} resized_shapes={resized_shapes}"
        ) from exc
    return hdr.astype(np.float32)


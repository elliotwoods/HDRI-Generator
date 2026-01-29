from __future__ import annotations

import numpy as np

from config import EV_OFFSETS


def apply_ev(img_lin: np.ndarray, ev: float) -> np.ndarray:
    return img_lin * (2.0 ** ev)


def get_default_ev_offsets() -> list[float]:
    evs = list(EV_OFFSETS)
    if 0.0 in evs:
        evs.remove(0.0)
        evs.insert(0, 0.0)
    return evs


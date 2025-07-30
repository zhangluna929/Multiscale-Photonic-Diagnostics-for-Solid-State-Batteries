"""metrics"""
from __future__ import annotations

from typing import Sequence, Tuple
import numpy as np

__all__ = ["peak_slope", "fwhm"]

def peak_slope(theta: Sequence[float], R: Sequence[float]) -> float:
    """Return max |dR/dθ| (slope) of SPR curve"""
    theta_arr = np.asarray(theta, dtype=float)
    R_arr = np.asarray(R, dtype=float)
    if theta_arr.size < 3:
        raise ValueError("Need >=3 points for slope")
    dR = np.gradient(R_arr, theta_arr)
    return float(np.max(np.abs(dR)))


def fwhm(x_hits: Sequence[float], bins: int = 100) -> float:
    """Compute FWHM of 1D spot distribution"""
    x = np.asarray(x_hits, dtype=float)
    if x.size == 0:
        return 0.0
    hist, edges = np.histogram(x, bins=bins)
    if hist.max() == 0:
        return 0.0
    half_max = hist.max() / 2.0
    # indices where histogram >= half max
    idx = np.where(hist >= half_max)[0]
    if idx.size == 0:
        return 0.0
    left_edge = edges[idx[0]]
    right_edge = edges[idx[-1] + 1]
    return float(right_edge - left_edge) 
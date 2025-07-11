"""metrics.py
Core evaluation metrics for SPR and beam profile.
"""
from __future__ import annotations

from typing import Sequence, Tuple
import numpy as np

__all__ = ["peak_slope", "fwhm"]

def peak_slope(theta: Sequence[float], R: Sequence[float]) -> float:
    """Return max |dR/dθ| (slope) of SPR curve.

    theta : angles (rad or deg, same unit in-out)
    R     : reflectance array (same length as theta)
    """
    theta_arr = np.asarray(theta, dtype=float)
    R_arr = np.asarray(R, dtype=float)
    if theta_arr.size < 3:
        raise ValueError("Need >=3 points for slope")
    dR = np.gradient(R_arr, theta_arr)
    return float(np.max(np.abs(dR)))


def fwhm(x_hits: Sequence[float], bins: int = 100) -> float:
    """Compute FWHM of 1D spot distribution.

    x_hits : hit positions (list/array)
    Returns width at half max (same units as x).
    """
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

__all__.append("phase_shift_sensitivity")

def phase_shift_sensitivity(film, wavelength_nm: float, theta_rad: float, nm_change: float) -> float:
    """Return |Δφ| (rad) when film thickness increases by nm_change.

    film : MetalFilm instance
    nm_change : thickness increment (nm)
    """
    base_phase = film.get_phase_shift(wavelength_nm, theta_rad)
    # clone film with increased thickness
    from copy import copy
    film2 = copy(film)
    film2.d = film.d + nm_change
    new_phase = film2.get_phase_shift(wavelength_nm, theta_rad)
    return abs(new_phase - base_phase) 
"""materials"""
from __future__ import annotations

from typing import Sequence
import numpy as np
from scipy.interpolate import interp1d

__all__ = ["gold_n_complex", "gold_n_complex_scalar", "sellmeier_coeff_to_n"]

# --- Johnson & Christy (1972) Au n & k, trimmed ---
_GOLD_NK_DATA = np.array([
    [500,  1.54, 1.85],
    [520,  1.40, 1.88],
    [540,  1.29, 1.97],
    [560,  1.20, 2.14],
    [580,  1.12, 2.33],
    [600,  1.06, 2.54],
    [620,  1.02, 2.73],
    [640,  1.00, 2.89],
    [660,  0.98, 3.04],
    [680,  0.96, 3.18],
    [700,  0.94, 3.30],
])

_wl = _GOLD_NK_DATA[:, 0]
_n = _GOLD_NK_DATA[:, 1]
_k = _GOLD_NK_DATA[:, 2]

_n_interp = interp1d(_wl, _n, kind="cubic", fill_value="extrapolate")  # type: ignore[arg-type]
_k_interp = interp1d(_wl, _k, kind="cubic", fill_value="extrapolate")  # type: ignore[arg-type]


def gold_n_complex(wavelength_nm: float | np.ndarray) -> complex | np.ndarray:
    """Return gold's complex index n - i k for given wavelength (nm)"""
    wl = np.asarray(wavelength_nm)
    n = _n_interp(wl)
    k = _k_interp(wl)
    return n - 1j * k 


def gold_n_complex_scalar(wavelength_nm: float) -> complex:
    """Scalar version – always returns complex (typing helper)"""
    return complex(gold_n_complex(wavelength_nm))


def sellmeier_coeff_to_n(coeffs: tuple[float, float, float, float, float, float], wavelength_nm: float) -> float:
    """Return n from Sellmeier coeffs (λ in µm)"""
    B1, B2, B3, C1, C2, C3 = coeffs
    lam_um = wavelength_nm / 1000.0
    lam2 = lam_um ** 2
    n2 = 1 + B1 * lam2 / (lam2 - C1) + B2 * lam2 / (lam2 - C2) + B3 * lam2 / (lam2 - C3)
    return float(np.sqrt(n2)) 
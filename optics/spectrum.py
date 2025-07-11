"""spectrum.py
Spectrum utilities and thickness optimizer.
"""
from __future__ import annotations

from typing import Sequence, Callable, Optional
import numpy as np
import optuna  # type: ignore[import-not-found]

from .elements import MetalFilm


def reflectance_spectrum(film: MetalFilm, wavelengths: Sequence[float], theta_rad: float) -> np.ndarray:
    """Calculate reflectance at fixed angle for multiple wavelengths."""
    return np.array([film._tm_reflectance(wl, theta_rad) for wl in wavelengths])


def optimize_thickness(
    target_wavelength_nm: float = 632.8,
    theta_deg: float = 60.0,
    n_trials: int = 50,
    n_metal_func: Optional[Callable[[float], complex] | complex] = None,
) -> float:
    """Optimize gold film thickness using Optuna to minimize reflectance."""
    theta_rad = np.deg2rad(theta_deg)

    def objective(trial: optuna.Trial):
        d = trial.suggest_uniform("thickness", 30.0, 70.0)
        film = MetalFilm(thickness_nm=d, n_metal=n_metal_func) if n_metal_func else MetalFilm(thickness_nm=d)
        R = film._tm_reflectance(target_wavelength_nm, theta_rad)
        return R

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_value  # best thickness reflectance 
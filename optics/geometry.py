"""geometry.py
Helper geometry generators.
"""
from __future__ import annotations

import numpy as np
from typing import List

from .elements import Mirror
from .ray import Ray


SOURCE_POS = np.array([0.0, 0.0])
SCREEN_Y = 5.0

__all__ = ["default_mirrors", "generate_rays", "gaussian_beam"]


def default_mirrors(num_mirrors: int, radius: float = 3.0) -> List[Mirror]:
    """Return list of evenly spaced mirrors, normals toward source."""
    mirrors: List[Mirror] = []
    for i in range(num_mirrors):
        angle = np.pi / 6 + (i / max(1, num_mirrors - 1)) * np.pi / 2  # start at 30°
        pos = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        normal_angle = angle - np.pi / 2  # faces source
        normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
        mirrors.append(Mirror(position=pos, normal=normal))
    return mirrors


def generate_rays(n_rays: int, theta_spread_deg: float, rng: np.random.Generator | None = None) -> List[Ray]:
    """Generate ray bundle with given angular spread (deg)."""
    rng = rng or np.random.default_rng()
    theta_0 = np.pi / 2  # vertical (0 deg along +y)
    theta_spread_rad = np.deg2rad(theta_spread_deg)
    rays: List[Ray] = []
    for _ in range(n_rays):
        theta = theta_0 + rng.uniform(-theta_spread_rad / 2, theta_spread_rad / 2)
        direction = np.array([np.cos(theta), np.sin(theta)])
        rays.append(Ray(position=SOURCE_POS.copy(), direction=direction))
    return rays


def gaussian_beam(N: int, waist: float, wavelength_nm: float, rng: np.random.Generator | None = None):
    """Generate N ray origins sampled from a Gaussian spot (2D).

    waist : beam 1/e^2 radius (mm or same unit as downstream optics)
    Returns array of shape (N,2).
    """
    rng = rng or np.random.default_rng()
    sigma = waist / 2  # quick relation between waist and sigma
    xy = rng.normal(scale=sigma, size=(N, 2))
    return xy 
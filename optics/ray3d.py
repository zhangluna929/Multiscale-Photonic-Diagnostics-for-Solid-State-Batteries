"""ray3d.py
Simple 3-D ray object for curved-shell tracing.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Union

ArrayLike = Union[np.ndarray, list, tuple]


def _norm(v: ArrayLike) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero vector")
    return v / n


@dataclass
class Ray3D:
    pos: np.ndarray  # (x,y,z)
    dir: np.ndarray  # unit vector

    wavelength: float = 632.8
    intensity: float = 1.0

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=float)
        self.dir = _norm(self.dir)

    def propagate(self, t: float) -> "Ray3D":
        return Ray3D(self.pos + self.dir * t, self.dir.copy(), self.wavelength, self.intensity) 
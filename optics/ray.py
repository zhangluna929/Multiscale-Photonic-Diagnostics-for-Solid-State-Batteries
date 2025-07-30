from __future__ import annotations

"""ray"""

from dataclasses import dataclass
from typing import Union
import numpy as np

ArrayLike = Union[np.ndarray, list, tuple]


@dataclass
class Ray:
    """Geometrical ray"""

    position: np.ndarray
    direction: np.ndarray
    wavelength: float = 632.8  # He-Ne 632.8 nm
    polarization: str = "TM"
    intensity: float = 1.0

    def __post_init__(self):
        self.direction = self._normalize(self.direction)
        self.position = np.asarray(self.position, dtype=float)

    @staticmethod
    def _normalize(vec: ArrayLike) -> np.ndarray:
        vec = np.asarray(vec, dtype=float)
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("Direction vector cannot be zero length.")
        return vec / norm

    def propagate(self, distance: float) -> "Ray":
        """Propagate along direction by given distance, return new Ray"""
        new_pos = self.position + self.direction * distance
        return Ray(new_pos, self.direction.copy(), self.wavelength, self.polarization, self.intensity) 
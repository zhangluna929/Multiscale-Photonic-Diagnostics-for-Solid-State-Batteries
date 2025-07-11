from __future__ import annotations

"""elements.py
Optical elements – abstract base & concrete bits.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Callable, Union
import numpy as np

from .ray import Ray
from .materials import gold_n_complex_scalar


class OpticalElement(ABC):
    """Abstract base for any optical element."""

    @abstractmethod
    def interact(self, ray: Ray) -> List[Ray]:
        """Interact with incoming ray, return list (split allowed)."""
        pass

    def intersect_distance(self, ray: Ray) -> float | None:
        """Distance t (>0) from ray.pos to element plane; None if miss."""
        return None

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Mirror(OpticalElement):
    """Ideal flat mirror."""

    def __init__(self, position: np.ndarray, normal: np.ndarray):
        self.position = np.asarray(position, dtype=float)
        self.normal = self._normalize(normal)

    @staticmethod
    def _normalize(vec):
        vec = np.asarray(vec, dtype=float)
        return vec / np.linalg.norm(vec)

    def interact(self, ray: Ray) -> List[Ray]:
        # Reflect direction
        incident = ray.direction
        reflect_dir = incident - 2 * np.dot(incident, self.normal) * self.normal
        new_ray = Ray(ray.position.copy(), reflect_dir, ray.wavelength, ray.polarization, ray.intensity)
        return [new_ray]

    def intersect_distance(self, ray: Ray) -> float | None:
        denom = np.dot(self.normal, ray.direction)
        if np.isclose(denom, 0.0):
            return None
        t = np.dot(self.normal, self.position - ray.position) / denom
        return t if t > 1e-9 else None


class Prism(OpticalElement):
    """Single-plane refraction, rough prism face model."""

    def __init__(self, normal: np.ndarray, n_out: float, n_in: float = 1.0):
        """normal points into prism; n_out inside, n_in outside (air=1)."""
        self.normal = self._normalize(normal)
        self.n_out = n_out
        self.n_in = n_in

    @staticmethod
    def _normalize(vec):
        vec = np.asarray(vec, dtype=float)
        return vec / np.linalg.norm(vec)

    def interact(self, ray: Ray) -> List[Ray]:
        incident = self._normalize(ray.direction)
        cos_theta_i = -np.dot(incident, self.normal)
        if cos_theta_i < 0:
            # Inside‐out case, swap indices & flip normal
            self.normal = -self.normal
            self.n_in, self.n_out = self.n_out, self.n_in
            cos_theta_i = -np.dot(incident, self.normal)

        sin_theta_i2 = max(0.0, 1.0 - cos_theta_i ** 2)
        eta = self.n_in / self.n_out
        sin_theta_t2 = eta ** 2 * sin_theta_i2

        if sin_theta_t2 > 1.0:
            # Total internal reflection
            reflect_dir = incident + 2 * cos_theta_i * self.normal
            return [Ray(ray.position.copy(), reflect_dir, ray.wavelength, ray.polarization, ray.intensity)]

        cos_theta_t = np.sqrt(1.0 - sin_theta_t2)
        refract_dir = eta * incident + (eta * cos_theta_i - cos_theta_t) * self.normal
        refract_dir = self._normalize(refract_dir)
        return [Ray(ray.position.copy(), refract_dir, ray.wavelength, ray.polarization, ray.intensity)]


class MetalFilm(OpticalElement):
    """Metal film for SPR (Kretschmann 3-layer setup)."""

    def __init__(
        self,
        thickness_nm: float,
        n_metal: Union[complex, Callable[[float], complex]] = gold_n_complex_scalar,
        n_prism: float = 1.515,
        n_sample: float = 1.33,
        normal: Optional[np.ndarray] = None,
    ):
        self.d = thickness_nm
        self.n_metal = n_metal  # complex n or callable
        self.n_prism = n_prism
        self.n_sample = n_sample
        # default normal +y
        self.normal = self._normalize(normal if normal is not None else np.array([0.0, 1.0]))
        self.position = np.zeros(2)  # plane through origin

    @staticmethod
    def _normalize(vec):
        vec = np.asarray(vec, dtype=float)
        return vec / np.linalg.norm(vec)

    # -- n lookup --
    def _n_m(self, wavelength_nm: float) -> complex:
        return self.n_metal(wavelength_nm) if callable(self.n_metal) else self.n_metal

    # -- TM reflectance via T-matrix --
    def _tm_reflectance(self, wavelength_nm: float, theta1: float) -> float:
        return float(np.abs(self._tm_r_coeff(wavelength_nm, theta1)) ** 2)

    def _tm_r_coeff(self, wavelength_nm: float, theta1: float) -> complex:
        """Return complex reflection coefficient r (TM)."""
        n1 = self.n_prism
        n2 = self._n_m(wavelength_nm)
        n3 = self.n_sample

        sin_theta1 = np.sin(theta1)
        sin_theta2 = n1 / n2 * sin_theta1
        sin_theta3 = n1 / n3 * sin_theta1

        cos_theta1 = np.cos(theta1)
        cos_theta2 = np.lib.scimath.sqrt(1 - sin_theta2 ** 2)
        cos_theta3 = np.lib.scimath.sqrt(1 - sin_theta3 ** 2)

        r12 = (n2 * cos_theta1 - n1 * cos_theta2) / (n2 * cos_theta1 + n1 * cos_theta2)
        r23 = (n3 * cos_theta2 - n2 * cos_theta3) / (n3 * cos_theta2 + n2 * cos_theta3)

        k0 = 2 * np.pi / (wavelength_nm * 1e-9)
        beta = n2 * cos_theta2 * k0 * self.d * 1e-9
        exp_term = np.exp(-2j * beta)
        return (r12 + r23 * exp_term) / (1 + r12 * r23 * exp_term)

    # -- phase shift --
    def get_phase_shift(self, wavelength_nm: float, theta_rad: float) -> float:
        """Return phase angle (rad) of reflection coefficient."""
        r = self._tm_r_coeff(wavelength_nm, theta_rad)
        return float(np.angle(r))

    # -- public interface --
    def get_reflectance(self, ray: Ray) -> float:
        if ray.polarization.upper() != "TM":
            return 1.0  # TE no SPR

        # incidence angle
        cos_theta1 = np.abs(np.dot(self._normalize(ray.direction), self.normal))
        theta1 = np.arccos(np.clip(cos_theta1, -1.0, 1.0))
        return self._tm_reflectance(ray.wavelength, theta1)

    def field_enhancement(self, ray: Ray) -> float:
        """Return field enhancement |E|² (simplified ≈ 1/R)."""
        R = self.get_reflectance(ray)
        if R == 0:
            return float('inf')
        return 1.0 / R

    def interact(self, ray: Ray) -> List[Ray]:
        R = self.get_reflectance(ray)
        incident = ray.direction
        reflect_dir = incident - 2 * np.dot(incident, self.normal) * self.normal
        new_intensity = ray.intensity * R
        new_ray = Ray(ray.position.copy(), reflect_dir, ray.wavelength, ray.polarization, new_intensity)
        return [new_ray]

    def intersect_distance(self, ray: Ray) -> float | None:
        denom = np.dot(self.normal, ray.direction)
        if np.isclose(denom, 0.0):
            return None
        t = np.dot(self.normal, self.position - ray.position) / denom
        return t if t > 1e-9 else None 
"""porous.py
Porous electrode effective-index models.
"""
from __future__ import annotations

from typing import Callable
import numpy as np

__all__ = [
    "bruggeman",
    "maxwell_garnett",
    "PorousElectrode",
    "porosity_soc_to_n",
]


def bruggeman(n_matrix: complex, n_pore: complex, phi: float) -> complex:
    """Bruggeman EMA: phi pores, (1-phi) matrix -> n_eff."""
    eps_m = n_matrix ** 2
    eps_p = n_pore ** 2
    # solve for eps (implicit); use Newton
    eps_eff = eps_m * (1 - phi) + eps_p * phi  # initial guess
    for _ in range(20):
        f = (1 - phi) * (eps_m - eps_eff) / (eps_m + 2 * eps_eff) + phi * (
            eps_p - eps_eff
        ) / (eps_p + 2 * eps_eff)
        df = (
            -(1 - phi) * (eps_m + 2 * eps_eff) - 2 * (1 - phi) * (eps_m - eps_eff)
        ) / (eps_m + 2 * eps_eff) ** 2 + (
            -phi * (eps_p + 2 * eps_eff) - 2 * phi * (eps_p - eps_eff)
        ) / (eps_p + 2 * eps_eff) ** 2
        eps_eff = eps_eff - f / df
    return np.sqrt(eps_eff)


def maxwell_garnett(n_matrix: complex, n_incl: complex, phi_incl: float) -> complex:
    """Maxwell-Garnett EMA (inclusions in host)."""
    eps_h = n_matrix ** 2
    eps_i = n_incl ** 2
    eps_eff = eps_h * (
        (2 * eps_h + eps_i) + 2 * phi_incl * (eps_i - eps_h)
    ) / (
        (2 * eps_h + eps_i) - phi_incl * (eps_i - eps_h)
    )
    return np.sqrt(eps_eff)


class PorousElectrode:
    """Porous electrode with SOC-dependent effective index.

    Parameters
    ----------
    n_solid : complex
        Index of active material (e.g., NMC).
    n_pore  : complex, default 1.0 (air) or electrolyte index.
    model   : "bruggeman" or "mg" (maxwell-garnett).
    phi0    : initial porosity at 0% SOC.
    """

    def __init__(
        self,
        n_solid: complex | Callable[[float], complex],
        n_pore: complex = 1.0,
        model: str = "bruggeman",
        phi0: float = 0.3,
    ):
        self.n_solid = n_solid
        self.n_pore = n_pore
        self.model = model.lower()
        self.phi0 = phi0

    # --- effective index as function of SOC (0-1) ---
    def n_eff(self, soc: float) -> complex:
        """Return n_eff for given state of charge (0-1)."""
        # assume porosity drops linearly with SOC (pore filled by Li)
        phi = self.phi0 * (1 - soc)
        n_matrix = self.n_solid if not callable(self.n_solid) else self.n_solid(632.8)
        if self.model.startswith("br"):  # bruggeman
            return bruggeman(n_matrix, self.n_pore, phi)
        else:  # mg
            return maxwell_garnett(n_matrix, self.n_pore, 1 - phi)  # inclusions=solid


# --- simple mapping: porosity & SOC to n ---

def porosity_soc_to_n(phi0: float, soc: float, n_solid: float, n_pore: float = 1.0) -> float:
    """Shortcut wrapper using Bruggeman."""
    phi = phi0 * (1 - soc)
    return float(np.real(bruggeman(n_solid, n_pore, phi))) 
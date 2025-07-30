"""stack"""
from __future__ import annotations

from typing import Sequence, Callable, Dict, Any

import numpy as np

# ---- Layer dictionary ----
# First and last may use d=None to mean semi-infinite.
Layer = Dict[str, Any]

__all__ = ["multi_layer_stack", "temp_drift"]


# ---- Core algorithm ----

from typing import Any as _Any

def _layer_params(layer: Layer, wavelength_nm: float) -> _Any:
    n = layer["n"]
    return n(wavelength_nm) if callable(n) else n


def _q_p(n: complex, cos_theta: complex, pol: str) -> complex:
    """Layer impedance q_j: TE n·cosθ, TM n/ cosθ"""
    if pol.upper() == "TE":
        return n * cos_theta
    else:  # TM
        return n / cos_theta


def _char_matrix(n: complex, d_nm: float, cos_theta: complex, wavelength_nm: float, pol: str):
    """Single-layer characteristic matrix (2×2)"""
    delta = 2 * np.pi * n * cos_theta * d_nm * 1e-9 / (wavelength_nm * 1e-9)
    q = _q_p(n, cos_theta, pol)
    cos_d = np.cos(delta)
    sin_d = 1j * np.sin(delta)
    return np.array([[cos_d, sin_d / q], [q * sin_d, cos_d]], dtype=complex)


def _global_transfer_matrix(layers: Sequence[Layer], wavelength_nm: float, theta_in: float, pol: str):
    # compute angle in each layer
    n0 = _layer_params(layers[0], wavelength_nm)
    sin_theta0 = np.sin(theta_in)

    M = np.identity(2, dtype=complex)
    cos_list = []
    for idx, layer in enumerate(layers):
        n_j = _layer_params(layer, wavelength_nm)
        if idx == 0:
            cos_j = np.cos(theta_in)
        else:
            # Snell
            sin_theta_j = n0 / n_j * sin_theta0
            cos_j = np.lib.scimath.sqrt(1 - sin_theta_j ** 2)
        cos_list.append(cos_j)

    # multiply internal layer matrices (excluding first, last)
    for j in range(1, len(layers) - 1):
        layer = layers[j]
        d_nm = layer.get("d")
        if d_nm is None:
            continue
        n_j = _layer_params(layer, wavelength_nm)
        M_j = _char_matrix(n_j, d_nm, cos_list[j], wavelength_nm, pol)
        M = M @ M_j
    return M, cos_list


def _reflectance(layers: Sequence[Layer], wavelength_nm: float, theta_in: float, pol: str = "TM") -> float:
    M, cos_list = _global_transfer_matrix(layers, wavelength_nm, theta_in, pol)
    n0 = _layer_params(layers[0], wavelength_nm)
    nN = _layer_params(layers[-1], wavelength_nm)
    cos0 = cos_list[0]
    cosN = cos_list[-1]
    q0 = _q_p(n0, cos0, pol)
    qN = _q_p(nN, cosN, pol)

    (M11, M12), (M21, M22) = M
    r = (q0 * (M11 + M12 * qN) - (M21 + M22 * qN)) / (
        q0 * (M11 + M12 * qN) + (M21 + M22 * qN)
    )
    return float(abs(r) ** 2)


# ---- Public API ----

def multi_layer_stack(layers: Sequence[Layer], pol: str = "TM") -> Callable[[float, float], float]:
    """Return callable reflect(wl_nm, theta_rad) -> R"""

    def _func(wavelength_nm: float, theta_rad: float) -> float:
        return _reflectance(layers, wavelength_nm, theta_rad, pol)

    return _func


# ---- Temperature drift ----

def temp_drift(layers: Sequence[Layer], delta_T: float, pol: str = "TM") -> Callable[[float, float], float]:
    """Return reflectance function after ΔT drift"""

    new_layers: list[Layer] = []
    for layer in layers:
        dn_dT = layer.get("dn_dT")
        if dn_dT is None:
            new_layers.append(layer)
            continue

        base_n = layer["n"]

        def _n_T_factory(base=base_n, dn=dn_dT):
            if callable(base):
                return lambda wl_nm: base(wl_nm) + dn * delta_T
            else:
                return base + dn * delta_T

        new_layer = layer.copy()
        new_layer["n"] = _n_T_factory()
        new_layers.append(new_layer)

    return multi_layer_stack(new_layers, pol=pol) 

__all__.append("thermo_mech_drift")

def thermo_mech_drift(layers: Sequence[Layer], delta_T: float, delta_sigma: float, pol: str = "TM"):
    """Return reflectance after simultaneous temp (ΔT, K) and stress (Δσ, MPa) drift"""
    new_layers = []
    for layer in layers:
        dn_T = layer.get("dn_dT", 0.0)
        dn_S = layer.get("dn_dSigma", 0.0)
        if dn_T == 0 and dn_S == 0:
            new_layers.append(layer)
            continue
        base_n = layer["n"]
        def _n_drift_factory(base=base_n, a=dn_T, b=dn_S):
            if callable(base):
                return lambda wl_nm: base(wl_nm) + complex(a * delta_T + b * delta_sigma)  # type: ignore[operator]
            else:
                try:
                    base_val: complex = complex(base)
                except Exception:
                    base_val = 0.0
                return base_val + complex(a * delta_T + b * delta_sigma)  # type: ignore[operator] 
        new_l = layer.copy()
        new_l["n"] = _n_drift_factory()
        new_layers.append(new_l)
    return multi_layer_stack(new_layers, pol=pol) 
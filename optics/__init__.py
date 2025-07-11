from .ray import Ray
from .elements import OpticalElement, Mirror, Prism, MetalFilm
from .tracer import RayTracer
from .materials import gold_n_complex_scalar as gold_n_complex
from .geometry import default_mirrors, generate_rays
from .optimizer import optimize_system, collect_hits, spot_uniformity_metric
from .stack import multi_layer_stack, temp_drift
from .materials import sellmeier_coeff_to_n
from .metrics import peak_slope, fwhm
from .utils import export_results
from .porous import PorousElectrode, porosity_soc_to_n

__all__ = [
    "Ray",
    "OpticalElement",
    "Mirror",
    "Prism",
    "MetalFilm",
    "RayTracer",
    "gold_n_complex",
]
__all__.extend(["default_mirrors","generate_rays","optimize_system"])
__all__.extend(["multi_layer_stack", "temp_drift", "sellmeier_coeff_to_n"])
__all__.extend(["peak_slope", "fwhm", "gaussian_beam", "export_results"])
__all__.extend(["collect_hits", "spot_uniformity_metric"])
__all__.extend(["PorousElectrode", "porosity_soc_to_n"])
__all__.extend(["phase_shift_sensitivity"]) 
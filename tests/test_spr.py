import numpy as np

from optics.elements import MetalFilm
from optics.materials import gold_n_complex_scalar
from optics.ray import Ray


def test_spr_reflectance_curve():
    film = MetalFilm(thickness_nm=50, n_metal=gold_n_complex_scalar)
    wl = 632.8  # nm
    angles = np.deg2rad(np.linspace(30, 80, 200))
    R_list = [film._tm_reflectance(wl, th) for th in angles]
    R_min = min(R_list)
    assert R_min < 0.2, "Min reflectance should drop below 0.2"


def test_field_enhancement():
    film = MetalFilm(thickness_nm=50, n_metal=gold_n_complex_scalar)
    ray = Ray(position=np.zeros(2), direction=np.array([0, 1]), wavelength=632.8, polarization="TM")
    R = film.get_reflectance(ray)
    enhancement = film.field_enhancement(ray)
    assert np.isclose(enhancement, 1 / R), "Enhancement should approx 1/R" 
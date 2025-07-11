import numpy as np
import matplotlib.pyplot as plt

from optics import multi_layer_stack, temp_drift, peak_slope
from optics.materials import gold_n_complex_scalar

# Layers: prism / Au / PDA / PEI / water
layers_ref = [
    {"n": 1.515, "d": None},
    {"n": gold_n_complex_scalar, "d": 50},
    {"n": 1.60, "d": 0},  # no coat
    {"n": 1.33, "d": None},
]
layers_100 = [
    {"n": 1.515, "d": None},
    {"n": gold_n_complex_scalar, "d": 50},
    {"n": 1.60, "d": 100},  # 100 nm PDA+PEI
    {"n": 1.33, "d": None},
]

reflect_ref = multi_layer_stack(layers_ref)
reflect_100 = multi_layer_stack(layers_100)

angles = np.deg2rad(np.linspace(40,75,400))
R_ref = [reflect_ref(632.8, th) for th in angles]
R_100 = [reflect_100(632.8, th) for th in angles]

plt.plot(np.rad2deg(angles), R_ref, label="0 nm")
plt.plot(np.rad2deg(angles), R_100, label="100 nm coat")
plt.xlabel("Angle (deg)")
plt.ylabel("Reflectance")
plt.legend()
plt.show()

print("delta angle @ min R: ")
min_ref_idx = int(np.argmin(R_ref))
min_100_idx = int(np.argmin(R_100))
print(np.rad2deg(angles[min_100_idx] - angles[min_ref_idx])) 
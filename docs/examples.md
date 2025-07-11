# Example runs

## 1. Plot SPR curve

```python
from optics import MetalFilm, peak_slope
from optics.materials import gold_n_complex_scalar
import numpy as np, matplotlib.pyplot as plt

film = MetalFilm(50, gold_n_complex_scalar)
angles = np.deg2rad(np.linspace(40,75,300))
R = [film._tm_reflectance(632.8, th) for th in angles]
plt.plot(np.rad2deg(angles), R)
plt.show()
print('slope', peak_slope(np.rad2deg(angles), R))
```

## 2. Ray-trace multi-mirror bundle

```python
from optics import default_mirrors, generate_rays, RayTracer
mirrors = default_mirrors(3)
rays = generate_rays(200, 10)
tracer = RayTracer(mirrors)
paths = tracer.trace(rays)
``` 
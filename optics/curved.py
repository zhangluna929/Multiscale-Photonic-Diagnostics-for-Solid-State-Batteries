"""curved"""
from __future__ import annotations
from typing import List, Callable
import numpy as np

from .ray3d import Ray3D
from .elements import OpticalElement

__all__ = ["CylindricalShell", "Tracer3D"]


class CylindricalShell(OpticalElement):
    """Simple y-axis cylinder shell, reflective interior"""

    def __init__(self, radius: float, center: np.ndarray | None = None):
        self.R = radius
        self.center = np.asarray(center if center is not None else [0.0, 0.0, 0.0], dtype=float)

    # overwrite 3D version
    def intersect_distance(self, ray: Ray3D) -> float | None:  # type: ignore[override]
        # Transform to cylinder coords
        p = ray.pos - self.center
        d = ray.dir
        # ignore y component: cylinder along y
        a = d[0] ** 2 + d[2] ** 2
        if a == 0:
            return None
        b = 2 * (p[0] * d[0] + p[2] * d[2])
        c = p[0] ** 2 + p[2] ** 2 - self.R ** 2
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        t_candidates = [t for t in (t1, t2) if t > 1e-6]
        return min(t_candidates) if t_candidates else None

    def interact(self, ray: Ray3D):  # type: ignore[override]
        # Reflect specularly
        hit_pos = ray.pos  # expected to call after propagate
        n = hit_pos - self.center
        n[1] = 0.0
        n = n / np.linalg.norm(n)
        new_dir = ray.dir - 2 * np.dot(ray.dir, n) * n
        return [Ray3D(hit_pos.copy(), new_dir, ray.wavelength, ray.intensity)]


class Tracer3D:
    """Minimal 3-D ray tracer (curved shells + mirrors planes TBD)"""

    def __init__(self, elements: List[OpticalElement]):
        self.elements = elements

    def trace(self, rays: List[Ray3D], max_hits: int = 10):
        paths = []
        for ray in rays:
            pts = [ray.pos.copy()]
            cur = ray
            for _ in range(max_hits):
                best_t = None
                hit_elem = None
                for e in self.elements:
                    t = e.intersect_distance(cur)  # type: ignore[arg-type]
                    if t is not None and (best_t is None or t < best_t):
                        best_t = t
                        hit_elem = e
                if best_t is None:
                    break
                cur = cur.propagate(best_t)
                pts.append(cur.pos.copy())
                new_rays = hit_elem.interact(cur)  # type: ignore[arg-type]
                cur = new_rays[0]
            paths.append(pts)
        return paths 
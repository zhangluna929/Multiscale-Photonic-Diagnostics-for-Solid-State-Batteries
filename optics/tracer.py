from __future__ import annotations

"""tracer"""
from typing import List, Sequence
import numpy as np

from .ray import Ray
from .elements import OpticalElement


class RayTracer:
    """Multi-element ray tracer"""

    def __init__(self, elements: Sequence[OpticalElement]):
        self.elements = elements

    def trace(self, rays: List[Ray], max_interactions: int = 10):
        """Trace list of rays, return list of point arrays per ray"""
        all_paths: List[List[np.ndarray]] = []
        for ray in rays:
            path = [ray.position.copy()]
            cur_ray = ray
            for _ in range(max_interactions):
                # find nearest intersection
                min_t = None
                hit_elem = None
                for elem in self.elements:
                    t = elem.intersect_distance(cur_ray)
                    if t is None:
                        continue
                    if min_t is None or t < min_t:
                        min_t = t
                        hit_elem = elem

                if min_t is None or hit_elem is None:
                    # no more hits, ray goes to infinity
                    break

                # propagate to hit
                cur_ray = cur_ray.propagate(min_t)
                path.append(cur_ray.position.copy())

                # element interaction
                new_rays = hit_elem.interact(cur_ray)
                if not new_rays:
                    break
                # keep first ray only (no splitting yet)
                cur_ray = new_rays[0]
            all_paths.append(path)
        return all_paths 
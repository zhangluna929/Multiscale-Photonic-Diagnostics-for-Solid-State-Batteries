"""optimizer.py
Weighted multi-objective tweak – spot uniformity + SPR depth.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import optuna  # type: ignore[import-not-found]

from .geometry import default_mirrors, generate_rays, SCREEN_Y
from .tracer import RayTracer
from .elements import MetalFilm
from .spectrum import reflectance_spectrum
from .materials import gold_n_complex_scalar


def spot_uniformity_metric(x_hits: np.ndarray) -> float:
    """Return std/width metric (smaller is better)."""
    if x_hits.size < 2:
        return 1.0
    return float(np.std(x_hits) / (np.ptp(x_hits) + 1e-9))


def collect_hits(tracer_paths, final_dirs):
    # extend each ray to screen y
    hits = []
    for path, dir_vec in zip(tracer_paths, final_dirs):
        start = path[-1]
        if dir_vec[1] == 0:
            continue
        t = (SCREEN_Y - start[1]) / dir_vec[1]
        hits.append(start + dir_vec * t)
    return np.array(hits)


def optimize_system(
    n_trials: int = 100,
    output_path: str | Path = "results_optuna.json",
):
    """Joint optimization of mirror count, spread, Au thickness."""

    def objective(trial: optuna.Trial):
        num_mirrors = trial.suggest_int("num_mirrors", 1, 6)
        theta_spread = trial.suggest_uniform("theta_spread", 5.0, 20.0)
        thickness = trial.suggest_uniform("thickness", 30.0, 70.0)

        mirrors = default_mirrors(num_mirrors)
        film = MetalFilm(thickness_nm=thickness, n_metal=gold_n_complex_scalar)
        elements = mirrors  # ray trace for spot only; SPR eval separately

        rays = generate_rays(200, theta_spread)
        tracer = RayTracer(elements)
        paths = tracer.trace(rays, max_interactions=3)
        final_dirs = [rays[i].direction for i in range(len(rays))]
        hits = collect_hits(paths, final_dirs)
        if hits.size == 0:
            uniformity = 1.0
        else:
            uniformity = spot_uniformity_metric(hits[:, 0])

        # SPR depth: min R in 60±5° window
        angles = np.deg2rad(np.linspace(55, 65, 50))
        R_curve = [film._tm_reflectance(632.8, th) for th in angles]
        spr_depth = min(R_curve)

        # weighted objective
        score = uniformity + spr_depth
        trial.set_user_attr("uniformity", uniformity)
        trial.set_user_attr("spr_depth", spr_depth)
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # save result
    best = study.best_trial
    result: Dict[str, Any] = {
        "params": best.params,
        "uniformity": best.user_attrs["uniformity"],
        "spr_depth": best.user_attrs["spr_depth"],
        "score": best.value,
    }
    Path(output_path).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result 
from __future__ import annotations

"""visualize"""
from typing import List

import numpy as np
import matplotlib.pyplot as plt


def plot_rays(paths: List[List[np.ndarray]]):
    """Plot ray paths"""
    plt.figure(figsize=(6, 4))
    for path in paths:
        pts = np.array(path)
        plt.plot(pts[:, 0], pts[:, 1], '-', alpha=0.7)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Ray paths")
    plt.tight_layout()
    plt.show()


def plot_spot_histogram(hits: np.ndarray, bins: int = 60):
    """Plot histogram of hit X positions"""
    if hits.size == 0:
        print("No hit points to display.")
        return
    plt.figure(figsize=(5, 3))
    plt.hist(hits[:, 0], bins=bins, alpha=0.8, color='royalblue')
    plt.xlabel("Screen X Position")
    plt.ylabel("Intensity")
    plt.title("Spot Intensity Distribution")
    plt.tight_layout()
    plt.show() 
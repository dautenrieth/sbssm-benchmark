"""
Project: Toward Standardized Benchmarking of Search-Based Scenario Selection Methods in Autonomous System Validation
Version: 1.0.0

Description:
    Implements three non-adaptive baseline selectors that draw scenario points
    from the criticality space without feedback from previous queries:

    - RandomSelector    — pure uniform random sampling.
    - SobolSelector     — scrambled Sobol quasi-random sequence (low-discrepancy).
    - LatinHypercubeSelector — stratified Latin Hypercube Sampling (LHS).

    All three follow the same interface: instantiate with a Space and a budget,
    call ``select()``, and the chosen points are stored in the space.

    - key role: Space-filling and uniform baseline samplers used as reference
                methods and for general coverage metrics.
    - dependency: numpy, scipy.stats.qmc, matplotlib, base.BaseSelector
    - output: List of selected scenario-point tuples stored in space.selected_points.

Usage:
    from selector_random import RandomSelector, SobolSelector, LatinHypercubeSelector
    selector = RandomSelector(space, n_select=100)
    selector.select()
"""

import numpy as np
from base import BaseSelector
from scipy.stats import qmc
import matplotlib.pyplot as plt


def plot_points2d(points, title="Selected points (2D projection)"):
    """
    Scatter plot of selected points projected onto the first two dimensions.

    Parameters
    ----------
    points : iterable of tuple
        Selected scenario coordinate tuples (must have at least 2 dimensions).
    title : str, optional
        Plot title (default ``"Selected points (2D projection)"``).

    Raises
    ------
    ValueError
        If ``points`` has fewer than 2 dimensions.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("plot_points2d requires points with >= 2 dimensions.")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pts[:, 0], pts[:, 1], s=18, alpha=0.8)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    plt.show()


class RandomSelector(BaseSelector):
    """Selects ``n_select`` scenario points via uniform random sampling."""

    def select(self) -> list[tuple]:
        """
        Draw ``n_select`` points uniformly at random from the space bounds.

        Returns
        -------
        list of tuple
            Randomly sampled and rounded scenario parameter points.
        """
        bounds = self.space.dimensions
        d = len(bounds)

        samples = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(self.n_select, d),
        )
        rounded = [
            tuple(np.round(p, self.space.decimal_precision)) for p in samples
        ]
        self.space.get_values_for_points(rounded)
        return rounded


class SobolSelector(BaseSelector):
    """Selects ``n_select`` scenario points using a scrambled Sobol sequence."""

    def select(self) -> list[tuple]:
        """
        Draw ``n_select`` points from a scrambled Sobol quasi-random sequence.

        Sobol sequences are low-discrepancy and provide better space coverage
        than pure random sampling for the same budget.

        Returns
        -------
        list of tuple
            Sobol-sampled and rounded scenario parameter points.
        """
        bounds = self.space.dimensions
        d = len(bounds)

        sampler = qmc.Sobol(d=d, scramble=True)
        sample = sampler.random(n=self.n_select)

        lower_bounds, upper_bounds = zip(*bounds)
        scaled = qmc.scale(sample, lower_bounds, upper_bounds)

        rounded = [
            tuple(np.round(p, self.space.decimal_precision)) for p in scaled
        ]
        self.space.get_values_for_points(rounded)
        return rounded


class LatinHypercubeSelector(BaseSelector):
    """Selects ``n_select`` scenario points using Latin Hypercube Sampling."""

    def select(self) -> list[tuple]:
        """
        Draw ``n_select`` points from a Latin Hypercube design.

        LHS stratifies the unit hypercube so that each marginal stratum is
        sampled exactly once, improving uniformity over pure random sampling.

        Returns
        -------
        list of tuple
            LHS-sampled and rounded scenario parameter points.
        """
        bounds = self.space.dimensions
        d = len(bounds)

        sampler = qmc.LatinHypercube(d=d)
        sample = sampler.random(n=self.n_select)

        lower_bounds, upper_bounds = zip(*bounds)
        scaled = qmc.scale(sample, lower_bounds, upper_bounds)

        rounded = [
            tuple(np.round(p, self.space.decimal_precision)) for p in scaled
        ]
        self.space.get_values_for_points(rounded)
        return rounded

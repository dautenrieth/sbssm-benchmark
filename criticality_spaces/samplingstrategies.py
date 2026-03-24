"""
Project: Toward Standardized Benchmarking of Search-Based Scenario Selection Methods in Autonomous System Validation
Version: 1.0.0

Description:
    Provides sampling strategies for drawing scenario parameter points from a
    bounded input space. Strategies differ in their space-filling properties and
    are interchangeable via the common interface used in run_demo.py.

    - key role: Generates the sample sets evaluated by Space.get_values_for_points()
                and benchmarked by the Metrics suite.
    - dependency: numpy, scipy.stats.qmc
    - output: List of d-tuples or ndarray of shape (N, d).

Strategies:
    random_sampling             — uniform random draw from a discretised grid
    grid_latin_hypercube_sampling — strength-2 LHS restricted to grid points
    latin_hypercube_sampling    — continuous LHS scaled to arbitrary bounds (recommended)

Usage:
    import samplingstrategies as ss
    samples = ss.latin_hypercube_sampling([(0, 10), (0, 10)], n_samples=50)
"""

import numpy as np
from scipy.stats import qmc

def random_sampling(dimensions, n_samples, n_points):
    """
    Draw samples uniformly at random from a discretised grid (with replacement).

    Parameters
    ----------
    dimensions : list of (float, float)
        Per-dimension (min, max) bounds.
    n_samples : int
        Number of points to draw.
    n_points : int
        Grid resolution per dimension (linspace nodes).

    Returns
    -------
    list of tuple
        ``n_samples`` points as d-tuples.
    """
    grids = [np.linspace(start, stop, n_points) for (start, stop) in dimensions]
    samples = np.array([np.random.choice(grid, n_samples, replace=True) for grid in grids])
    return [tuple(sample) for sample in np.array(samples).T]

def grid_latin_hypercube_sampling(dimensions, n_samples, n_points):
    """
    Strength-2 Latin Hypercube samples snapped to a discretised grid.
    Ensures stratified coverage while keeping samples on predefined grid nodes,
    which is useful when function evaluation is only valid at grid coordinates.

    Parameters
    ----------
    dimensions : list of (float, float)
        Per-dimension (min, max) bounds.
    n_samples : int
        Number of samples; must satisfy n_samples <= n_points.
    n_points : int
        Grid resolution per dimension.

    Returns
    -------
    list of tuple
        ``n_samples`` points as d-tuples, each snapped to the nearest grid node.

    Raises
    ------
    ValueError
        If n_samples > n_points.
    """
    if n_samples > n_points:
        raise ValueError("Latin hypercube sampling requires that n_samples <= n_points per dimension")
    sampler = qmc.LatinHypercube(d=len(dimensions), strength=2)
    sample = sampler.random(n_samples)
    # Generate a grid for each dimension and select samples based on Latin hypercube indices
    grids = [np.linspace(start, stop, n_points) for (start, stop) in dimensions]
    # Scale the hypercube output to map to the indices of the grid
    grid_indices = np.floor(sample * n_points).astype(int)
    samples = [grids[dim][indices] for dim, indices in enumerate(grid_indices.T)]
    # Transpose to get samples as list of tuples
    return [tuple(sample) for sample in np.array(samples).T]

def latin_hypercube_sampling(bounds, n_samples, seed=42, as_array=False):
    """
    Generate Latin Hypercube Samples within specified bounds.

    Parameters
    ----------
    bounds : list of tuple(float, float)
        A list of (min, max) tuples for each dimension.
    n_samples : int
        Number of samples to generate.
    seed : int or None, optional
        Random seed for reproducibility.
    as_array : bool, default False
        If True, return a NumPy array. If False, return list of tuples.

    Returns
    -------
    samples : list of tuples or np.ndarray
        The scaled Latin Hypercube samples.
    """
    d = len(bounds)
    sampler = qmc.LatinHypercube(d=d, seed=seed, scramble=False)
    sample_unit = sampler.random(n_samples)  # in [0, 1)^d

    # Extract lower and upper bounds for each dimension
    lower_bounds, upper_bounds = zip(*bounds)

    # Scale to given bounds
    scaled = qmc.scale(sample_unit, lower_bounds, upper_bounds)

    return scaled if as_array else [tuple(row) for row in scaled]


def generate_samples(self, n_samples):
    random_sampling(self.dimensions, n_samples)
    latin_hypercube_sampling(self.dimensions, n_samples)
    return
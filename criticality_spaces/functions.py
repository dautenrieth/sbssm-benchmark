"""
Project: Toward Standardized Benchmarking of Search-Based Scenario Selection Methods in Autonomous System Validation
Version: 1.0.0

Description:
    Defines the primitive criticality function types used to compose scenario
    spaces. Each class represents a distinct analytic shape (hypercube,
    Gaussian, linear ramp, noise, sinusoidal) and exposes a common interface
    for evaluation on meshgrids and arbitrary point sets.

    - key role: Building blocks for JSON-defined criticality spaces; consumed
                by functionloader and Space.
    - dependency: numpy, scipy.stats
    - output: Per-point criticality values in [0, 1] (or unbounded for Noise/Sinus).

Usage:
    from functions import Gaussian, Ramp, Noise, Sinus, StaticHypercube
    # Instances are normally created via functionloader.load_functions_from_json().
"""


import numpy as np
from scipy.stats import multivariate_normal


class StaticHypercube:
    """
    Axis-aligned hypercube that returns a constant criticality value inside its
    bounds and 0 outside. Useful for modelling hard exclusion or saturation zones.
    """

    def __init__(self, dimensions, static_value=1.0):
        """
        Parameters
        ----------
        dimensions : list of (float, float)
            Per-dimension (start, stop) bounds of the hypercube.
        static_value : float
            Criticality value assigned to points inside the hypercube.
        """
        self.dimensions = dimensions
        self.static_value = static_value

    def calculate_relevant_values(self, meshgrid):
        """
        Evaluate on a meshgrid; returns a boolean mask and the constant values
        for all points inside the hypercube.
        """
        conditions = [
            np.logical_and(grid >= start, grid <= stop)
            for (grid, (start, stop)) in zip(meshgrid, self.dimensions)
        ]
        subgrid_mask = np.logical_and.reduce(conditions)
        values = np.full(np.count_nonzero(subgrid_mask), self.static_value)
        return subgrid_mask, values

    def get_values_for_points(self, points):
        """
        Evaluate at arbitrary points (N, d); returns static_value inside,
        0 outside.
        """
        conditions = [
            np.logical_and(points[:, i] >= start, points[:, i] <= stop)
            for i, (start, stop) in enumerate(self.dimensions)
        ]
        inside_bounds = np.logical_and.reduce(conditions)
        values = np.zeros(points.shape[0])
        values[inside_bounds] = self.static_value
        return values


class Gaussian:
    """
    Multivariate Gaussian criticality bump with diagonal covariance.
    Evaluation is restricted to a bounding box of ±range_factor·σ around the
    mean to avoid unnecessary computation in the tails.
    """

    def __init__(
        self,
        mean,
        std_dev,
        max_amplitude=1.0,
        range_factor=3,
        scipy=False,
        active_dims=None,
    ):
        """
        Parameters
        ----------
        mean : array-like of float
            Centre of the Gaussian in each active dimension.
        std_dev : array-like of float
            Standard deviation per active dimension (diagonal covariance).
        max_amplitude : float
            Peak value of the Gaussian (scales the output to [0, max_amplitude]).
        range_factor : float
            Half-width of the evaluation bounding box in units of std_dev.
        scipy : bool
            If True, use scipy.stats.multivariate_normal (supports full cov).
            If False, use the faster manual formula (diagonal cov only).
        active_dims : list of int, optional
            Indices of input dimensions used for evaluation; defaults to all.
        """
        self.mean = np.array(mean)
        self.std_dev = np.array(std_dev)
        self.max_amplitude = max_amplitude
        self.range_factor = range_factor
        self.scipy = scipy
        self.active_dims = (
            active_dims if active_dims is not None else list(range(len(mean)))
        )
        self.bounds = self.calculate_boundaries()

    def _project(self, mesh_or_points):
        return (
            [mesh_or_points[i] for i in self.active_dims]
            if isinstance(mesh_or_points, (list, tuple))
            else mesh_or_points[:, self.active_dims]
        )

    def calculate_boundaries(self):
        return [
            (m - self.range_factor * s, m + self.range_factor * s)
            for m, s in zip(self.mean, self.std_dev)
        ]

    def calculate_subgrid(self, meshgrid):
        meshgrid_proj = self._project(meshgrid)
        conditions = [
            np.logical_and(grid >= start, grid <= stop)
            for grid, (start, stop) in zip(meshgrid_proj, self.bounds)
        ]
        return np.logical_and.reduce(conditions)

    def calculate_relevant_values(self, meshgrid, cov=None):
        meshgrid_proj = self._project(meshgrid)
        subgrid_mask = self.calculate_subgrid(meshgrid)

        relevant_points = np.stack([g[subgrid_mask] for g in meshgrid_proj], axis=-1)
        if not self.scipy:
            distances_squared = np.sum(
                ((relevant_points - self.mean) ** 2) / (2 * self.std_dev**2), axis=1
            )
            return subgrid_mask, self.max_amplitude * np.exp(-distances_squared)
        else:
            if cov is None:
                cov = np.diag(self.std_dev**2)
            values = multivariate_normal.pdf(relevant_points, mean=self.mean, cov=cov)
            return subgrid_mask, self.max_amplitude * values

    def get_values_for_points(self, points, cov=None):
        points_proj = self._project(points)
        conditions = [
            np.logical_and(points_proj[:, i] >= start, points_proj[:, i] <= stop)
            for i, (start, stop) in enumerate(self.bounds)
        ]
        inside = np.logical_and.reduce(conditions)
        values = np.zeros(points.shape[0])
        if np.any(inside):
            filtered = points_proj[inside]
            if not self.scipy:
                d2 = np.sum(
                    ((filtered - self.mean) ** 2) / (2 * self.std_dev**2), axis=1
                )
                values[inside] = self.max_amplitude * np.exp(-d2)
            else:
                if cov is None:
                    cov = np.diag(self.std_dev**2)
                values[inside] = self.max_amplitude * multivariate_normal.pdf(
                    filtered, self.mean, cov
                )
        return values


class Ramp:
    """
    Linear ramp criticality function clipped to [min_value, max_value].
    Coefficients are automatically rescaled so the linear output spans exactly
    [min_value, max_value] over the declared bounds.
    """

    def __init__(
        self, coeff: list, min_value, max_value, bounds: list, active_dims=None
    ):
        """
        Parameters
        ----------
        coeff : list of float
            Direction vector of the ramp (one value per active dimension).
            Will be normalised so the output range matches [min_value, max_value].
        min_value, max_value : float
            Output clipping bounds; also define the rescaling target.
        bounds : list of (float, float)
            Axis-aligned bounding box; points outside return 0.
        active_dims : list of int, optional
            Indices of input dimensions used; defaults to all.
        """
        self.coeff, self.intercept = self.scale_coefficients(
            coeff, min_value, max_value, bounds
        )
        self.min_value = min_value
        self.max_value = max_value
        self.bounds = bounds
        self.active_dims = (
            active_dims if active_dims is not None else list(range(len(coeff)))
        )

    def _project(self, mesh_or_points):
        return (
            [mesh_or_points[i] for i in self.active_dims]
            if isinstance(mesh_or_points, (list, tuple))
            else mesh_or_points[:, self.active_dims]
        )

    def _linear_function(self, points):
        return np.dot(points, self.coeff) + self.intercept

    @staticmethod
    def _generate_boundary_points_static(bounds):
        from itertools import product

        return np.array(list(product(*bounds)))

    @staticmethod
    def scale_coefficients(coeff, min_value, max_value, bounds):
        coeff = np.array(coeff, dtype=float)
        points = Ramp._generate_boundary_points_static(bounds)
        values = np.dot(points, coeff)
        min_val, max_val = np.min(values), np.max(values)
        scaled_coeff = coeff * (max_value - min_value) / (max_val - min_val)
        intercept = min_value - np.min(np.dot(points, scaled_coeff))
        return scaled_coeff, intercept

    def calculate_relevant_values(self, meshgrid):
        mesh_proj = self._project(meshgrid)
        conditions = [
            np.logical_and(g >= lo, g <= hi)
            for g, (lo, hi) in zip(mesh_proj, self.bounds)
        ]
        mask = np.logical_and.reduce(conditions)
        points = np.vstack([g[mask] for g in mesh_proj]).T
        values = np.clip(self._linear_function(points), self.min_value, self.max_value)
        return mask, values

    def get_values_for_points(self, points):
        proj_points = self._project(points)
        bounds_check = np.all(
            [
                np.logical_and(proj_points[:, i] >= lo, proj_points[:, i] <= hi)
                for i, (lo, hi) in enumerate(self.bounds)
            ],
            axis=0,
        )
        values = np.zeros(points.shape[0])
        if np.any(bounds_check):
            v = self._linear_function(proj_points[bounds_check])
            values[bounds_check] = np.clip(v, self.min_value, self.max_value)
        return values


class Noise:
    """
    Additive Gaussian noise term applied uniformly over the entire space.
    Typically combined with other function primitives to simulate measurement
    uncertainty or model imprecision in the criticality surface.
    """

    def __init__(self, amplitude, std_dev=1.0, seed=42):
        """
        Parameters
        ----------
        amplitude : float
            Scales the noise output (peak-to-peak magnitude ≈ 2·amplitude·std_dev).
        std_dev : float
            Standard deviation of the zero-mean Gaussian noise.
        seed : int or None
            RNG seed for reproducibility; pass None for non-deterministic noise.
        """
        self.amplitude = amplitude
        self.std_dev = std_dev
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def calculate_relevant_values(self, meshgrid):
        """Returns a full-space mask (all True) and iid noise for every grid point."""
        shape = meshgrid[0].shape
        noise = self.rng.normal(0, self.std_dev, size=shape) * self.amplitude
        return np.ones(shape, dtype=bool), noise.ravel()

    def get_values_for_points(self, points):
        """Returns iid noise of shape (N,) for N input points."""
        noise = self.rng.normal(0, self.std_dev, size=points.shape[0]) * self.amplitude
        return noise


class Sinus:
    """
    Represents a multi-dimensional sinusoidal function that is not strictly aligned
    with the coordinate axes. Each wave depends on a dot product of the input
    coordinates with a given weight vector, ensuring a 'tilt' in the wave pattern.

    Mathematically, for a point x in R^d:

        f(x) = offset
               + sum_i [ amplitude[i] * sin( frequency[i] * (w_i . x) + phase[i]) ]

    where w_i . x is the dot product of w_i with x.
    Optionally, bounds can be used to restrict computation to a subset of the space.
    Outside those bounds, values default to 0.
    """

    def __init__(
        self,
        amplitudes,
        frequencies,
        weight_vectors,
        phases=None,
        offset=0.0,
        bounds=None,
        minimum_value=None,
        active_dims=None,
    ):
        """
        Parameters
        ----------
        amplitudes : list or ndarray of float
            Amplitudes (A_i) for each wave (number_of_waves-dimensional).

        frequencies : list or ndarray of float
            Frequencies (f_i) for each wave (number_of_waves-dimensional).

        weight_vectors : list or ndarray
            A list/array of shape (number_of_waves, d),
            where each row is a weight vector w_i in R^d.
            This defines how each wave depends on the input coordinates.

        phases : list or ndarray of float, optional
            Phases (phi_i) for each wave (number_of_waves-dimensional).
            If None, all phases default to 0.

        offset : float, optional
            A constant value added to the sum of sinusoidal components.

        bounds : list of tuples, optional
            Defines a bounding box for evaluation. If provided, must be a list
            of d tuples in the form [(min_1, max_1), ..., (min_d, max_d)].
            Points outside these ranges return 0.
        """
        self.amplitudes = np.array(amplitudes, dtype=float)
        self.frequencies = np.array(frequencies, dtype=float)
        self.weight_vectors = np.array(weight_vectors, dtype=float)
        self.active_dims = (
            active_dims
            if active_dims is not None
            else list(range(len(weight_vectors[0])))
        )

        # Basic checks to ensure consistent array lengths
        n_waves = len(self.amplitudes)
        if any(len(arr) != n_waves for arr in [self.frequencies]):
            raise ValueError("amplitudes and frequencies must have the same length.")
        if self.weight_vectors.shape[0] != n_waves:
            raise ValueError("Number of weight vectors must match number of waves.")

        # If phases are not specified, use zeros
        if phases is None:
            self.phases = np.zeros(n_waves, dtype=float)
        else:
            if len(phases) != n_waves:
                raise ValueError(
                    "phases must have the same length as amplitudes/frequencies."
                )
            self.phases = np.array(phases, dtype=float)

        self.offset = offset
        self.bounds = bounds
        self.minimum_value = minimum_value

    def _project(self, points):
        return points[:, self.active_dims]

    def _compute_sinus_values(self, points):
        """
        Compute the sinusoidal function values for given points in R^d.

        Parameters
        ----------
        points : ndarray of shape (N, d)
            N points in d-dimensional space.

        Returns
        -------
        values : ndarray of shape (N,)
            The evaluated sinusoidal function at each point.
        """
        # Ensure we have a 2D array of points
        if points.ndim == 1:
            points = points[np.newaxis, :]

        points = self._project(points)
        # We'll accumulate results from each wave
        # For wave i:
        #   wave_i = amplitude[i] * sin( frequency[i] * (weight_vectors[i] . x) + phase[i] )
        # Then sum across waves and add offset.
        #
        #   shape of points = (N, d)
        #   shape of weight_vectors[i] = (d,)

        # Dot product for all waves at once:
        #   (N, d) dot (d, number_of_waves) = (N, number_of_waves)
        # We'll transpose weight_vectors to shape (d, number_of_waves) for this operation.
        dot_products = points.dot(self.weight_vectors.T)  # (N, number_of_waves)

        # Multiply each column by its frequency, add phase, then take sin
        # shape => (N, number_of_waves)
        sinus_args = self.frequencies * dot_products + self.phases
        sinus_values = np.sin(sinus_args)

        # Multiply by amplitudes (broadcast: shape (1, number_of_waves))
        sinus_values *= self.amplitudes

        # Sum over waves => shape (N,)
        total_values = np.sum(sinus_values, axis=1)

        # Add offset
        total_values += self.offset

        # Apply minimum value constraint if specified
        if self.minimum_value is not None:
            total_values = np.maximum(total_values, self.minimum_value)

        return total_values

    def calculate_relevant_values(self, meshgrid):
        """
        Evaluate the sinusoidal function on a meshgrid, applying bounds if present.

        Parameters
        ----------
        meshgrid : list of ndarrays
            The coordinate arrays for each dimension.
            E.g., for d=2, meshgrid = [X, Y] from np.meshgrid.

        Returns
        -------
        subgrid_mask : ndarray (bool)
            A boolean mask indicating which points are inside the bounds.
        values : ndarray (float)
            The sinusoidal values for those points (flattened).
        """
        # If no bounds specified, all points are valid
        if not self.bounds:
            subgrid_mask = np.ones_like(meshgrid[0], dtype=bool)
        else:
            conditions = [
                np.logical_and(coord >= low, coord <= high)
                for coord, (low, high) in zip(meshgrid, self.bounds)
            ]
            subgrid_mask = np.logical_and.reduce(conditions)

        # Extract the relevant points that are inside the bounding region
        relevant_points = np.vstack([coord[subgrid_mask] for coord in meshgrid]).T
        # Evaluate the sinus function
        relevant_values = self._compute_sinus_values(relevant_points)

        return subgrid_mask, relevant_values

    def get_values_for_points(self, points):
        """
        Evaluate the sinusoidal function at arbitrary points, returning 0 for out-of-bounds points.

        Parameters
        ----------
        points : ndarray of shape (N, d)
            The input points where the function should be evaluated.

        Returns
        -------
        values : ndarray of shape (N,)
            The function values at each input point, or 0 if out of bounds.
        """
        # If no bounds specified, compute for all points
        if not self.bounds:
            return self._compute_sinus_values(points)

        # Identify points within all bounds
        conditions = []
        for i, (low, high) in enumerate(self.bounds):
            in_range = np.logical_and(points[:, i] >= low, points[:, i] <= high)
            conditions.append(in_range)

        inside_mask = np.logical_and.reduce(conditions)

        # Prepare output array
        values = np.zeros(points.shape[0], dtype=float)

        # For points inside bounds, compute normally
        if np.any(inside_mask):
            values[inside_mask] = self._compute_sinus_values(points[inside_mask])

        return values

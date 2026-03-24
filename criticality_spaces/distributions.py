"""
Project: Toward Standardized Benchmarking of Search-Based Scenario Selection Methods in Autonomous System Validation
Version: 1.0.0

Description:
    Wraps scipy.stats distributions to model per-axis input distributions of a
    criticality space. Supports any continuous distribution available in
    scipy.stats; parameters are supplied via JSON and validated at construction.

    NOTE: This module implements a planned feature for future work and is not
    actively used in the benchmark paper. Only a single example distribution
    spec is provided (Distributions/dist3d.json).

    - key role: Marginal and joint density evaluation for importance-weighted
                scenario sampling.
    - dependency: scipy.stats
    - output: Per-axis densities and joint density at queried points.

Usage:
    from distributions import DistributionHandler
    handler = DistributionHandler(distribution_specs)   # specs loaded from JSON
    joint, axes = handler.get_joint_density((x0, x1))
"""

from scipy import stats
from typing import List, Dict, Union, Tuple

class DistributionHandler:
    def __init__(self, distribution_specs: List[Dict]):
        """
        Parameters
        ----------
        distribution_specs : list of dict
            Each dict must contain a ``"type"`` key matching a scipy.stats
            distribution name (e.g. ``"norm"``, ``"beta"``), plus any required
            shape parameters and optional ``loc`` / ``scale``.
            Note: ``"type"`` is consumed from each dict in-place (pop).
        """
        self.distributions = []

        for spec in distribution_specs:
            dist_type = spec.pop("type").lower()

            if not hasattr(stats, dist_type):
                raise NotImplementedError(f"Distribution type '{dist_type}' is not available in scipy.stats.")

            dist_class = getattr(stats, dist_type)

            # Validate and filter parameters
            valid_params, missing = self._filter_valid_params(dist_class, spec)

            if missing:
                raise TypeError(f"Missing required shape parameters for '{dist_type}': {missing}")

            self.distributions.append(dist_class(**valid_params))

    def _filter_valid_params(self, dist_class, params: Dict) -> Tuple[Dict, List[str]]:
        """
        Filters valid parameters for a scipy.stats distribution and checks for required shape parameters.

        Returns:
            valid_params (Dict): Parameters that are valid for the distribution.
            missing (List[str]): List of required parameters that are missing.
        """
        # Extract required shape parameters using distribution shapes
        required_shape_params = getattr(dist_class, "shapes", "")
        shape_param_names = [s.strip() for s in required_shape_params.split(",")] if required_shape_params else []

        # Include standard parameters
        accepted_params = set(shape_param_names + ["loc", "scale"])

        valid = {}
        missing = []

        for name in shape_param_names:
            if name not in params:
                missing.append(name)

        for k, v in params.items():
            if k in accepted_params:
                valid[k] = v
            else:
                print(f"[WARNING] Invalid parameter for '{type(dist_class).__name__}': {k} - ignored.")

        return valid, missing

    def get_axis_densities(self, point: Union[List[float], Tuple[float]]) -> List[float]:
        """
        Returns the probability densities for each coordinate along the individual axes.

        These values represent the marginal density f_i(x_i) at each coordinate x_i
        according to the axis-specific distributions. They are not probabilities,
        but densities in the sense of continuous probability theory.
        
        Returns:
            List[float]: List of probability densities per axis at the given point.
        """
        return [dist.pdf(coord) for dist, coord in zip(self.distributions, point)]

    def get_joint_density(self, point: Union[List[float], Tuple[float]]) -> float:
        """
        Returns the joint probability density at a point assuming axis independence.

        The joint density is computed as the product of marginal densities:
            f(x) = ∏ f_i(x_i)
        This is not a probability, but a density in R^d (units: 1/unit^d),
        and can be used to approximate probability by integrating or multiplying
        with a small volume element.

        Returns:
            Tuple[float, List[float]]: 
                - joint_density: the product of marginal densities at the point
                - axis_densities: list of individual axis densities
        """
        axis_probs = self.get_axis_densities(point)
        joint = 1.0
        for p in axis_probs:
            joint *= p
        return joint, axis_probs

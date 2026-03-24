"""
Project: Toward Standardized Benchmarking of Search-Based Scenario Selection Methods in Autonomous System Validation
Version: 1.0.0

Description:
    Implements the Nearest-Neighbour Distance-Variance (NNDV) selector and its
    smoothed variant (S-NDV). Starting from a Latin Hypercube seed, each
    iteration draws a large uniform candidate pool and scores every candidate
    by an information metric I = Var_n^g · D_n^v, where Var_n is the label
    variance among the k nearest selected neighbours and D_n is their mean
    distance. The top-``n_batch`` candidates are queried and added to the
    selected set. In S-NDV mode, kernel-weighted statistics over all selected
    points replace the k-NN estimates.

    - key role: Boundary-focused adaptive sampler that concentrates queries
                near decision-boundary transitions for boundary-detection metrics.
    - dependency: numpy, scipy (qmc, KDTree), tqdm, base.BaseSelector
    - output: List of selected scenario-point tuples stored in space.selected_points.

Usage:
    from selector_nndv import NNDVSelector
    selector = NNDVSelector(space, n_select=100)
    selector.select()
"""

import numpy as np
from base import BaseSelector
from scipy.spatial import KDTree
from scipy.stats import qmc
from tqdm import tqdm


class NNDVSelector(BaseSelector):
    """
    Nearest-Neighbour Distance-Variance (NNDV) adaptive sampling selector.

    Scores candidate points by a combined variance–distance information metric
    and iteratively queries the most informative candidates to concentrate
    samples near criticality decision boundaries. Supports a smoothed (S-NDV)
    variant that replaces k-NN statistics with Gaussian kernel-weighted
    estimates over all selected points.
    """

    def __init__(
        self,
        space,
        n_select: int,
        n_initial: int = 20,
        k: int = 10,
        candidate_pool_size: int = 5000,
        g: float = 1.0,
        v: float = 1.0,
        n_batch: int = 1,
        smoothed: bool = False,
        sigma: float = 0.1,
        vis: bool = False,
    ):
        """
        Parameters
        ----------
        space : Space
            The criticality space to query.
        n_select : int
            Total number of scenario points to select.
        n_initial : int, optional
            Number of initial LHS seed samples (default 20).
        k : int, optional
            Number of nearest neighbours used to compute variance and distance
            scores in NNDV mode (default 10).
        candidate_pool_size : int, optional
            Number of uniform random candidates drawn each iteration (default 5000).
        g : float, optional
            Exponent applied to the variance score Var_n in the information
            metric I (default 1.0).
        v : float, optional
            Exponent applied to the distance score D_n in the information
            metric I (default 1.0).
        n_batch : int, optional
            Number of top-scoring candidates queried per iteration (default 1).
        smoothed : bool, optional
            If True, use the S-NDV kernel-weighted variant instead of k-NN
            (slower but smoother estimates; default False).
        sigma : float, optional
            Kernel bandwidth for S-NDV mode (default 0.1).
        vis : bool, optional
            If True, render a 2-D decision-boundary plot after selection
            (default False).
        """
        super().__init__(space, n_select)
        self.n_initial = n_initial
        self.k = k
        self.candidate_pool_size = candidate_pool_size
        self.g = g  # variance weight
        self.v = v  # distance/exploration weight
        self.n_batch = n_batch
        self.smoothed = smoothed
        self.sigma = sigma
        self.vis = vis

    def select(self) -> list[tuple]:
        """
        Run the NNDV adaptive sampling loop and store results in the space.

        Seeds with an LHS sample of discrete labels, then iteratively draws a
        uniform candidate pool, scores each candidate by
        I = Var_n^g · D_n^v (NNDV) or a kernel-weighted analogue (S-NDV),
        and queries the top-``n_batch`` candidates until the budget is reached.

        Returns
        -------
        list of tuple
            Up to ``n_select`` selected scenario parameter points.
        """
        dim = len(self.space.dimensions)
        bounds = np.array(self.space.dimensions)

        # --- Initial LHS seed sample ---
        sampler = qmc.LatinHypercube(d=dim)
        initial_points = sampler.random(self.n_initial)
        initial_points = qmc.scale(initial_points, bounds[:, 0], bounds[:, 1])
        initial_points = [
            tuple(np.round(p, self.space.decimal_precision)) for p in initial_points
        ]
        initial_values = self.space.get_discrete_values_for_points(
            initial_points,
            thresholds=self.space.criticality_thresholds,
            save_points=True,
        )
        selected_points = list(initial_points)
        selected_values = list(initial_values)
        selected_set = set(selected_points)

        n_to_select = self.n_select - len(selected_points)

        # --- Adaptive NNDV sampling with tqdm ---
        with tqdm(total=n_to_select, desc="NNDV Adaptive Selection") as pbar:
            while n_to_select > 0:
                # a. Generate candidate points uniformly in the space
                candidates = np.random.uniform(
                    low=bounds[:, 0],
                    high=bounds[:, 1],
                    size=(self.candidate_pool_size, dim),
                )
                # Round for consistency
                candidates_rounded = np.round(candidates, self.space.decimal_precision)
                candidate_tuples = [tuple(row) for row in candidates_rounded]

                # Set-based deduplication
                candidate_tuples_set = set(candidate_tuples)
                new_candidates = list(candidate_tuples_set - selected_set)
                if not new_candidates:
                    break
                candidates = np.array(new_candidates)

                # b. Build KDTree from selected points
                sel_pts_array = np.array(selected_points)
                sel_vals = np.array(selected_values)

                # --- NNDV or S-NDV ---
                if self.smoothed:
                    # ---- S-NDV: kernel-weighted over ALL selected points (takes way longer) ----
                    from scipy.spatial.distance import cdist

                    all_dists = cdist(
                        candidates, sel_pts_array
                    )  # shape (n_candidates, n_selected)
                    w = np.exp(-(all_dists**2) / (2 * self.sigma**2)) + 1e-12
                    cutoff = (
                        3 * self.sigma
                    )  # Optimization - not mentioned in the original paper
                    w[all_dists > cutoff] = 0
                    w_sum = np.sum(w, axis=1)
                    w_sum[w_sum == 0] = 1e-12
                    w_mean = np.sum(w * sel_vals[None, :], axis=1) / w_sum
                    w_var = (
                        np.sum(w * (sel_vals[None, :] - w_mean[:, None]) ** 2, axis=1)
                        / w_sum
                    )
                    w_density = 1.0 / w_sum
                    local_variance = w_var
                    local_density = w_density
                else:
                    # ---- NNDV: k-nearest neighbors ----
                    tree = KDTree(sel_pts_array)
                    k_eff = min(self.k, len(selected_points))
                    dists, idxs = tree.query(candidates, k=k_eff)
                    if dists.ndim == 1:
                        dists = dists[:, None]
                        idxs = idxs[:, None]
                    neighbor_values = sel_vals[idxs]
                    local_variance = np.var(neighbor_values, axis=1)
                    local_density = np.mean(dists, axis=1)
                info_metric = (local_variance**self.g) * (local_density**self.v)

                # Select top points (batch selection)
                n_batch = min(n_to_select, self.n_batch)
                top_idx = np.argsort(info_metric)[-n_batch:]
                new_points = [new_candidates[i] for i in top_idx]
                new_values = self.space.get_discrete_values_for_points(
                    new_points,
                    thresholds=self.space.criticality_thresholds,
                    save_points=True,
                )
                selected_points.extend(new_points)
                selected_values.extend(new_values)
                selected_set.update(new_points)
                n_added = len(new_points)
                n_to_select -= n_added
                pbar.update(n_added)

        # Optional: visualize results
        if self.vis:
            # self.space.visualizer.plot_3d_two_varied_discrete(dim1=0, dim2=1)
            self.space.visualizer.plot_2d_decision_boundary()

        return selected_points[: self.n_select]

"""
Project: Toward Standardized Benchmarking of Search-Based Scenario Selection Methods in Autonomous System Validation
Version: 1.0.0

Description:
    Implements the Gradient- and Distance-based Nearest-Neighbour Adaptive
    Sampling (GDNNAS) selector. Starting from a Latin Hypercube initial set,
    each iteration evaluates a combined gradient–distance score (M_n) for all
    current samples, selects an elite subset, generates new LHS candidates
    around each elite point within a shrinking neighbourhood, and then
    eliminates samples that are far from the decision boundary.

    - key role: Boundary-focused adaptive sampler that concentrates queries
                near criticality transitions for boundary-detection metrics.
    - dependency: numpy, scipy (qmc, KDTree), matplotlib, base.BaseSelector
    - output: List of selected scenario-point tuples stored in space.selected_points.

Usage:
    from selector_gdnnas import GDNNASSelector
    selector = GDNNASSelector(space, n_select=100)
    selector.select()
"""

import numpy as np
from base import BaseSelector
from scipy.stats import qmc
from scipy.spatial import KDTree
import matplotlib.pyplot as plt


class GDNNASSelector(BaseSelector):
    """
    Gradient- and Distance-based Nearest-Neighbour Adaptive Sampling selector.

    Iteratively focuses samples on regions with high label gradients and large
    inter-sample distances, then prunes points far from the decision boundary.
    The combined score M_n = G_n^g * D_n^v drives elite selection and
    neighbourhood generation at each iteration.
    """

    def __init__(
        self,
        space,
        n_select: int,
        n_initial: int = 100,
        k: int = 8,
        g: float = 2.0,
        v: float = 1.0,
        p0: float = 0.2,
        a: float = 240,
        b: float = 72,
        omega0: float = 5.0,
        omega_min: float = 1.5,
        vis=False,
    ):
        """
        Parameters
        ----------
        space : Space
            The criticality space to query.
        n_select : int
            Total number of scenario points to select.
        n_initial : int, optional
            Number of initial LHS samples (default 100).
        k : int, optional
            Number of nearest neighbours used when computing the gradient
            and distance scores (default 8).
        g : float, optional
            Exponent applied to the gradient score G_n in M_n (default 2.0).
        v : float, optional
            Exponent applied to the distance score D_n in M_n (default 1.0).
        p0 : float, optional
            Initial elite fraction of the current sample set (default 0.2).
        a : float, optional
            Slope parameter of the neighbourhood sampling-density formula
            (default 240).
        b : float, optional
            Intercept parameter of the neighbourhood sampling-density formula
            (default 72).
        omega0 : float, optional
            Initial elimination multiplier for the boundary-distance filter
            (default 5.0).
        omega_min : float, optional
            Minimum elimination multiplier (default 1.5).
        vis : bool, optional
            If True, render a 2-D decision-boundary plot after selection (default False).
        """
        super().__init__(space, n_select)
        # Algorithm parameters
        self.n_initial = n_initial  # initial LHS samples
        self.k = k  # nearest neighbours for evaluation
        self.g = g  # gradient weight
        self.v = v  # distance weight
        self.p0 = p0  # initial elite fraction
        self.a = a  # sampling density parameter
        self.b = b  # sampling density parameter
        self.omega0 = omega0  # initial elimination multiplier
        self.omega_min = omega_min  # minimum elimination multiplier
        self.vis = vis
        self._n_queries = 0

    def plot_iteration_summary(
        self, iteration: int, new_points: list, eliminated_points: list
    ):
        """
        Plot newly generated points for one iteration (2-D spaces only).

        Kept points are shown in green, eliminated points in red.

        Parameters
        ----------
        iteration : int
            Current iteration index (used in the plot title).
        new_points : list of tuple
            Points generated around elite samples this iteration.
        eliminated_points : list of tuple
            Points that were removed by the boundary-distance filter.
        """
        new_points_arr = np.array(new_points)
        eliminated_arr = np.array(eliminated_points)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)

        if new_points_arr.size > 0:
            ax.scatter(
                new_points_arr[:, 0],
                new_points_arr[:, 1],
                c="green",
                label="Kept",
                alpha=0.6,
                marker="o",
            )
        if eliminated_arr.size > 0:
            ax.scatter(
                eliminated_arr[:, 0],
                eliminated_arr[:, 1],
                c="red",
                label="Eliminated",
                alpha=0.6,
                marker="x",
            )

        ax.set_title(f"Iteration {iteration} – New Sampled Points")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    def select(self) -> list[tuple]:
        """
        Run the GDNNAS adaptive sampling loop and store results in the space.

        Steps per iteration:
          1. Evaluate M_n = G_n^g * D_n^v for all current samples.
          2. Select the top-p_t fraction as elites.
          3. Generate LHS candidates within a shrinking hypercube around each elite.
          4. Eliminate samples whose distance to the nearest differently-labelled
             point exceeds omega_t * D_mean.

        Returns
        -------
        list of tuple
            Up to ``n_select`` selected scenario parameter points.
        """
        dim = len(self.space.dimensions)
        # extract bounds
        lowers = np.array([d[0] for d in self.space.dimensions])
        uppers = np.array([d[1] for d in self.space.dimensions])
        ranges = uppers - lowers
        # scalar range for hypercube edge length
        R_scalar = float(np.mean(ranges))
        # initial hypercube edge length
        r0 = R_scalar / 10.0

        # --- 1) Initial Latin Hypercube Sampling ---
        sampler = qmc.LatinHypercube(d=dim)
        init_unit = sampler.random(self.n_initial)
        init_points = qmc.scale(init_unit, lowers, uppers)
        # round and tuple-ify
        init_points = [
            tuple(np.round(p, self.space.decimal_precision)) for p in init_points
        ]
        # query discrete labels
        init_labels = self.space.get_discrete_values_for_points(
            init_points, thresholds=self.space.criticality_thresholds, save_points=False
        )
        self._n_queries += len(init_points)

        selected_points = list(init_points)
        selected_labels = list(init_labels)
        selected_set = set(selected_points)

        t = 1
        # main adaptive loop
        while len(selected_points) < self.n_select:
            # --- 2) Evaluation: compute M_n for each sample ---
            pts_array = np.array(selected_points)
            labels_arr = np.array(selected_labels)
            tree = KDTree(pts_array)
            # query k+1 nearest (including self)
            k_eff = min(self.k + 1, len(selected_points))
            dists, idxs = tree.query(pts_array, k=k_eff)
            # drop self-distance at index 0
            if dists.ndim == 1:
                dists = dists[np.newaxis, :]
                idxs = idxs[np.newaxis, :]
            dists = dists[:, 1:]
            idxs = idxs[:, 1:]
            # compute gradient and distance criteria
            # delta labels
            neighbor_labels = labels_arr[idxs]
            delta = np.abs(neighbor_labels - labels_arr[:, None])
            # avoid division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                grads = np.abs(np.arctan(delta / dists))
            G_n = np.nan_to_num(grads.mean(axis=1))
            D_n = np.nan_to_num(dists.mean(axis=1))
            M_n = (G_n**self.g) * (D_n**self.v)

            # --- 3) Select elite samples ---
            p_t = max(self.p0, 0.01)
            elite_count = max(int(p_t * len(selected_points)), 1)
            elite_indices = np.argsort(M_n)[-elite_count:]

            # --- 4) Generate new samples around elites ---
            new_points = []
            # hypercube edge length this iteration
            r_t = r0 / np.sqrt(t)
            # relative area ratio
            rel_area = r_t / R_scalar
            rho = -self.a * rel_area + self.b
            n_t = max(int(abs(rel_area * rho + 0.5)), 1)

            for idx in elite_indices:
                center = np.array(selected_points[idx])
                # define hypercube bounds
                low = np.maximum(lowers, center - 0.5 * r_t)
                high = np.minimum(uppers, center + 0.5 * r_t)
                # LHS within neighborhood
                sampler_local = qmc.LatinHypercube(d=dim)
                local_unit = sampler_local.random(n_t)
                local_pts = low + (high - low) * local_unit
                # round and tuple
                local_pts = [
                    tuple(np.round(p, self.space.decimal_precision)) for p in local_pts
                ]
                for p in local_pts:
                    if p not in selected_set:
                        new_points.append(p)

            # break if no new points
            if not new_points:
                break

            # query labels for new points
            new_labels = self.space.get_discrete_values_for_points(
                new_points,
                thresholds=self.space.criticality_thresholds,
                save_points=False,
            )
            self._n_queries += len(new_points)
            # add to selected
            for p, lbl in zip(new_points, new_labels):
                selected_points.append(p)
                selected_labels.append(int(lbl))
                selected_set.add(p)

            # --- 5) Elimination: remove points far from boundary ---
            pts_array = np.array(selected_points)
            labels_arr = np.array(selected_labels)
            # compute D_mode for each point
            D_mode_list = []
            for i, pt in enumerate(pts_array):
                # distances to all points
                d_all = np.linalg.norm(pts_array - pt, axis=1)
                # mask different label
                mask = labels_arr != labels_arr[i]
                if np.any(mask):
                    D_mode_list.append(d_all[mask].min())
                else:
                    D_mode_list.append(np.inf)
            D_mode_arr = np.array(D_mode_list)
            # mean finite distance
            finite = np.isfinite(D_mode_arr)
            D_mean = D_mode_arr[finite].mean() if finite.any() else 0.0
            omega_t = (self.omega0 / np.sqrt(t)) + self.omega_min
            # filter
            keep = D_mode_arr <= omega_t * D_mean
            filtered = [
                (p, lbl)
                for p, lbl, kf in zip(selected_points, selected_labels, keep)
                if kf
            ]
            if filtered:
                selected_points, selected_labels = zip(*filtered)
                selected_points = list(selected_points)
                selected_labels = list(selected_labels)
                selected_set = set(selected_points)

            if self.vis:
                new_pts_array = np.array(new_points)
                kept_set = set(selected_points)
                eliminated = [p for p in new_points if p not in kept_set]
                self.plot_iteration_summary(
                    iteration=t, new_points=new_pts_array, eliminated_points=eliminated
                )

            t += 1

        selected_points = selected_points[: self.n_select]
        self.space.reset_selected_points()
        a = self.space.get_values_for_points(
            selected_points
        )  # Selected points are written in Space-Object

        self.space.n_query_calls = int(self._n_queries)

        # Optional: visualize results
        if self.vis:
            # self.space.visualizer.plot_3d_two_varied_discrete(dim1=0, dim2=1)
            self.space.visualizer.plot_2d_decision_boundary()

        # return up to n_select points
        return selected_points[: self.n_select]

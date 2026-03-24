"""
Project: Toward Standardized Benchmarking of Search-Based Scenario Selection Methods in Autonomous System Validation
Version: 1.0.0

Description:
    Implements the Inverse Distance Weighting (IDW) surrogate-model selector.
    An initial Latin Hypercube sample is evaluated, then each iteration draws a
    fresh random candidate pool, predicts with the IDW regressor, and selects
    the top-k candidates by an acquisition score AC = (ŷ - ŷ_min)^β · D_min
    that balances exploitation of high predicted values with exploration of
    under-sampled regions.

    - key role: Fits a non-parametric IDW surrogate of the criticality surface
                for model-reconstruction metric evaluation.
    - dependency: numpy, scipy (qmc, spatial), matplotlib, base.BaseModelSelector
    - output: Trained IDWRegressor instance compatible with
              space.metrics.run_metrics_suite(surrogate_model=...).

Usage:
    from selector_idw import IDWSelector
    selector = IDWSelector(space, n_select=100)
    model = selector.get_model()
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, List
from scipy.spatial import distance, cKDTree
from scipy.stats import qmc
import logging
import matplotlib.pyplot as plt

from base import BaseModelSelector

logger = logging.getLogger(__name__)


# ==============================
# Surrogate model: IDW Regressor
# ==============================
class IDWRegressor:
    """
    Inverse Distance Weighting (IDW) regressor with a scikit-learn-like API.

    - Non-parametric: 'fit' only stores support points.
    - Predicts via weighted average with weights w_i ∝ (||x - x_i|| + eps)^(-power).
    - Recommended: k-NN restriction via KD-Tree for scalability/robustness.
    """

    def __init__(
        self,
        power: float = 2,
        n_neighbors: Optional[int] = None,
        epsilon: float = 1e-12,
        leafsize: int = 16,
    ) -> None:
        """
        Parameters
        ----------
        power : float, optional
            Distance exponent p; weights ∝ distance^(-p) (default 2).
            When called via IDWSelector the selector's ``idw_power`` is used.
        n_neighbors : int or None, optional
            If set, restrict IDW to the k nearest neighbours via KD-Tree.
            ``None`` uses all support points (default None).
        epsilon : float, optional
            Distance floor added before inversion for numerical stability
            (default 1e-12).
        leafsize : int, optional
            Leaf size of the internal KD-Tree (default 16).
        """
        self.power = float(power)
        self.n_neighbors = n_neighbors
        self.epsilon = float(epsilon)
        self.leafsize = int(leafsize)
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._tree: Optional[cKDTree] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "IDWRegressor":
        """
        Store support points and (optionally) build the KD-Tree index.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, d)
            Support point coordinates.
        y : np.ndarray, shape (n_samples,)
            Target values at each support point.

        Returns
        -------
        IDWRegressor
            Self (for method chaining).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if not (X.ndim == 2 and X.shape[0] == y.shape[0]):
            raise ValueError("X must be (n_samples, d) and y must be (n_samples,).")
        self._X = X
        self._y = y
        self._tree = cKDTree(self._X, leafsize=self.leafsize) if self.n_neighbors is not None else None
        return self

    def _predict_all(self, Xq: np.ndarray) -> np.ndarray:
        """
        IDW prediction using all support points — O(n·m).

        Exact duplicates of support points are assigned their stored value
        directly to avoid division by zero.
        """
        D = distance.cdist(Xq, self._X, metric="euclidean")
        zero_mask = D <= self.epsilon
        preds = np.empty(Xq.shape[0], dtype=float)

        # Direct copy for query rows that coincide with a support point
        rows_zero = np.where(zero_mask.any(axis=1))[0]
        for r in rows_zero:
            j = np.where(zero_mask[r])[0][0]
            preds[r] = self._y[j]

        # Standard IDW for the remaining rows
        rows_rest = np.where(~zero_mask.any(axis=1))[0]
        if len(rows_rest) > 0:
            Dr = D[rows_rest]
            W = 1.0 / np.maximum(Dr, self.epsilon) ** self.power
            preds[rows_rest] = (W @ self._y) / W.sum(axis=1)
        return preds

    def _predict_knn(self, Xq: np.ndarray) -> np.ndarray:
        """
        IDW prediction restricted to the k nearest neighbours via KD-Tree.

        Scales better than the full O(n·m) variant and is generally more
        robust against distant outlier support points.
        """
        assert self._tree is not None and self.n_neighbors is not None
        dists, idx = self._tree.query(Xq, k=self.n_neighbors, workers=-1)
        if self.n_neighbors == 1:
            dists = dists[:, None]
            idx = idx[:, None]

        near_zero = dists <= self.epsilon
        preds = np.empty(Xq.shape[0], dtype=float)

        rows_zero = np.where(near_zero.any(axis=1))[0]
        for r in rows_zero:
            j = idx[r, np.where(near_zero[r])[0][0]]
            preds[r] = self._y[j]

        rows_rest = np.where(~near_zero.any(axis=1))[0]
        if len(rows_rest) > 0:
            d = dists[rows_rest]
            idc = idx[rows_rest]
            w = 1.0 / np.maximum(d, self.epsilon) ** self.power
            y_nb = self._y[idc]
            preds[rows_rest] = (w * y_nb).sum(axis=1) / w.sum(axis=1)

        return preds

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict criticality values for query points.

        Parameters
        ----------
        X : np.ndarray, shape (n, d) or (d,)
            Query point(s).

        Returns
        -------
        np.ndarray, shape (n,)
            Predicted values.
        """
        if self._X is None or self._y is None:
            raise RuntimeError("Call fit() before predict().")
        Xq = np.asarray(X, dtype=float)
        if Xq.ndim == 1:
            Xq = Xq[None, :]
        if self.n_neighbors is None:
            return self._predict_all(Xq)
        return self._predict_knn(Xq)


# ===========================================
# Selector (maximization-focused)
# ===========================================
class IDWSelector(BaseModelSelector):
    """
    Surrogate-model selector that fits an IDW regressor to the criticality space.

    Selection loop
    --------------
    1. Draw ``n_initial`` LHS samples and evaluate them (SUT).
    2. Iterate until the budget is reached:
       a. Draw ``n_candidates`` uniform random candidates.
       b. Predict with the current IDW model.
       c. Compute ``D_min`` — minimum distance from each candidate to the
          tested set.
       d. Rank by AC = (ŷ - ŷ_min)^β · D_min (maximise).
       e. Query the top ``batch_size`` candidates and add them to the tested set.
    """

    def __init__(
        self,
        space,
        n_select: int,
        n_initial: int = None,
        n_candidates: int = 250,
        batch_size: int = 5,
        beta: float = 1.0,
        idw_power: float = 5.0,
        idw_n_neighbors: Optional[int] = 5,
        epsilon: float = 1e-12,
        rng_seed: Optional[int] = None,
        vis: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        space : Space
            The criticality space to query.
        n_select : int
            Total evaluation budget (maximum number of SUT calls).
        n_initial : int, optional
            Number of initial LHS seed samples. Defaults to ``n_select // 5``.
        n_candidates : int, optional
            Number of uniform random candidates drawn each iteration (default 250).
        batch_size : int, optional
            Number of top-AC candidates queried per iteration (default 5).
        beta : float, optional
            Exploitation exponent in the acquisition score AC (default 1.0).
        idw_power : float, optional
            Distance exponent p for the IDW regressor (default 5.0).
        idw_n_neighbors : int or None, optional
            k-NN limit for local IDW; ``None`` uses all support points (default 5).
        epsilon : float, optional
            Distance floor for numerical stability in IDW (default 1e-12).
        rng_seed : int or None, optional
            Seed for the random candidate generator (default None).
        vis : bool, optional
            If True, render a 3-D IDW prediction surface after fitting (default False).
        """
        super().__init__(space, n_select)
        if n_initial is None:
            self.n_initial = int(n_select / 5)
        else:
            self.n_initial = n_initial
        self.n_candidates = int(n_candidates)
        self.batch_size = int(batch_size)
        self.beta = float(beta)
        self.idw_power = float(idw_power)
        self.idw_n_neighbors = idw_n_neighbors
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(rng_seed)

        self.vis = vis

        self.selected_points: List[Tuple[float, ...]] = []  # tested points (rounded)
        self._tested_y: List[float] = []                    # ground-truth values for tested points

    # --------------------------
    # Visualization
    # --------------------------

    def _visualize_2d_prediction_3d(
        self,
        model,
        resolution: int = 100,
        title: str = "3D Surface of IDW Predictions",
        alpha_surface: float = 0.9,
        scatter_size: int = 30,
        elev: float | None = None,
        azim: float | None = None,
    ):
        """
        Visualize IDW predictions over a 2D input space as a 3D surface
        and show tested points in red using their true values.

        Uses:
        - self.space.dimensions
        - self.selected_points
        - self._tested_y
        """
        assert len(self.space.dimensions) == 2, "3D visualization only supports 2D input spaces."

        bounds = self.space.dimensions
        tested_points = np.asarray(self.selected_points, dtype=float)
        tested_values = np.asarray(self._tested_y, dtype=float)

        # Build grid
        x_vals = np.linspace(bounds[0][0], bounds[0][1], resolution)
        y_vals = np.linspace(bounds[1][0], bounds[1][1], resolution)
        X, Y = np.meshgrid(x_vals, y_vals)
        grid_points = np.column_stack([X.ravel(), Y.ravel()])

        # Model prediction
        Z_pred = model.predict(grid_points).reshape(X.shape)

        # Plot surface
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z_pred, cmap="viridis", edgecolor="none", alpha=alpha_surface)

        # Plot tested points in red using TRUE values
        if tested_points.size > 0:
            ax.scatter(
                tested_points[:, 0], tested_points[:, 1], tested_values,
                c="red", s=scatter_size, depthshade=True, edgecolors="k", linewidths=0.5
            )

        # Labels
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Value")
        ax.set_title(title)
        # fig.colorbar(surf, shrink=0.5, aspect=10, label="Predicted Value")

        if elev is not None or azim is not None:
            ax.view_init(elev=elev if elev is not None else ax.elev,
                        azim=azim if azim is not None else ax.azim)

        plt.tight_layout()
        plt.show()

    # --------------------------
    # Internals / helper methods
    # --------------------------
    def _lhs_initial(self) -> np.ndarray:
        """Draw n_initial LHS points from bounds and round to space.decimal_precision."""
        bounds = self.space.dimensions
        d = len(bounds)
        sampler = qmc.LatinHypercube(d, seed=0)
        lhs = sampler.random(n=self.n_initial)
        X0 = qmc.scale(lhs, [b[0] for b in bounds], [b[1] for b in bounds])
        X0 = np.asarray([tuple(np.round(p, self.space.decimal_precision)) for p in X0], dtype=float)
        return X0

    def _sample_uniform(self, n: int) -> np.ndarray:
        """Draw n candidates uniformly from bounds (independent each iteration)."""
        bounds = self.space.dimensions
        d = len(bounds)
        X = self.rng.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(n, d)
        )
        return np.round(X, decimals=self.space.decimal_precision)

    def _build_surrogate(self, X: np.ndarray, y: np.ndarray) -> IDWRegressor:
        """Fit IDW on the currently tested set."""
        model = IDWRegressor(
            power=self.idw_power,
            n_neighbors=self.idw_n_neighbors,
            epsilon=self.epsilon
        )
        model.fit(X, y)
        return model

    def _compute_dmin(self, candidates: np.ndarray, tested: np.ndarray) -> np.ndarray:
        """D_min = min distance from a candidate to any tested point."""
        tree = cKDTree(tested)
        dmin, _ = tree.query(candidates, k=1, workers=-1)
        return dmin

    def _acquisition_scores(self, yhat: np.ndarray, dmin: np.ndarray) -> np.ndarray:
        """
        Maximization-oriented AC:
            AC = (y_hat - y_hat_min)^beta * D_min
        """
        y_min = np.min(yhat)
        exploit = np.maximum(yhat - y_min, 0.0) ** self.beta
        return exploit * dmin

    # ---------------
    # Public methods
    # ---------------
    def _init_model(self) -> List[Tuple[float, ...]]:
        """
        Run the full IDW selection loop and populate ``self.selected_points``.

        Seeds with ``n_initial`` LHS samples, then iterates the
        fit → candidate → score → query cycle until ``n_select`` points have
        been evaluated.

        Returns
        -------
        list of tuple
            All tested points as rounded coordinate tuples.
        """
        # ---- Initialisation with LHS ----
        X0 = self._lhs_initial()
        y0 = self.space.get_values_for_points(X0, save_points=True)

        self.selected_points = [tuple(p) for p in X0]
        self._tested_y = [float(v) for v in y0]

        # ---- Iterate until test budget is reached ----
        while len(self.selected_points) < self.n_select:
            # Fit surrogate on tested set
            X_tested = np.asarray(self.selected_points, dtype=float)
            y_tested = np.asarray(self._tested_y, dtype=float)
            model = self._build_surrogate(X_tested, y_tested)

            # Draw fresh random candidates and predict
            X_cand = self._sample_uniform(self.n_candidates)
            yhat = model.predict(X_cand)

            # Minimum distance to tested set (exploration term)
            dmin = self._compute_dmin(X_cand, X_tested)

            # Acquisition scores and top-k selection
            scores = self._acquisition_scores(yhat, dmin)
            top = min(self.batch_size, self.n_select - len(self.selected_points))
            idx_top = np.argsort(scores)[-top:]  # pick largest AC values
            X_sel = X_cand[idx_top]

            # Evaluate selected points with SUT, round, and add to tested set
            y_sel = self.space.get_values_for_points(X_sel, save_points=True)

            self.selected_points.extend(X_sel)
            self._tested_y.extend([float(v) for v in y_sel])

            if len(self.selected_points) >= self.n_select:
                break

        return self.selected_points

    def get_model(self) -> IDWRegressor:
        """
        Fit the IDW surrogate and return the trained model.

        Runs the full selection loop if no points have been tested yet, then
        fits a final IDW regressor on the complete tested set.

        Returns
        -------
        IDWRegressor
            Trained model with ``predict(X)`` compatible with the metric suite.
        """
        if len(self.selected_points) == 0:
            self._init_model()

        X = np.asarray(self.selected_points, dtype=float)
        y = np.asarray(self._tested_y, dtype=float)
        model = self._build_surrogate(X, y)

        if self.vis and len(self.space.dimensions) == 2:
            self._visualize_2d_prediction_3d(model, azim=-45, elev=45)

        return model

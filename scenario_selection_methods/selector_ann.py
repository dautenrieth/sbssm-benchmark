"""
Project: Toward Standardized Benchmarking of Search-Based Scenario Selection Methods in Autonomous System Validation
Version: 1.0.0

Description:
    Implements the Artificial Neural Network (ANN) surrogate-model selector.
    A two-hidden-layer feedforward network is trained on Latin Hypercube
    samples drawn from the criticality space. A lightweight grid search over
    hidden-layer width and learning rate is run in parallel via
    multiprocessing, and the configuration with the lowest training MSE is
    selected as the final surrogate model.

    - key role: Fits a neural-network surrogate of the criticality surface
                for model-reconstruction metric evaluation.
    - dependency: torch, numpy, scipy.stats.qmc, sklearn, base.BaseModelSelector
    - output: Trained SimpleANN instance compatible with
              space.metrics.run_metrics_suite(surrogate_model=...).

Usage:
    from selector_ann import ANNSelector
    selector = ANNSelector(space, n_select=100)
    model = selector.get_model()
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from base import BaseModelSelector
from scipy.stats import qmc
import torch.multiprocessing as mp
from itertools import product

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import logging
logger = logging.getLogger(__name__)


class SimpleANN(nn.Module):
    """
    Two-hidden-layer fully-connected network with ReLU activations.

    Accepts d-dimensional input and produces a scalar criticality estimate.
    Exposes a ``predict`` method that mirrors the scikit-learn interface for
    compatibility with the metric suite.
    """

    def __init__(self, input_dim, hidden_dim = 128):
        """
        Parameters
        ----------
        input_dim : int
            Number of input features (space dimensionality).
        hidden_dim : int, optional
            Width of both hidden layers (default 128).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Mimics scikit-learn's predict interface."""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            preds = self.forward(X_tensor).squeeze().cpu().numpy()
        return preds
    

class ANNSelector(BaseModelSelector):
    """
    Surrogate-model selector that fits a SimpleANN to the criticality space.

    Training data are drawn via Latin Hypercube Sampling. A grid search over
    hidden-layer width and learning rate is parallelised with
    ``torch.multiprocessing``; the best configuration (lowest training MSE)
    is returned as the final model.
    """

    def __init__(
        self, space, n_select, n_initial=100, n_epochs=10000, lr=1e-3
    ):
        """
        Parameters
        ----------
        space : Space
            The criticality space to query for training samples.
        n_select : int
            Sample budget (passed to BaseModelSelector; used as a reference
            budget, actual training size is controlled by ``n_initial``).
        n_initial : int, optional
            Number of LHS training samples (default 100).
        n_epochs : int, optional
            Maximum training epochs per configuration (default 10000).
            Early stopping may terminate training earlier.
        lr : float, optional
            Default learning rate (overridden by the grid search).
        """
        super().__init__(space, n_select)
        self.n_initial = n_initial
        self.n_epochs = n_epochs
        self.lr = lr

    def _visualize_2d_prediction_3d(self, model, bounds: list[tuple],
                                    resolution: int = 100,
                                    elev: float | None = None,
                                    azim: float | None = None,
                                    sample_points: np.ndarray | None = None,
                                    sample_values: np.ndarray | None = None):
        """
        Visualizes the prediction of a neural network model over a 2D input space in 3D.

        Args:
            model (torch.nn.Module): Trained PyTorch model with input shape (2,)
            bounds (list[tuple]): [(x_min, x_max), (y_min, y_max)]
            resolution (int): Number of grid points per axis
        """
        assert len(bounds) == 2, "3D visualization only supports 2D input spaces."

        model.eval()
        x_vals = np.linspace(bounds[0][0], bounds[0][1], resolution)
        y_vals = np.linspace(bounds[1][0], bounds[1][1], resolution)
        X, Y = np.meshgrid(x_vals, y_vals)
        grid_points = np.column_stack([X.ravel(), Y.ravel()])

        with torch.no_grad():
            predictions = model(torch.tensor(grid_points, dtype=torch.float32)).squeeze().numpy()

        Z = predictions.reshape(X.shape)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

        if elev is not None or azim is not None:
            ax.view_init(elev=elev if elev is not None else ax.elev,
                        azim=azim if azim is not None else ax.azim)
            
        if sample_points is not None and sample_values is not None:
            ax.scatter(
                sample_points[:, 0],
                sample_points[:, 1],
                sample_values,
                c="red",
                s=30,
                edgecolors="k",
                linewidths=0.5,
                depthshade=True
            )

        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Predicted Value")
        ax.set_title("3D Surface of ANN Predictions")
        # fig.colorbar(surf, shrink=0.5, aspect=10, label="Predicted Value")

        plt.tight_layout()
        plt.show()

    def _sample_initial_data(self, dim):
        """Draw ``n_initial`` LHS training points scaled to the space bounds."""
        sampler = qmc.LatinHypercube(dim, seed=0)
        samples = sampler.random(n=self.n_initial)
        scaled_samples = qmc.scale(
            samples,
            [b[0] for b in self.space.dimensions],
            [b[1] for b in self.space.dimensions]
        )
        return scaled_samples

    def _sample_candidates(self, dim):
        """Sample a uniform random candidate pool from the space bounds."""
        return np.random.uniform(
            low=[b[0] for b in self.space.dimensions],
            high=[b[1] for b in self.space.dimensions],
            size=(self.candidate_pool_size, dim)
        )

    def _train_and_evaluate_model(self, args):
        """
        Train a SimpleANN for one hyperparameter configuration and return metrics.

        Designed to be called by a multiprocessing pool; all arguments are
        packed into a single tuple so that ``pool.map`` can be used.

        Parameters
        ----------
        args : tuple
            ``(hidden_dim, lr, X_train_np, y_train_np, input_dim, n_epochs)``

        Returns
        -------
        dict
            Keys: ``hidden_dim``, ``lr``, ``mse``, ``state_dict``.
        """
        hidden_dim, lr, X_train_np, y_train_np, input_dim, n_epochs = args

        # Reconstruct tensors in subprocess
        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        y_train = torch.tensor(y_train_np, dtype=torch.float32)

        model = SimpleANN(input_dim=input_dim, hidden_dim=hidden_dim)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        best_loss = float("inf")
        patience_counter = 0
        patience = 200
        min_delta = 1e-5

        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            preds = model(X_train)
            loss = loss_fn(preds, y_train)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()

            # Early stopping logic
            if best_loss - current_loss > min_delta:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}, best_loss={best_loss:.6f}")
                break

        model.eval()
        with torch.no_grad():
            preds = model(X_train).squeeze().cpu().numpy()

        mse = mean_squared_error(y_train.cpu().numpy(), preds)

        return {
            "hidden_dim": hidden_dim,
            "lr": lr,
            "mse": mse,
            "state_dict": model.state_dict(),  # Save model weights only
        }


    def get_model(self) -> torch.nn.Module:
        """
        Fit the ANN surrogate and return the best-performing model.

        Samples training data via LHS, runs a parallel grid search over
        hidden-layer width and learning rate, and returns the SimpleANN
        instance with the lowest training MSE.

        Returns
        -------
        SimpleANN
            Trained model with ``predict(X)`` compatible with the metric suite.
        """
        dim = len(self.space.dimensions)

        # Sample training data
        initial_points = self._sample_initial_data(dim)
        initial_points = [tuple(np.round(p, self.space.decimal_precision)) for p in initial_points]
        initial_values = self.space.get_values_for_points(initial_points, save_points=True)

        X_train = torch.tensor(initial_points, dtype=torch.float32)
        y_train = torch.tensor(initial_values, dtype=torch.float32).unsqueeze(1)

        # Prepare numpy versions for multiprocessing
        X_train_np = X_train.numpy()
        y_train_np = y_train.numpy()

        # Define grid
        hidden_dims = [64, 32, 128]
        lrs = [1e-3, 1e-2]
        search_space = list(product(hidden_dims, lrs))

        # Prepare args
        args_list = [
            (h, lr, X_train_np, y_train_np, dim, self.n_epochs)
            for h, lr in search_space
        ]

        # Multiprocessing pool
        with mp.Pool(processes=min(mp.cpu_count(), len(args_list))) as pool:
            results = pool.map(self._train_and_evaluate_model, args_list)

        # Find best result
        best_result = min(results, key=lambda r: r["mse"])
        logger.info(
            f"\nBest config: hidden_dim={best_result['hidden_dim']}, "
            f"lr={best_result['lr']:.0e}, MSE={best_result['mse']:.4f}"
        )

        # Restore best model
        best_model = SimpleANN(input_dim=dim, hidden_dim=best_result["hidden_dim"])
        best_model.load_state_dict(best_result["state_dict"])

        if dim == 2:
            self._visualize_2d_prediction_3d(best_model, self.space.dimensions,
                azim=-45, elev=45, sample_points=X_train.numpy(),
                sample_values=y_train.squeeze().numpy())

        return best_model

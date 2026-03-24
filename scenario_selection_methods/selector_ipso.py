"""
Project: Toward Standardized Benchmarking of Search-Based Scenario Selection Methods in Autonomous System Validation
Version: 1.0.0

Description:
    Implements an Iterative Particle Swarm Optimisation (IPSO) selector.
    Particles are initialised via Latin Hypercube Sampling and updated each
    iteration using a local-best topology: each particle's neighbourhood is
    defined by a distance threshold S derived from the swarm's current spread.
    When the swarm converges (maximum inter-particle distance drops below
    ``restart_threshold``), a full reinitialization is triggered after three
    consecutive convergent iterations, preventing premature stagnation.

    - key role: Extremum-seeking sampler that concentrates queries near high-value
                regions of the criticality space for extremum-search metrics.
    - dependency: numpy, scipy.stats.qmc, matplotlib, base.BaseSelector
    - output: List of selected scenario-point tuples stored in space.selected_points.

Usage:
    from selector_ipso import IPSOSearchSelector
    selector = IPSOSearchSelector(space, n_select=100)
    selector.select()
"""

from base import BaseSelector
import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt


class IPSOSearchSelector(BaseSelector):
    """
    Iterative Particle Swarm Optimisation selector with local-best topology.

    Each iteration updates particle velocities using personal-best and
    neighbourhood-best attractors, clamps positions to the space bounds, and
    evaluates the criticality function. The swarm is reinitialised after three
    consecutive convergent iterations to escape local optima.
    """

    def __init__(
        self,
        space,
        n_select: int,
        n_particles: int = 10,
        w: float = 0.8,
        c1: float = 1.5 / 20,
        c2: float = 1.5 / 20,
        restart_threshold: float = 1,
        vis: bool = False,
    ):
        """
        Parameters
        ----------
        space : Space
            The criticality space to query.
        n_select : int
            Total number of scenario points to select (evaluation budget).
        n_particles : int, optional
            Swarm size (default 10).
        w : float, optional
            Inertia weight controlling velocity dampening (default 0.8).
        c1 : float, optional
            Cognitive acceleration coefficient — attraction to personal best
            (default 1.5/20).
        c2 : float, optional
            Social acceleration coefficient — attraction to neighbourhood best
            (default 1.5/20).
        restart_threshold : float, optional
            Maximum inter-particle distance below which the swarm is considered
            converged (default 1).
        vis : bool, optional
            If True, render a 2-D particle progression plot after selection
            (default False).
        """
        super().__init__(space, n_select)
        self.dimensions = space.dimensions
        self.dimension_count = len(self.dimensions)
        self.n_particles = n_particles
        self.w = w    # inertia
        self.c1 = c1  # cognitive — attraction to personal best
        self.c2 = c2  # social — attraction to neighbourhood best
        self.restart_threshold = restart_threshold
        self.decimal_precision = space.decimal_precision
        self.vis = vis

    def _initialize_particles(self):
        """
        Initialise particle positions via LHS and velocities from a uniform draw.

        Initial velocities are sampled uniformly from ±15 % of each dimension's
        range to avoid large initial displacements.

        Returns
        -------
        positions : np.ndarray, shape (n_particles, d)
            LHS-distributed starting positions within the space bounds.
        velocities : np.ndarray, shape (n_particles, d)
            Initial velocity vectors.
        """
        sampler = qmc.LatinHypercube(d=self.dimension_count)
        initial_positions = qmc.scale(
            sampler.random(n=self.n_particles),
            [b[0] for b in self.dimensions],
            [b[1] for b in self.dimensions],
        )
        velocities = np.zeros_like(initial_positions)
        for d in range(self.dimension_count):
            low, high = self.dimensions[d]
            range_d = 0.15 * (high - low)
            velocities[:, d] = np.random.uniform(-range_d, range_d, self.n_particles)
        return initial_positions, velocities

    def _evaluate(self, positions):
        """
        Query the criticality space at the given particle positions.

        Parameters
        ----------
        positions : np.ndarray, shape (n_particles, d)
            Current particle positions.

        Returns
        -------
        np.ndarray, shape (n_particles,)
            Criticality values at each position.
        """
        return self.space.get_values_for_points(positions, save_points=True)

    def _update_velocity(self, velocities, positions, pbest, lbest):
        """
        Compute updated velocities using the PSO update rule.

        v ← w·v + c1·r1·(pbest − x) + c2·r2·(lbest − x)

        Parameters
        ----------
        velocities : np.ndarray, shape (n_particles, d)
        positions : np.ndarray, shape (n_particles, d)
        pbest : np.ndarray, shape (n_particles, d)
            Personal-best positions.
        lbest : np.ndarray, shape (n_particles, d)
            Neighbourhood-best positions for each particle.

        Returns
        -------
        np.ndarray, shape (n_particles, d)
            Updated velocities.
        """
        r1 = np.random.rand(self.n_particles, self.dimension_count)
        r2 = np.random.rand(self.n_particles, self.dimension_count)
        return (
            self.w * velocities
            + self.c1 * r1 * (pbest - positions)
            + self.c2 * r2 * (lbest - positions)
        )

    def _get_local_best(self, positions, fitness):
        """
        Determine the neighbourhood-best position for each particle.

        The neighbourhood radius S is derived from the current maximum
        inter-particle distance scaled by the swarm size.

        Parameters
        ----------
        positions : np.ndarray, shape (n_particles, d)
        fitness : np.ndarray, shape (n_particles,)

        Returns
        -------
        np.ndarray, shape (n_particles, d)
            Best position found within each particle's neighbourhood.
        """
        lbest = np.copy(positions)
        max_dist = np.max(np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1))
        S = max_dist / 2 * self.n_particles

        for i in range(self.n_particles):
            distances = np.linalg.norm(positions - positions[i], axis=1)
            mask = (distances > 0) & (distances <= S)
            if np.any(mask):
                neighbor_idx = np.argmax(fitness[mask])
                lbest[i] = positions[mask][neighbor_idx]
            else:
                lbest[i] = positions[i]
                print("No neighbors available")
        return lbest

    def _has_converged(self, positions):
        """
        Check whether the swarm has converged.

        Returns True if the maximum pairwise distance between particles drops
        below ``restart_threshold``.

        Parameters
        ----------
        positions : np.ndarray, shape (n_particles, d)

        Returns
        -------
        bool
        """
        dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        return np.max(dists) < self.restart_threshold
    
    def plot_ipso_progression(self, history, bounds, title="IPSO Particle Progression"):
        """
        Visualise particle positions across IPSO iterations (2-D spaces only).

        Particles are coloured from cool (early) to warm (late) iterations, with
        opacity increasing over time to highlight progressive convergence.

        Parameters
        ----------
        history : list of np.ndarray, each shape (n_particles, 2)
            Particle positions recorded at each iteration.
        bounds : list of tuple
            Per-dimension ``(lower, upper)`` bounds used as axis limits.
        title : str, optional
            Plot title (default ``"IPSO Particle Progression"``).
        """
        cmap = plt.cm.cool
        n_iters = len(history)

        plt.figure(figsize=(8, 6))
        for i, positions in enumerate(history):
            alpha = 0.3 + 0.7 * (i / max(1, n_iters - 1))
            color = cmap(i / max(1, n_iters - 1))
            plt.scatter(
                positions[:, 0], positions[:, 1],
                color=color,
                alpha=alpha,
                edgecolor='k',
                s=30,
                label=f"Iter {i+1}" if i in [0, n_iters - 1] else None
            )

        plt.xlim(*bounds[0])
        plt.ylim(*bounds[1])
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        # plt.title(title)
        plt.grid(True)
        plt.gca().set_aspect('equal')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def select(self) -> list[tuple]:
        """
        Run the IPSO selection loop and store results in the space.

        Particles are evaluated each iteration; personal bests are updated when
        fitness improves. After three consecutive convergent iterations the swarm
        is fully reinitialised. The position history is truncated to ``n_select``
        snapshots and flattened to produce the final point list.

        Returns
        -------
        list of tuple
            Up to ``n_select`` selected scenario parameter points.
        """
        history = []
        evaluated_points = 0
        convergence_counter = 0
        positions, velocities = self._initialize_particles()
        history.append(positions.copy())
        fitness = self._evaluate(positions)
        evaluated_points += len(positions)
        pbest = np.copy(positions)
        pbest_fitness = np.copy(fitness)

        while evaluated_points < self.n_select:
            history.append(positions.copy())
            lbest = self._get_local_best(positions, pbest_fitness)
            velocities = self._update_velocity(velocities, positions, pbest, lbest)
            positions += velocities

            # Clamp particles to the space bounds
            for i, (low, high) in enumerate(self.dimensions):
                positions[:, i] = np.clip(positions[:, i], low, high)

            fitness = self._evaluate(positions)
            evaluated_points += len(positions)
            improved = fitness > pbest_fitness
            pbest[improved] = positions[improved]
            pbest_fitness[improved] = fitness[improved]

            if self._has_converged(positions):
                print("Swarm converged — incrementing restart counter.")
                convergence_counter += 1
                if convergence_counter >= 3:
                    positions, velocities = self._initialize_particles()
                    fitness = self._evaluate(positions)
                    pbest = np.copy(positions)
                    pbest_fitness = np.copy(fitness)
                    convergence_counter = 0
            else:
                convergence_counter = 0

        history = history[: self.n_select]

        if self.vis:
            self.plot_ipso_progression(history, bounds=self.dimensions)

        all_points = np.concatenate(history, axis=0)
        rounded_points = [tuple(np.round(p, self.decimal_precision)) for p in all_points]
        return rounded_points
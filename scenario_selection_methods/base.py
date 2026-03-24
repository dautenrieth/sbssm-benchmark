"""
Project: Toward Standardized Benchmarking of Search-Based Scenario Selection Methods in Autonomous System Validation
Version: 1.0.0

Description:
    Defines the abstract base classes shared by all scenario selection methods.
    Every selector must inherit from exactly one of these bases and implement
    the corresponding abstract method, which is called by the evaluation loop
    in main.py.

    - key role: Enforces a uniform interface across all nine SBSSMs so that
                main.py can dispatch them generically via get_selector().
    - dependency: abc
    - output: None (interface definition only).
"""

from abc import ABC, abstractmethod


class BaseSelector(ABC):
    """
    Base class for selectors that directly return a set of sampled points.

    Subclasses implement point-based methods (e.g. random, LHS, IPSO, NNDV)
    where the result is a list of selected scenario coordinate tuples.
    """

    def __init__(self, space, n_select: int):
        """
        Parameters
        ----------
        space : Space
            The criticality space to query during selection.
        n_select : int
            Number of scenario points to select (sample budget).
        """
        self.space = space
        self.n_select = n_select

    @abstractmethod
    def select(self) -> list[tuple]:
        """
        Run the selection algorithm and return the chosen scenario points.

        Returns
        -------
        list of tuple
            The selected scenario parameter points.
        """
        pass


class BaseModelSelector(ABC):
    """
    Base class for selectors that fit a surrogate model over the space.

    Subclasses implement model-based methods (e.g. GPR, ANN, IDW) where the
    primary output is a trained model rather than a fixed point set. The model
    is passed to the metric suite for surrogate-fidelity evaluation.
    """

    def __init__(self, space, n_select: int):
        """
        Parameters
        ----------
        space : Space
            The criticality space to query during model fitting.
        n_select : int
            Number of training points to sample (sample budget).
        """
        self.space = space
        self.n_select = n_select

    @abstractmethod
    def get_model(self):
        """
        Fit the surrogate model and return it.

        Returns
        -------
        object
            A trained model object compatible with
            ``space.metrics.run_metrics_suite(surrogate_model=...)``.
        """
        pass
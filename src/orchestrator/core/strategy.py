"""Experiment generation strategies (pipeline stage 2: *generate experiments*).

A *strategy* turns an :class:`~orchestrator.core.models.Objective` into concrete
:class:`~orchestrator.core.models.Experiment` proposals by sampling the
objective's search space. Different strategies embody different search policies
-- random search, grid search, Bayesian optimization, and so on.

This module defines only the **interface** and a lightweight **registry**.
Concrete strategies live in their own modules and register themselves so the
rest of the pipeline can select one by name without importing it directly.

The interface is deliberately feedback-aware: :meth:`ExperimentStrategy.propose`
receives the history of already-run experiments, which adaptive strategies use
to steer the search (pipeline stage 5: *adjust hyperparameters*). Stateless
strategies such as random search simply ignore it.

Example
-------
.. code-block:: python

    from orchestrator.core.strategy import ExperimentStrategy, register_strategy

    @register_strategy
    class RandomSearch(ExperimentStrategy):
        name = "random"

        def propose(self, objective, *, count, history=None):
            return [
                self._new_experiment(objective, {...}, index=i)
                for i in range(count)
            ]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from orchestrator.core.models import Experiment, Objective


class StrategyError(Exception):
    """Raised for invalid strategy usage or registry lookups."""


class ExperimentStrategy(ABC):
    """Interface every experiment-generation strategy implements.

    Subclasses must set a unique :attr:`name` and implement :meth:`propose`.
    Use :meth:`_new_experiment` to construct proposals so that the link back to
    the originating objective and the default naming stay consistent.
    """

    #: Stable identifier used for registry lookup, selection, and reporting.
    name: ClassVar[str] = ""

    @abstractmethod
    def propose(
        self,
        objective: Objective,
        *,
        count: int,
        history: Sequence[Experiment] | None = None,
    ) -> list[Experiment]:
        """Propose up to ``count`` new experiments for ``objective``.

        Parameters
        ----------
        objective:
            The validated objective whose search space is being explored.
        count:
            The maximum number of experiments to propose. Callers typically pass
            the remaining experiment budget; a strategy may return fewer (for
            example, grid search once the grid is exhausted) but never more.
        history:
            Previously generated experiments, including any recorded metrics.
            Adaptive strategies use this to inform the next proposals; stateless
            strategies may ignore it.

        Returns
        -------
        list[Experiment]
            Newly created ``PENDING`` experiments, each linked to ``objective``.
        """

    @staticmethod
    def _check_count(count: int) -> None:
        """Validate the ``count`` argument shared by all strategies."""
        if count < 1:
            raise StrategyError(f"count must be a positive integer, got {count}")

    @staticmethod
    def _new_experiment(
        objective: Objective,
        hyperparameters: Mapping[str, Any],
        *,
        index: int | None = None,
        name: str | None = None,
    ) -> Experiment:
        """Build a ``PENDING`` experiment linked to ``objective``.

        Provide either an explicit ``name`` or an ``index`` (used to derive a
        readable default like ``"tune-resnet-0001"``).
        """
        if name is None:
            suffix = f"{index:04d}" if index is not None else "exp"
            name = f"{objective.name}-{suffix}"
        return Experiment(
            objective_id=objective.id,
            name=name,
            hyperparameters=dict(hyperparameters),
        )


_REGISTRY: dict[str, type[ExperimentStrategy]] = {}


def register_strategy(cls: type[ExperimentStrategy]) -> type[ExperimentStrategy]:
    """Class decorator that registers a strategy under its :attr:`name`.

    Raises
    ------
    StrategyError
        If the class declares no ``name`` or the name is already registered.
    """
    name = getattr(cls, "name", "")
    if not name:
        raise StrategyError(f"{cls.__name__} must define a non-empty 'name' to be registered")
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise StrategyError(f"strategy name {name!r} already registered to {existing.__name__}")
    _REGISTRY[name] = cls
    return cls


def get_strategy(name: str, **kwargs: Any) -> ExperimentStrategy:
    """Instantiate a registered strategy by ``name``.

    Extra keyword arguments are forwarded to the strategy's constructor.

    Raises
    ------
    StrategyError
        If no strategy is registered under ``name``.
    """
    try:
        cls = _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise StrategyError(f"unknown strategy {name!r}; registered: {known}") from None
    return cls(**kwargs)


def available_strategies() -> list[str]:
    """Return the sorted names of all registered strategies."""
    return sorted(_REGISTRY)

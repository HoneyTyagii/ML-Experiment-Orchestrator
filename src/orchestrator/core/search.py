"""Grid and random hyperparameter search strategies.

Two concrete :class:`~orchestrator.core.strategy.ExperimentStrategy`
implementations that cover the common, dependency-free baselines:

* :class:`RandomSearch` -- draws independent random samples from the search
  space. Stateless and seedable for reproducibility.
* :class:`GridSearch` -- enumerates the cartesian product of the search space in
  a deterministic order. Continuous ``float`` dimensions are discretized into a
  fixed number of points; the product is generated lazily so large grids stay
  cheap until the points are actually requested.

Both register themselves with the strategy registry (as ``"random"`` and
``"grid"``) on import, so :func:`orchestrator.core.strategy.get_strategy` can
resolve them by name.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from itertools import islice, product
from typing import Any, ClassVar

from orchestrator.core.models import (
    Experiment,
    HyperparameterSpec,
    HyperparameterType,
    Objective,
)
from orchestrator.core.strategy import ExperimentStrategy, StrategyError, register_strategy


def _validate_space(objective: Objective) -> None:
    """Fail fast with a strategy error if the search space is inconsistent."""
    try:
        objective.validate_search_space()
    except ValueError as exc:
        raise StrategyError(f"cannot sample invalid search space: {exc}") from exc


@register_strategy
class RandomSearch(ExperimentStrategy):
    """Sample hyperparameter configurations uniformly at random.

    Parameters
    ----------
    seed:
        Optional seed for the internal RNG. Pass an integer for reproducible
        proposals; leave as ``None`` for nondeterministic sampling.
    """

    name: ClassVar[str] = "random"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def propose(
        self,
        objective: Objective,
        *,
        count: int,
        history: Sequence[Experiment] | None = None,
    ) -> list[Experiment]:
        self._check_count(count)
        _validate_space(objective)

        start = len(history or [])

        # With nothing to vary, a single default configuration is all there is.
        if not objective.search_space:
            return [] if start >= 1 else [self._new_experiment(objective, {}, index=0)]

        experiments: list[Experiment] = []
        for offset in range(count):
            config = {spec.name: self._sample(spec) for spec in objective.search_space}
            experiments.append(self._new_experiment(objective, config, index=start + offset))
        return experiments

    def _sample(self, spec: HyperparameterSpec) -> Any:
        """Draw a single value for ``spec``."""
        if spec.type is HyperparameterType.FLOAT:
            return self._sample_numeric(spec, as_int=False)
        if spec.type is HyperparameterType.INT:
            return self._sample_numeric(spec, as_int=True)
        if spec.type is HyperparameterType.CATEGORICAL:
            return self._rng.choice(spec.choices or [])
        if spec.type is HyperparameterType.BOOL:
            return self._rng.choice(spec.choices or [True, False])
        raise StrategyError(f"unsupported hyperparameter type: {spec.type}")

    def _sample_numeric(self, spec: HyperparameterSpec, *, as_int: bool) -> float | int:
        low, high = float(spec.low), float(spec.high)  # type: ignore[arg-type]
        if spec.log_scale and low > 0 and high > 0:
            value = math.exp(self._rng.uniform(math.log(low), math.log(high)))
        else:
            value = self._rng.uniform(low, high)
        if as_int:
            return int(round(value))
        return value


@register_strategy
class GridSearch(ExperimentStrategy):
    """Enumerate the cartesian product of the search space deterministically.

    Discrete dimensions (``int``, ``categorical``, ``bool``) are enumerated
    exhaustively; continuous ``float`` dimensions are discretized into
    ``points_per_float`` evenly spaced points (geometrically spaced when the spec
    declares ``log_scale``).

    Repeated calls continue where the previous call stopped, using ``len(history)``
    as the offset into the (stably ordered) product. Once the grid is exhausted,
    :meth:`propose` returns an empty list.

    Parameters
    ----------
    points_per_float:
        Number of points to sample along each continuous ``float`` dimension.
    """

    name: ClassVar[str] = "grid"

    def __init__(self, points_per_float: int = 5) -> None:
        if points_per_float < 1:
            raise StrategyError("points_per_float must be a positive integer")
        self.points_per_float = points_per_float

    def propose(
        self,
        objective: Objective,
        *,
        count: int,
        history: Sequence[Experiment] | None = None,
    ) -> list[Experiment]:
        self._check_count(count)
        _validate_space(objective)

        start = len(history or [])

        if not objective.search_space:
            return [] if start >= 1 else [self._new_experiment(objective, {}, index=0)]

        names = [spec.name for spec in objective.search_space]
        value_lists = [self._grid_values(spec) for spec in objective.search_space]

        combos = islice(product(*value_lists), start, start + count)
        experiments: list[Experiment] = []
        for offset, combo in enumerate(combos):
            config = dict(zip(names, combo))
            experiments.append(self._new_experiment(objective, config, index=start + offset))
        return experiments

    def _grid_values(self, spec: HyperparameterSpec) -> list[Any]:
        """Return the discrete set of values to enumerate for ``spec``."""
        if spec.type is HyperparameterType.CATEGORICAL:
            return list(spec.choices or [])
        if spec.type is HyperparameterType.BOOL:
            return list(spec.choices) if spec.choices else [True, False]
        if spec.type is HyperparameterType.INT:
            low, high = int(round(spec.low)), int(round(spec.high))  # type: ignore[arg-type]
            return list(range(low, high + 1))
        if spec.type is HyperparameterType.FLOAT:
            return self._float_points(spec)
        raise StrategyError(f"unsupported hyperparameter type: {spec.type}")

    def _float_points(self, spec: HyperparameterSpec) -> list[float]:
        low, high = float(spec.low), float(spec.high)  # type: ignore[arg-type]
        n = self.points_per_float
        if low == high or n == 1:
            return [low]
        if spec.log_scale and low > 0 and high > 0:
            lo, hi = math.log(low), math.log(high)
            return [math.exp(lo + (hi - lo) * i / (n - 1)) for i in range(n)]
        return [low + (high - low) * i / (n - 1) for i in range(n)]

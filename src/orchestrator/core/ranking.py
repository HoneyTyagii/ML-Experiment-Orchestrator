"""Run comparison and ranking (pipeline stage 6: *compare results*).

Once experiments finish, the orchestrator must decide which one "won" relative
to the objective. This module ranks experiments by their primary metric under
the objective's goal, with optional secondary-metric tie-breaking, and exposes
the result as a :class:`Leaderboard` ready for reporting and deployment.

Scoring uses each experiment's *best* observed value for a metric by default
(``method="best"``), or the latest observation (``method="last"``). Experiments
that never recorded the primary metric -- failures, or runs cancelled before
reporting -- cannot be scored and are surfaced separately as ``unranked`` rather
than silently dropped.

Ties on the primary metric are broken by the ``secondary`` metrics in order;
any remaining ties preserve input order (a stable sort), which for a tuning run
is generation order.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from orchestrator.core.models import Experiment, ExperimentStatus, Goal

if TYPE_CHECKING:  # avoid an import cycle; only needed for type hints
    from orchestrator.core.loop import TuningResult

#: A metric to rank or tie-break by, paired with its optimization direction.
MetricKey = tuple[str, Goal]

_WORST = math.inf  # transformed-space sentinel for "missing/worst"


def score_experiment(
    experiment: Experiment, metric: str, goal: Goal, *, method: str = "best"
) -> float | None:
    """Return an experiment's score for ``metric``, or ``None`` if unavailable.

    ``method="best"`` uses the best observed value under ``goal``; ``method="last"``
    uses the most recent observation.
    """
    if method == "best":
        return experiment.best_metric(metric, goal)
    if method in ("last", "latest"):
        return experiment.latest_metric(metric)
    raise ValueError(f"unknown scoring method: {method!r}")


def _transform(value: float | None, goal: Goal) -> float:
    """Map a score into a space where smaller is always better (for sorting)."""
    if value is None:
        return _WORST
    return value if goal is Goal.MINIMIZE else -value


def compare_experiments(
    a: Experiment, b: Experiment, metric: str, goal: Goal, *, method: str = "best"
) -> int:
    """Three-way compare two experiments by ``metric``: -1 if ``a`` is better.

    Returns ``-1`` when ``a`` ranks ahead of ``b``, ``1`` when behind, ``0`` when
    tied. Missing scores rank last.
    """
    ta = _transform(score_experiment(a, metric, goal, method=method), goal)
    tb = _transform(score_experiment(b, metric, goal, method=method), goal)
    if ta < tb:
        return -1
    if ta > tb:
        return 1
    return 0


@dataclass(frozen=True)
class RankedExperiment:
    """One row of a :class:`Leaderboard`."""

    rank: int
    experiment: Experiment
    score: float
    metrics: dict[str, float]

    @property
    def experiment_id(self) -> str:
        return self.experiment.id

    @property
    def is_best(self) -> bool:
        return self.rank == 1


@dataclass
class RankingSummary:
    """Aggregate statistics over the scored experiments."""

    metric: str
    goal: Goal
    count: int
    best: float | None = None
    worst: float | None = None
    mean: float | None = None
    median: float | None = None


@dataclass
class Leaderboard:
    """Experiments ranked best-first by the primary metric."""

    metric: str
    goal: Goal
    entries: list[RankedExperiment] = field(default_factory=list)
    unranked: list[Experiment] = field(default_factory=list)

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def best(self) -> RankedExperiment | None:
        """The top-ranked entry, or ``None`` if nothing could be scored."""
        return self.entries[0] if self.entries else None

    def top(self, n: int) -> list[RankedExperiment]:
        """The ``n`` best entries."""
        return self.entries[:n]

    def get(self, experiment_id: str) -> RankedExperiment | None:
        """Look up an entry by experiment id."""
        for entry in self.entries:
            if entry.experiment.id == experiment_id:
                return entry
        return None

    def rows(self) -> list[dict]:
        """Flat, serializable rows for reporting/tabulation."""
        return [
            {
                "rank": e.rank,
                "experiment_id": e.experiment.id,
                "name": e.experiment.name,
                "score": e.score,
                "status": e.experiment.status.value,
                **{f"metric.{k}": v for k, v in e.metrics.items()},
                **{f"hp.{k}": v for k, v in e.experiment.hyperparameters.items()},
            }
            for e in self.entries
        ]

    def summary(self, *, method: str = "best") -> RankingSummary:
        """Aggregate statistics over the ranked scores."""
        scores = [e.score for e in self.entries]
        if not scores:
            return RankingSummary(metric=self.metric, goal=self.goal, count=0)
        return RankingSummary(
            metric=self.metric,
            goal=self.goal,
            count=len(scores),
            best=scores[0],
            worst=scores[-1],
            mean=statistics.fmean(scores),
            median=statistics.median(scores),
        )


def rank_experiments(
    experiments: Sequence[Experiment],
    metric: str,
    goal: Goal,
    *,
    secondary: Sequence[MetricKey] = (),
    method: str = "best",
    statuses: Sequence[ExperimentStatus] | None = None,
) -> Leaderboard:
    """Rank ``experiments`` best-first by ``metric`` under ``goal``.

    Parameters
    ----------
    secondary:
        Additional ``(metric, goal)`` keys used, in order, to break ties on the
        primary metric.
    method:
        ``"best"`` (default) or ``"last"`` -- how to reduce each metric's
        observations to a single score.
    statuses:
        If given, only experiments in these statuses are considered for ranking.
    """
    keys: list[MetricKey] = [(metric, goal), *secondary]

    candidates: list[Experiment] = []
    unranked: list[Experiment] = []
    for exp in experiments:
        if statuses is not None and exp.status not in statuses:
            unranked.append(exp)
            continue
        if score_experiment(exp, metric, goal, method=method) is None:
            unranked.append(exp)
            continue
        candidates.append(exp)

    def sort_key(exp: Experiment) -> tuple[float, ...]:
        return tuple(_transform(score_experiment(exp, m, g, method=method), g) for m, g in keys)

    ordered = sorted(candidates, key=sort_key)

    entries: list[RankedExperiment] = []
    for rank, exp in enumerate(ordered, start=1):
        metric_values: dict[str, float] = {}
        for m, g in keys:
            value = score_experiment(exp, m, g, method=method)
            if value is not None:
                metric_values[m] = value
        entries.append(
            RankedExperiment(
                rank=rank,
                experiment=exp,
                score=score_experiment(exp, metric, goal, method=method),  # type: ignore[arg-type]
                metrics=metric_values,
            )
        )

    return Leaderboard(metric=metric, goal=goal, entries=entries, unranked=unranked)


def rank_result(
    result: TuningResult,
    *,
    secondary: Sequence[MetricKey] = (),
    method: str = "best",
    statuses: Sequence[ExperimentStatus] | None = None,
) -> Leaderboard:
    """Rank the experiments of a :class:`~orchestrator.core.loop.TuningResult`."""
    return rank_experiments(
        result.experiments,
        result.primary_metric,
        result.goal,
        secondary=secondary,
        method=method,
        statuses=statuses,
    )

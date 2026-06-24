"""Early-stopping policies and metric thresholds.

Sitting between monitoring (stage 4) and hyperparameter adjustment (stage 5),
this module decides *when to cut a run short*. Stopping early frees capacity for
more promising configurations and ends runs that have already met their target.

The design separates two concerns:

* **Decision logic** -- :class:`StoppingPolicy` implementations are pure
  functions over a metric series. They have no knowledge of launchers or threads
  and are trivially testable. :class:`EarlyStopper` composes several policies and
  tracks per-experiment history.
* **Integration** -- :class:`EarlyStoppingListener` adapts an
  :class:`EarlyStopper` to the monitor's :class:`~orchestrator.core.monitor.MetricListener`
  interface: as metrics stream in it consults the policies and, on a stop
  decision, cancels the corresponding job through the launcher.

Built-in policies
-----------------
* :class:`TargetThresholdPolicy` -- stop once the metric reaches the objective's
  target value (success).
* :class:`PatiencePolicy` -- stop when the metric stops improving for ``patience``
  observations (plateau).
* :class:`FloorThresholdPolicy` -- prune a run that is still underperforming a
  threshold after a warm-up step.
* :class:`DivergencePolicy` -- stop immediately on a NaN/infinite metric.
* :class:`MaxStepsPolicy` -- stop once a step cap is exceeded.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from orchestrator.core.launcher import TrainingJob, TrainingLauncher
from orchestrator.core.models import ExperimentStatus, Goal, MetricValue
from orchestrator.core.monitor import MetricEvent, MetricListener


class StopReason(str, Enum):
    """Why an experiment was stopped early."""

    TARGET_REACHED = "target_reached"
    NO_IMPROVEMENT = "no_improvement"
    BELOW_THRESHOLD = "below_threshold"
    DIVERGED = "diverged"
    MAX_STEPS = "max_steps"


@dataclass(frozen=True)
class StopDecision:
    """The outcome of evaluating stopping policies for one experiment."""

    reason: StopReason
    message: str
    metric: str
    value: float
    step: int
    experiment_id: str | None = None

    def with_experiment(self, experiment_id: str) -> StopDecision:
        """Return a copy tagged with the owning experiment id."""
        return StopDecision(
            reason=self.reason,
            message=self.message,
            metric=self.metric,
            value=self.value,
            step=self.step,
            experiment_id=experiment_id,
        )


def _improved(value: float, best: float, goal: Goal, min_delta: float) -> bool:
    """Whether ``value`` is a meaningful improvement over ``best`` under ``goal``."""
    if goal is Goal.MAXIMIZE:
        return value > best + min_delta
    return value < best - min_delta


class StoppingPolicy(ABC):
    """A pure decision rule over the observations of a single metric.

    A policy watches exactly one metric (:attr:`metric`) and is evaluated each
    time a new observation of that metric arrives, with the full ordered history
    for that experiment.
    """

    metric: str

    @abstractmethod
    def evaluate(self, history: Sequence[MetricValue]) -> StopDecision | None:
        """Return a :class:`StopDecision` to stop, or ``None`` to continue."""


class TargetThresholdPolicy(StoppingPolicy):
    """Stop once the metric reaches a target value (the objective is satisfied)."""

    def __init__(self, metric: str, goal: Goal, target: float) -> None:
        self.metric = metric
        self.goal = goal
        self.target = target

    def _meets(self, value: float) -> bool:
        return value >= self.target if self.goal is Goal.MAXIMIZE else value <= self.target

    def evaluate(self, history: Sequence[MetricValue]) -> StopDecision | None:
        for obs in history:
            if self._meets(obs.value):
                return StopDecision(
                    reason=StopReason.TARGET_REACHED,
                    message=f"{self.metric}={obs.value:g} reached target {self.target:g}",
                    metric=self.metric,
                    value=obs.value,
                    step=obs.step,
                )
        return None


class PatiencePolicy(StoppingPolicy):
    """Stop when the metric fails to improve for ``patience`` observations."""

    def __init__(self, metric: str, goal: Goal, patience: int, min_delta: float = 0.0) -> None:
        if patience < 1:
            raise ValueError("patience must be a positive integer")
        self.metric = metric
        self.goal = goal
        self.patience = patience
        self.min_delta = min_delta

    def evaluate(self, history: Sequence[MetricValue]) -> StopDecision | None:
        best: float | None = None
        since = 0
        last = history[-1] if history else None
        for obs in history:
            if best is None or _improved(obs.value, best, self.goal, self.min_delta):
                best = obs.value
                since = 0
            else:
                since += 1
        if last is not None and since >= self.patience:
            return StopDecision(
                reason=StopReason.NO_IMPROVEMENT,
                message=(
                    f"{self.metric} did not improve for {since} observations "
                    f"(patience={self.patience})"
                ),
                metric=self.metric,
                value=last.value,
                step=last.step,
            )
        return None


class FloorThresholdPolicy(StoppingPolicy):
    """Prune a run still underperforming ``threshold`` after a warm-up step."""

    def __init__(self, metric: str, goal: Goal, threshold: float, after_step: int = 0) -> None:
        self.metric = metric
        self.goal = goal
        self.threshold = threshold
        self.after_step = after_step

    def _underperforms(self, value: float) -> bool:
        return value < self.threshold if self.goal is Goal.MAXIMIZE else value > self.threshold

    def evaluate(self, history: Sequence[MetricValue]) -> StopDecision | None:
        if not history:
            return None
        last = history[-1]
        if last.step >= self.after_step and self._underperforms(last.value):
            return StopDecision(
                reason=StopReason.BELOW_THRESHOLD,
                message=(
                    f"{self.metric}={last.value:g} below threshold {self.threshold:g} "
                    f"at step {last.step}"
                ),
                metric=self.metric,
                value=last.value,
                step=last.step,
            )
        return None


class DivergencePolicy(StoppingPolicy):
    """Stop immediately if the metric becomes NaN or infinite."""

    def __init__(self, metric: str) -> None:
        self.metric = metric

    def evaluate(self, history: Sequence[MetricValue]) -> StopDecision | None:
        if not history:
            return None
        last = history[-1]
        if math.isnan(last.value) or math.isinf(last.value):
            return StopDecision(
                reason=StopReason.DIVERGED,
                message=f"{self.metric} diverged to {last.value} at step {last.step}",
                metric=self.metric,
                value=last.value,
                step=last.step,
            )
        return None


class MaxStepsPolicy(StoppingPolicy):
    """Stop once the metric's step index reaches ``max_steps``."""

    def __init__(self, metric: str, max_steps: int) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.metric = metric
        self.max_steps = max_steps

    def evaluate(self, history: Sequence[MetricValue]) -> StopDecision | None:
        if not history:
            return None
        last = history[-1]
        if last.step >= self.max_steps:
            return StopDecision(
                reason=StopReason.MAX_STEPS,
                message=f"{self.metric} reached step {last.step} (max_steps={self.max_steps})",
                metric=self.metric,
                value=last.value,
                step=last.step,
            )
        return None


class EarlyStopper:
    """Composes stopping policies and tracks per-experiment metric history.

    Feed observations via :meth:`update`; the first policy to fire wins, and the
    decision is cached so an experiment is only stopped once.
    """

    def __init__(self, policies: Sequence[StoppingPolicy] | None = None) -> None:
        self._policies: list[StoppingPolicy] = list(policies or [])
        self._history: dict[str, dict[str, list[MetricValue]]] = {}
        self._decisions: dict[str, StopDecision] = {}

    @classmethod
    def from_objective(
        cls,
        objective,
        *,
        patience: int | None = None,
        min_delta: float = 0.0,
        watch_divergence: bool = True,
    ) -> EarlyStopper:
        """Build a stopper from an objective's primary metric, goal, and target."""
        policies: list[StoppingPolicy] = []
        metric, goal = objective.primary_metric, objective.goal
        if objective.target_metric_value is not None:
            policies.append(TargetThresholdPolicy(metric, goal, objective.target_metric_value))
        if patience is not None:
            policies.append(PatiencePolicy(metric, goal, patience, min_delta))
        if watch_divergence:
            policies.append(DivergencePolicy(metric))
        return cls(policies)

    def add_policy(self, policy: StoppingPolicy) -> None:
        """Append a policy to the evaluation order."""
        self._policies.append(policy)

    def decision_for(self, experiment_id: str) -> StopDecision | None:
        """Return the cached stop decision for ``experiment_id``, if any."""
        return self._decisions.get(experiment_id)

    def update(self, experiment_id: str, metric: MetricValue) -> StopDecision | None:
        """Record an observation and return a stop decision if one is triggered."""
        if experiment_id in self._decisions:
            return None  # already stopped; ignore further observations

        per_metric = self._history.setdefault(experiment_id, {})
        series = per_metric.setdefault(metric.name, [])
        series.append(metric)

        for policy in self._policies:
            if policy.metric != metric.name:
                continue
            decision = policy.evaluate(series)
            if decision is not None:
                tagged = decision.with_experiment(experiment_id)
                self._decisions[experiment_id] = tagged
                return tagged
        return None


class EarlyStoppingListener(MetricListener):
    """Monitor listener that stops underperforming/finished jobs via the launcher.

    Register the jobs to govern (at construction or with :meth:`register`); as
    metrics arrive the listener consults its :class:`EarlyStopper` and cancels the
    matching job on a stop decision, invoking ``on_stop`` for observability.
    """

    def __init__(
        self,
        launcher: TrainingLauncher,
        stopper: EarlyStopper,
        *,
        jobs: Sequence[TrainingJob] | None = None,
        on_stop: Callable[[StopDecision], None] | None = None,
    ) -> None:
        self._launcher = launcher
        self._stopper = stopper
        self._on_stop = on_stop
        self._jobs: dict[str, TrainingJob] = {}
        self._stopped: set[str] = set()
        for job in jobs or []:
            self.register(job)

    def register(self, job: TrainingJob) -> None:
        """Make ``job`` eligible for early stopping."""
        self._jobs[job.experiment_id] = job

    def register_all(self, jobs: Sequence[TrainingJob]) -> None:
        for job in jobs:
            self.register(job)

    @property
    def decisions(self) -> dict[str, StopDecision]:
        """Mapping of experiment id to the decision that stopped it."""
        return dict(self._stopper._decisions)  # noqa: SLF001 - read-only view

    def on_metric(self, event: MetricEvent) -> None:
        if event.experiment_id in self._stopped:
            return
        observation = MetricValue(name=event.name, value=event.value, step=event.step)
        decision = self._stopper.update(event.experiment_id, observation)
        if decision is None:
            return
        self._stopped.add(event.experiment_id)
        job = self._jobs.get(event.experiment_id)
        if job is not None:
            self._launcher.cancel(job)
        if self._on_stop is not None:
            self._on_stop(decision)

    def on_status(
        self, experiment_id: str, status: ExperimentStatus, result
    ) -> None:
        # Nothing to do on natural completion; cleanup is implicit.
        return None

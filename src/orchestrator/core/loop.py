"""Adaptive hyperparameter tuning loop (pipeline stage 5: *adjust hyperparameters*).

:class:`TuningLoop` is the conductor that turns the individual pieces -- strategy,
launcher, monitor, early stopping -- into a closed feedback loop:

1. Ask the strategy for the next batch of experiments, **passing the full history
   of finished runs** so adaptive strategies can adjust their proposals.
2. Launch the batch on the launcher (up to ``max_concurrency`` at a time).
3. Stream their metrics through a :class:`~orchestrator.core.monitor.MetricMonitor`,
   with an :class:`~orchestrator.core.stopping.EarlyStoppingListener` pruning runs
   that meet their target, plateau, or diverge.
4. Apply each run's result back onto its experiment and append it to the history.
5. Stop when the experiment budget is exhausted, the strategy runs dry, or the
   objective's target metric value has been reached.

The loop is strategy-agnostic: random and grid search ignore the history, while a
future Bayesian strategy reads it -- the adaptiveness lives in the feedback the
loop provides, not in the loop itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from orchestrator.core.launcher import TrainingJob, TrainingLauncher, apply_result
from orchestrator.core.local_backend import LocalLauncher
from orchestrator.core.models import Experiment, Goal, Objective
from orchestrator.core.monitor import MetricHistory, MetricListener, MetricMonitor
from orchestrator.core.stopping import EarlyStopper, EarlyStoppingListener, StopDecision
from orchestrator.core.strategy import ExperimentStrategy, get_strategy


@dataclass
class TuningResult:
    """The outcome of a tuning run."""

    objective_id: str
    primary_metric: str
    goal: Goal
    experiments: list[Experiment] = field(default_factory=list)
    best_experiment: Experiment | None = None
    best_value: float | None = None
    target_reached: bool = False
    rounds: int = 0
    stop_decisions: list[StopDecision] = field(default_factory=list)

    @property
    def completed(self) -> list[Experiment]:
        """Experiments that produced at least one value for the primary metric."""
        return [e for e in self.experiments if e.best_metric(self.primary_metric, self.goal) is not None]


class TuningLoop:
    """Run the adaptive generate/launch/monitor/adjust loop for an objective.

    Parameters
    ----------
    objective:
        The validated objective driving the search (provides the search space,
        budgets, primary metric, goal, and optional target value).
    launcher:
        Backend used to run each experiment. For real concurrency, configure it
        with ``max_workers >= objective.max_concurrency``.
    strategy:
        The experiment-generation strategy. Defaults to random search.
    stopper:
        Early-stopping policies. Defaults to one derived from the objective
        (target threshold + divergence guard).
    monitor_interval:
        Polling interval for the metric monitor.
    listeners:
        Extra monitor listeners (e.g. dashboards) attached for the whole run.
    on_round:
        Optional callback invoked as ``on_round(round_index, experiments)`` after
        each round completes.
    """

    def __init__(
        self,
        objective: Objective,
        launcher: TrainingLauncher,
        *,
        strategy: ExperimentStrategy | None = None,
        stopper: EarlyStopper | None = None,
        monitor_interval: float = 0.05,
        listeners: list[MetricListener] | None = None,
        on_round: Callable[[int, list[Experiment]], None] | None = None,
    ) -> None:
        self.objective = objective
        self.launcher = launcher
        self.strategy = strategy or get_strategy("random")
        self.stopper = stopper if stopper is not None else EarlyStopper.from_objective(objective)
        self.monitor_interval = monitor_interval
        self._extra_listeners = list(listeners or [])
        self._on_round = on_round

    def run(self, *, round_timeout: float | None = None) -> TuningResult:
        """Execute the loop and return a :class:`TuningResult`."""
        obj = self.objective
        experiments: list[Experiment] = []
        stop_decisions: list[StopDecision] = []

        monitor = MetricMonitor(self.launcher, interval=self.monitor_interval)
        history = MetricHistory()
        monitor.add_listener(history)
        for listener in self._extra_listeners:
            monitor.add_listener(listener)
        stop_listener = EarlyStoppingListener(
            self.launcher, self.stopper, on_stop=stop_decisions.append
        )
        monitor.add_listener(stop_listener)

        rounds = 0
        target_reached = False

        while len(experiments) < obj.max_experiments:
            remaining = obj.max_experiments - len(experiments)
            batch_size = min(obj.max_concurrency, remaining)

            proposed = self.strategy.propose(obj, count=batch_size, history=experiments)
            if not proposed:
                break  # strategy exhausted (e.g. grid fully enumerated)

            rounds += 1
            jobs: dict[str, TrainingJob] = {}
            for experiment in proposed:
                job = self.launcher.launch(experiment)
                jobs[experiment.id] = job
                stop_listener.register(job)
                monitor.track(job)

            # Drain this batch's metrics until every job is terminal.
            for _ in monitor.stream(timeout=round_timeout):
                pass

            for experiment in proposed:
                result = self.launcher.result(jobs[experiment.id])
                apply_result(experiment, result)
                experiments.append(experiment)

            if self._on_round is not None:
                self._on_round(rounds, proposed)

            if self._target_reached(experiments):
                target_reached = True
                break

        best_experiment, best_value = self._best(experiments)
        return TuningResult(
            objective_id=obj.id,
            primary_metric=obj.primary_metric,
            goal=obj.goal,
            experiments=experiments,
            best_experiment=best_experiment,
            best_value=best_value,
            target_reached=target_reached,
            rounds=rounds,
            stop_decisions=stop_decisions,
        )

    # -- helpers -----------------------------------------------------------

    def _best(self, experiments: list[Experiment]) -> tuple[Experiment | None, float | None]:
        metric, goal = self.objective.primary_metric, self.objective.goal
        scored = [
            (e, e.best_metric(metric, goal))
            for e in experiments
            if e.best_metric(metric, goal) is not None
        ]
        if not scored:
            return None, None
        chooser = max if goal is Goal.MAXIMIZE else min
        best_experiment, best_value = chooser(scored, key=lambda pair: pair[1])
        return best_experiment, best_value

    def _target_reached(self, experiments: list[Experiment]) -> bool:
        target = self.objective.target_metric_value
        if target is None:
            return False
        _, best_value = self._best(experiments)
        if best_value is None:
            return False
        if self.objective.goal is Goal.MAXIMIZE:
            return best_value >= target
        return best_value <= target


def run_local(
    objective: Objective,
    train_fn: Callable[..., object],
    *,
    strategy: ExperimentStrategy | str | None = None,
    stopper: EarlyStopper | None = None,
    on_round: Callable[[int, list[Experiment]], None] | None = None,
) -> TuningResult:
    """Convenience: run a tuning loop on a local backend sized to the objective.

    Builds a :class:`~orchestrator.core.local_backend.LocalLauncher` with
    ``max_workers`` set to the objective's ``max_concurrency`` and runs the loop
    to completion. ``strategy`` may be a strategy instance or a registered name.
    """
    if isinstance(strategy, str):
        strategy = get_strategy(strategy)
    with LocalLauncher(train_fn, max_workers=objective.max_concurrency) as launcher:
        loop = TuningLoop(
            objective,
            launcher,
            strategy=strategy,
            stopper=stopper,
            on_round=on_round,
        )
        return loop.run()

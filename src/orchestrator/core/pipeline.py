"""End-to-end orchestration pipeline.

This is the capstone that wires the individual stages into the full lifecycle
described in the project README:

    intake -> generate -> launch -> monitor -> adjust -> compare -> report -> deploy

:class:`Orchestrator` holds the policy choices (strategy, early-stopping, extra
monitor listeners, deployment target) and runs them against an objective:

1. **Intake & validate** the objective (:func:`~orchestrator.core.intake.intake_objective`).
2. **Run the adaptive loop** (:class:`~orchestrator.core.loop.TuningLoop`), which
   itself covers generate/launch/monitor/adjust.
3. **Compare** the finished runs into a leaderboard
   (:func:`~orchestrator.core.ranking.rank_result`).
4. **Report** a human-readable summary (:func:`~orchestrator.core.report.build_report`).
5. **Deploy** the best model to a target (:func:`~orchestrator.core.deployment.deploy_best`).

The result of every stage is collected into a single :class:`PipelineResult`.
:func:`run_pipeline` is a one-call convenience for the common local case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from orchestrator.core.deployment import (
    Deployment,
    DeploymentTarget,
    LocalDeploymentTarget,
    deploy_best,
)
from orchestrator.core.intake import intake_objective
from orchestrator.core.launcher import TrainingLauncher
from orchestrator.core.local_backend import LocalLauncher
from orchestrator.core.loop import TuningLoop, TuningResult
from orchestrator.core.models import Experiment, Objective
from orchestrator.core.monitor import MetricListener
from orchestrator.core.ranking import Leaderboard, MetricKey, rank_result
from orchestrator.core.report import Report, build_report
from orchestrator.core.stopping import EarlyStopper
from orchestrator.core.strategy import ExperimentStrategy, get_strategy


@dataclass
class PipelineResult:
    """Everything produced by an end-to-end run, stage by stage."""

    objective: Objective
    tuning: TuningResult
    leaderboard: Leaderboard
    report: Report
    deployment: Deployment | None = None

    @property
    def best_experiment(self) -> Experiment | None:
        return self.tuning.best_experiment

    @property
    def best_value(self) -> float | None:
        return self.tuning.best_value

    def to_markdown(self) -> str:
        """Render the run's report as Markdown."""
        return self.report.to_markdown()


class Orchestrator:
    """Runs the full experimentation lifecycle for an objective.

    Parameters
    ----------
    strategy:
        Experiment-generation strategy (instance or registered name). Defaults to
        random search.
    stopper:
        Early-stopping policies. Defaults to one derived from each objective.
    deploy:
        Whether to deploy the best model after comparison.
    deploy_target:
        Target to deploy to (instance or registered name). Defaults to a
        :class:`~orchestrator.core.deployment.LocalDeploymentTarget`.
    listeners:
        Extra monitor listeners (e.g. MLflow/W&B) attached for live logging.
    secondary:
        Secondary ``(metric, goal)`` keys for ranking/selection tie-breaks.
    report_top_n:
        Leaderboard rows to include in the report.
    require_target_for_deploy:
        Only deploy if the objective's target metric value was reached.
    min_score:
        Only deploy if the best score meets this threshold.
    """

    def __init__(
        self,
        *,
        strategy: ExperimentStrategy | str | None = None,
        stopper: EarlyStopper | None = None,
        deploy: bool = True,
        deploy_target: DeploymentTarget | str | None = None,
        listeners: Sequence[MetricListener] | None = None,
        secondary: Sequence[MetricKey] = (),
        monitor_interval: float = 0.05,
        report_top_n: int = 10,
        require_target_for_deploy: bool = False,
        min_score: float | None = None,
    ) -> None:
        self._strategy = get_strategy(strategy) if isinstance(strategy, str) else strategy
        self._stopper = stopper
        self._deploy = deploy
        self._deploy_target = deploy_target
        self._listeners = list(listeners or [])
        self._secondary = tuple(secondary)
        self._monitor_interval = monitor_interval
        self._report_top_n = report_top_n
        self._require_target = require_target_for_deploy
        self._min_score = min_score

    def run(
        self,
        objective: Objective | str | dict[str, Any],
        train_fn: Callable[..., object] | None = None,
        *,
        launcher: TrainingLauncher | None = None,
    ) -> PipelineResult:
        """Execute the full pipeline and return a :class:`PipelineResult`.

        Provide either a ``launcher`` or a ``train_fn`` (which is wrapped in a
        local backend sized to the objective's concurrency).
        """
        # 1. Intake & validate.
        obj = intake_objective(objective)

        # 2. Generate / launch / monitor / adjust via the tuning loop.
        own_launcher = False
        if launcher is None:
            if train_fn is None:
                raise ValueError("provide either a launcher or a train_fn")
            launcher = LocalLauncher(train_fn, max_workers=obj.max_concurrency)
            own_launcher = True

        stopper = self._stopper or EarlyStopper.from_objective(obj)
        try:
            loop = TuningLoop(
                obj,
                launcher,
                strategy=self._strategy,
                stopper=stopper,
                monitor_interval=self._monitor_interval,
                listeners=self._listeners,
            )
            tuning = loop.run()
        finally:
            if own_launcher:
                close = getattr(launcher, "close", None)
                if callable(close):
                    close()

        # 3. Compare.
        leaderboard = rank_result(tuning, secondary=self._secondary)

        # 4. Report.
        report = build_report(
            tuning, objective=obj, leaderboard=leaderboard, top_n=self._report_top_n
        )

        # 5. Deploy best model.
        deployment: Deployment | None = None
        if self._deploy:
            target = self._deploy_target if self._deploy_target is not None else LocalDeploymentTarget()
            deployment = deploy_best(
                tuning,
                target,
                secondary=self._secondary,
                min_score=self._min_score,
                require_target_reached=self._require_target,
            )

        return PipelineResult(
            objective=obj,
            tuning=tuning,
            leaderboard=leaderboard,
            report=report,
            deployment=deployment,
        )


def run_pipeline(
    objective: Objective | str | dict[str, Any],
    train_fn: Callable[..., object],
    *,
    strategy: ExperimentStrategy | str | None = None,
    stopper: EarlyStopper | None = None,
    deploy: bool = True,
    deploy_target: DeploymentTarget | str | None = None,
    listeners: Sequence[MetricListener] | None = None,
    secondary: Sequence[MetricKey] = (),
) -> PipelineResult:
    """One-call end-to-end run against a local backend."""
    orchestrator = Orchestrator(
        strategy=strategy,
        stopper=stopper,
        deploy=deploy,
        deploy_target=deploy_target,
        listeners=listeners,
        secondary=secondary,
    )
    return orchestrator.run(objective, train_fn)

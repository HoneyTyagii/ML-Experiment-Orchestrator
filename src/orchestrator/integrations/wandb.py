"""Weights & Biases integration for metric logging.

Logs the orchestrator's experiments to Weights & Biases: one W&B run per
:class:`~orchestrator.core.models.Experiment`, with its hyperparameters as the
run ``config``, recorded :class:`~orchestrator.core.models.MetricValue`
observations logged as stepped history, and the final status recorded on the run
summary before the run is finished.

W&B is an *optional* dependency. Importing this module never imports ``wandb``;
the ``wandb.init`` entry point is resolved lazily on first use, and a clear
:class:`WandbError` is raised if the extra isn't installed::

    pip install ml-experiment-orchestrator[wandb]

Runs are created through an injectable ``init`` callable (defaulting to
``wandb.init`` with ``reinit=True`` so concurrent experiments each get their own
run). The callable returns a run object exposing ``log(data, step=...)``,
a mutable ``summary`` mapping, and ``finish(exit_code=...)`` -- which makes the
tracker straightforward to test with a fake.

Usage
-----
.. code-block:: python

    tracker = WandbTracker.for_objective(objective, entity="my-team")
    tracker.log_result(result)                       # after a run
    monitor.add_listener(WandbListener(tracker, experiments))  # or live
"""

from __future__ import annotations

from typing import Any, Callable

from orchestrator.core.models import Experiment, ExperimentStatus, MetricValue
from orchestrator.core.monitor import MetricEvent, MetricListener

#: Orchestrator status -> W&B run exit code (0 = success, non-zero = failure).
_EXIT_CODE = {
    ExperimentStatus.COMPLETED: 0,
    ExperimentStatus.RUNNING: 0,
    ExperimentStatus.PENDING: 0,
    ExperimentStatus.FAILED: 1,
    ExperimentStatus.CANCELLED: 1,
}


class WandbError(Exception):
    """Raised when W&B is unavailable or a logging operation is misused."""


def _default_init() -> Callable[..., Any]:
    """Return ``wandb.init`` (lazy import)."""
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - exercised only without wandb
        raise WandbError(
            "wandb is not installed; install the extra with "
            "`pip install ml-experiment-orchestrator[wandb]`"
        ) from exc
    return wandb.init


class WandbTracker:
    """Logs experiments and their metrics to Weights & Biases.

    Parameters
    ----------
    project:
        W&B project to log runs under.
    entity:
        Optional W&B entity (team/user).
    group:
        Optional run group. When omitted, each experiment's run is grouped by its
        objective id so all runs of one objective cluster together.
    init:
        Callable used to create runs (defaults to a lazily imported
        ``wandb.init``). Injectable for testing.
    """

    def __init__(
        self,
        project: str,
        *,
        entity: str | None = None,
        group: str | None = None,
        init: Callable[..., Any] | None = None,
    ) -> None:
        self.project = project
        self.entity = entity
        self.group = group
        self._init = init
        self._runs: dict[str, Any] = {}  # orchestrator experiment id -> wandb run
        self._last_step: dict[str, int] = {}

    @classmethod
    def for_objective(
        cls,
        objective,
        *,
        entity: str | None = None,
        group: str | None = None,
        init: Callable[..., Any] | None = None,
    ) -> WandbTracker:
        """Build a tracker whose W&B project is named after the objective."""
        return cls(objective.name, entity=entity, group=group, init=init)

    @property
    def init_fn(self) -> Callable[..., Any]:
        """The run-creation callable (resolved lazily to ``wandb.init``)."""
        if self._init is None:
            self._init = _default_init()
        return self._init

    def run_for(self, experiment_id: str) -> Any | None:
        """Return the W&B run mapped to an orchestrator experiment, if any."""
        return self._runs.get(experiment_id)

    # -- logging primitives ------------------------------------------------

    def start_run(self, experiment: Experiment) -> Any:
        """Create a W&B run for ``experiment`` with its hyperparameters as config."""
        if experiment.id in self._runs:
            return self._runs[experiment.id]
        run = self.init_fn(
            project=self.project,
            entity=self.entity,
            name=experiment.name or experiment.id,
            group=self.group or experiment.objective_id,
            config=dict(experiment.hyperparameters),
            tags=[experiment.objective_id],
            reinit=True,
        )
        self._runs[experiment.id] = run
        self._last_step[experiment.id] = -1
        return run

    def log_metric(self, experiment_id: str, metric: MetricValue) -> None:
        """Log a single metric observation to the experiment's run."""
        run = self._runs.get(experiment_id)
        if run is None:
            raise WandbError(f"no active W&B run for experiment {experiment_id}")
        # W&B requires non-decreasing steps; fall back to commit-without-step
        # if an out-of-order (e.g. final, step-0) observation arrives late.
        last = self._last_step.get(experiment_id, -1)
        if metric.step >= last:
            run.log({metric.name: metric.value}, step=metric.step)
            self._last_step[experiment_id] = metric.step
        else:
            run.log({metric.name: metric.value})

    def finish_run(
        self, experiment: Experiment, *, status: ExperimentStatus | None = None
    ) -> None:
        """Record the final status on the run summary and finish the run."""
        run = self._runs.get(experiment.id)
        if run is None:
            return
        final = status or experiment.status
        run.summary["final_status"] = final.value
        run.finish(exit_code=_EXIT_CODE.get(final, 0))

    # -- high-level helpers ------------------------------------------------

    def log_experiment(self, experiment: Experiment) -> Any:
        """Log a finished experiment in one shot (config, all metrics, status)."""
        run = self.start_run(experiment)
        for metric in experiment.metrics:
            self.log_metric(experiment.id, metric)
        self.finish_run(experiment)
        return run

    def log_result(self, result, *, mark_best: bool = True) -> None:
        """Log every experiment in a :class:`~orchestrator.core.loop.TuningResult`.

        When ``mark_best`` is set, the winning run's summary is flagged ``best``.
        """
        best_id = result.best_experiment.id if result.best_experiment is not None else None
        for experiment in result.experiments:
            run = self.start_run(experiment)
            for metric in experiment.metrics:
                self.log_metric(experiment.id, metric)
            if mark_best and experiment.id == best_id:
                run.summary["best"] = True
            self.finish_run(experiment)


class WandbListener(MetricListener):
    """Stream live metrics into W&B as a monitor listener.

    Starts a W&B run lazily on an experiment's first metric (or completion), logs
    each observation, and finishes the run on completion.
    """

    def __init__(self, tracker: WandbTracker, experiments) -> None:
        self._tracker = tracker
        self._experiments: dict[str, Experiment] = {e.id: e for e in experiments}

    def register(self, experiment: Experiment) -> None:
        """Make ``experiment`` known to the listener so its run can be created."""
        self._experiments[experiment.id] = experiment

    def _ensure_run(self, experiment_id: str) -> Experiment | None:
        experiment = self._experiments.get(experiment_id)
        if experiment is None:
            return None
        if self._tracker.run_for(experiment_id) is None:
            self._tracker.start_run(experiment)
        return experiment

    def on_metric(self, event: MetricEvent) -> None:
        if self._ensure_run(event.experiment_id) is None:
            return
        self._tracker.log_metric(
            event.experiment_id,
            MetricValue(name=event.name, value=event.value, step=event.step, timestamp=event.timestamp),
        )

    def on_status(self, experiment_id: str, status: ExperimentStatus, result) -> None:
        experiment = self._ensure_run(experiment_id)
        if experiment is None:
            return
        self._tracker.finish_run(experiment, status=status)


def track_result(
    result,
    *,
    objective=None,
    project: str | None = None,
    entity: str | None = None,
    init: Callable[..., Any] | None = None,
    mark_best: bool = True,
) -> WandbTracker:
    """Convenience: log a completed tuning result to W&B and return the tracker."""
    name = project or (objective.name if objective is not None else result.objective_id)
    tracker = WandbTracker(name, entity=entity, init=init)
    tracker.log_result(result, mark_best=mark_best)
    return tracker

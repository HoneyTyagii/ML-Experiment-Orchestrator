"""MLflow integration for experiment tracking.

Logs the orchestrator's experiments to an MLflow tracking server: one MLflow run
per :class:`~orchestrator.core.models.Experiment`, with its hyperparameters as
params, recorded :class:`~orchestrator.core.models.MetricValue` observations as
stepped metrics, and the final status as the run's terminal state.

MLflow is an *optional* dependency. Importing this module never imports MLflow;
the real client is loaded lazily on first use, and a clear :class:`MlflowError`
is raised if the extra isn't installed::

    pip install ml-experiment-orchestrator[mlflow]

The tracker talks to the low-level ``MlflowClient`` (explicit run ids) rather
than the global fluent API, so concurrent experiments log without clobbering a
shared "active run". A client may be injected for testing or custom backends.

Usage
-----
.. code-block:: python

    tracker = MlflowTracker.for_objective(objective, tracking_uri="http://localhost:5000")

    # after a run:
    tracker.log_result(result)

    # or live, wired into the monitor:
    monitor.add_listener(MlflowListener(tracker, experiments))
"""

from __future__ import annotations

from typing import Any

from orchestrator.core.models import Experiment, ExperimentStatus, MetricValue
from orchestrator.core.monitor import MetricEvent, MetricListener

#: Map orchestrator statuses onto MLflow's RunStatus vocabulary.
_MLFLOW_STATUS = {
    ExperimentStatus.PENDING: "SCHEDULED",
    ExperimentStatus.RUNNING: "RUNNING",
    ExperimentStatus.COMPLETED: "FINISHED",
    ExperimentStatus.FAILED: "FAILED",
    ExperimentStatus.CANCELLED: "KILLED",
}


class MlflowError(Exception):
    """Raised when MLflow is unavailable or a tracking operation is misused."""


def _default_client(tracking_uri: str | None) -> Any:
    """Construct a real ``mlflow.tracking.MlflowClient`` (lazy import)."""
    try:
        from mlflow.tracking import MlflowClient
    except ImportError as exc:  # pragma: no cover - exercised only without mlflow
        raise MlflowError(
            "mlflow is not installed; install the extra with "
            "`pip install ml-experiment-orchestrator[mlflow]`"
        ) from exc
    return MlflowClient(tracking_uri=tracking_uri)


class MlflowTracker:
    """Logs experiments and their metrics to an MLflow experiment.

    Parameters
    ----------
    experiment_name:
        Name of the MLflow experiment to log under (created if absent).
    tracking_uri:
        MLflow tracking server URI. Ignored when ``client`` is supplied.
    client:
        An object implementing the subset of ``MlflowClient`` used here. When
        ``None`` (default), a real client is created lazily on first use.
    """

    def __init__(
        self,
        experiment_name: str,
        *,
        tracking_uri: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self._tracking_uri = tracking_uri
        self._client = client
        self._experiment_id: str | None = None
        self._runs: dict[str, str] = {}  # orchestrator experiment id -> mlflow run id

    @classmethod
    def for_objective(
        cls, objective, *, tracking_uri: str | None = None, client: Any | None = None
    ) -> MlflowTracker:
        """Build a tracker whose MLflow experiment is named after the objective."""
        return cls(objective.name, tracking_uri=tracking_uri, client=client)

    # -- client / experiment plumbing -------------------------------------

    @property
    def client(self) -> Any:
        """The underlying MLflow client (created lazily on first access)."""
        if self._client is None:
            self._client = _default_client(self._tracking_uri)
        return self._client

    def _ensure_experiment(self) -> str:
        if self._experiment_id is None:
            existing = self.client.get_experiment_by_name(self.experiment_name)
            if existing is None:
                self._experiment_id = self.client.create_experiment(self.experiment_name)
            else:
                self._experiment_id = existing.experiment_id
        return self._experiment_id

    def run_id_for(self, experiment_id: str) -> str | None:
        """Return the MLflow run id mapped to an orchestrator experiment, if any."""
        return self._runs.get(experiment_id)

    # -- logging primitives ------------------------------------------------

    def start_run(self, experiment: Experiment) -> str:
        """Create an MLflow run for ``experiment`` and log its hyperparameters."""
        if experiment.id in self._runs:
            return self._runs[experiment.id]
        experiment_id = self._ensure_experiment()
        tags = {
            "objective_id": experiment.objective_id,
            "experiment_name": experiment.name,
            "mlflow.runName": experiment.name or experiment.id,
        }
        run = self.client.create_run(experiment_id, tags=tags)
        run_id = run.info.run_id
        self._runs[experiment.id] = run_id
        for key, value in experiment.hyperparameters.items():
            self.client.log_param(run_id, key, value)
        return run_id

    def log_metric(self, experiment_id: str, metric: MetricValue) -> None:
        """Log a single metric observation to the experiment's run."""
        run_id = self._runs.get(experiment_id)
        if run_id is None:
            raise MlflowError(f"no active MLflow run for experiment {experiment_id}")
        timestamp_ms = int(metric.timestamp.timestamp() * 1000)
        self.client.log_metric(
            run_id, metric.name, metric.value, timestamp=timestamp_ms, step=metric.step
        )

    def finish_run(
        self, experiment: Experiment, *, status: ExperimentStatus | None = None
    ) -> None:
        """Terminate the experiment's run, recording its final status."""
        run_id = self._runs.get(experiment.id)
        if run_id is None:
            return
        final = status or experiment.status
        self.client.set_tag(run_id, "final_status", final.value)
        self.client.set_terminated(run_id, _MLFLOW_STATUS.get(final, "FINISHED"))

    # -- high-level helpers ------------------------------------------------

    def log_experiment(self, experiment: Experiment) -> str:
        """Log a finished experiment in one shot (params, all metrics, status)."""
        run_id = self.start_run(experiment)
        for metric in experiment.metrics:
            self.log_metric(experiment.id, metric)
        self.finish_run(experiment)
        return run_id

    def log_result(self, result, *, mark_best: bool = True) -> None:
        """Log every experiment in a :class:`~orchestrator.core.loop.TuningResult`.

        When ``mark_best`` is set, the winning run is tagged ``best=true``.
        """
        for experiment in result.experiments:
            self.log_experiment(experiment)
        if mark_best and result.best_experiment is not None:
            run_id = self._runs.get(result.best_experiment.id)
            if run_id is not None:
                self.client.set_tag(run_id, "best", "true")


class MlflowListener(MetricListener):
    """Stream live metrics into MLflow as a monitor listener.

    Starts an MLflow run lazily the first time an experiment reports a metric (or
    finishes), logs each observation, and terminates the run on completion.
    """

    def __init__(self, tracker: MlflowTracker, experiments) -> None:
        self._tracker = tracker
        self._experiments: dict[str, Experiment] = {e.id: e for e in experiments}

    def register(self, experiment: Experiment) -> None:
        """Make ``experiment`` known to the listener so its run can be created."""
        self._experiments[experiment.id] = experiment

    def _ensure_run(self, experiment_id: str) -> Experiment | None:
        experiment = self._experiments.get(experiment_id)
        if experiment is None:
            return None
        if self._tracker.run_id_for(experiment_id) is None:
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
    experiment_name: str | None = None,
    tracking_uri: str | None = None,
    client: Any | None = None,
    mark_best: bool = True,
) -> MlflowTracker:
    """Convenience: log a completed tuning result to MLflow and return the tracker."""
    name = experiment_name or (objective.name if objective is not None else result.objective_id)
    tracker = MlflowTracker(name, tracking_uri=tracking_uri, client=client)
    tracker.log_result(result, mark_best=mark_best)
    return tracker

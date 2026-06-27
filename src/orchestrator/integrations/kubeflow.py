"""Kubeflow Pipelines orchestration backend.

A :class:`~orchestrator.core.launcher.TrainingLauncher` that runs each experiment
as a Kubeflow pipeline run. It is the natural fit for the launcher abstraction's
asynchronous lifecycle: :meth:`launch` submits a run and returns immediately,
:meth:`poll` maps the pipeline run's state to an
:class:`~orchestrator.core.models.ExperimentStatus`, :meth:`result` collects the
run's metrics once it finishes, and :meth:`cancel` terminates it.

Kubeflow (``kfp``) is an *optional* dependency. Importing this module never
imports ``kfp``; the real ``kfp.Client`` is created lazily on first use, and a
clear :class:`KubeflowError` is raised if the extra isn't installed::

    pip install ml-experiment-orchestrator[kubeflow]

Every interaction with the cluster goes through small, overridable hooks
(``submit_fn`` / ``status_fn`` / ``metrics_fn`` / ``terminate_fn``) with sensible
defaults for ``kfp``. This keeps the backend adaptable across kfp versions and
trivially testable with a fake client.

The launcher registers itself as ``"kubeflow"`` once this module is imported, so
``get_launcher("kubeflow", pipeline=..., ...)`` resolves it.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Callable

from orchestrator.core.launcher import (
    TERMINAL_STATUSES,
    LauncherError,
    TrainingJob,
    TrainingLauncher,
    TrainingResult,
    register_launcher,
)
from orchestrator.core.models import Experiment, ExperimentStatus, MetricValue

#: Lower-cased Kubeflow/Argo run states mapped onto orchestrator statuses.
_STATE_MAP = {
    "": ExperimentStatus.PENDING,
    "pending": ExperimentStatus.PENDING,
    "scheduled": ExperimentStatus.PENDING,
    "queued": ExperimentStatus.PENDING,
    "running": ExperimentStatus.RUNNING,
    "succeeded": ExperimentStatus.COMPLETED,
    "success": ExperimentStatus.COMPLETED,
    "completed": ExperimentStatus.COMPLETED,
    "failed": ExperimentStatus.FAILED,
    "error": ExperimentStatus.FAILED,
    "canceled": ExperimentStatus.CANCELLED,
    "cancelled": ExperimentStatus.CANCELLED,
    "terminated": ExperimentStatus.CANCELLED,
    "skipped": ExperimentStatus.CANCELLED,
}


class KubeflowError(LauncherError):
    """Raised when Kubeflow is unavailable or a pipeline operation is misused."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _default_client(host: str | None, namespace: str | None) -> Any:
    """Construct a real ``kfp.Client`` (lazy import)."""
    try:
        import kfp
    except ImportError as exc:  # pragma: no cover - exercised only without kfp
        raise KubeflowError(
            "kfp is not installed; install the extra with "
            "`pip install ml-experiment-orchestrator[kubeflow]`"
        ) from exc
    return kfp.Client(host=host, namespace=namespace)


def _map_state(raw: Any) -> ExperimentStatus:
    """Map a raw pipeline run state onto an :class:`ExperimentStatus`."""
    return _STATE_MAP.get(str(raw or "").strip().lower(), ExperimentStatus.RUNNING)


def _default_submit(
    client: Any,
    pipeline: Any,
    *,
    arguments: Mapping[str, Any],
    run_name: str,
    experiment_name: str | None,
) -> str:
    """Submit a pipeline run via ``kfp.Client`` and return its run id."""
    if pipeline is None:
        raise KubeflowError("no pipeline configured; pass pipeline=... to KubeflowLauncher")
    result = client.create_run_from_pipeline_package(
        pipeline,
        arguments=dict(arguments),
        run_name=run_name,
        experiment_name=experiment_name,
    )
    return getattr(result, "run_id", getattr(result, "id", result))


def _default_status(client: Any, run_id: str) -> Any:
    """Read a run's state across common ``kfp`` client/response shapes."""
    run = client.get_run(run_id)
    state = getattr(run, "state", None)
    if state is None:
        inner = getattr(run, "run", None)
        state = getattr(inner, "state", None) if inner is not None else None
    if state is None:
        state = getattr(run, "status", None)
    return state


def _default_terminate(client: Any, run_id: str) -> None:
    """Terminate a run across common ``kfp`` client shapes."""
    terminate = getattr(client, "terminate_run", None)
    if callable(terminate):
        terminate(run_id)
        return
    runs = getattr(client, "runs", None)
    if runs is not None and hasattr(runs, "terminate_run"):
        runs.terminate_run(run_id)
        return
    raise KubeflowError("kfp client exposes no terminate_run method")


@register_launcher
class KubeflowLauncher(TrainingLauncher):
    """Run experiments as Kubeflow pipeline runs.

    Parameters
    ----------
    pipeline:
        The pipeline to run for each experiment -- whatever ``submit_fn`` accepts
        (by default a compiled pipeline package path or pipeline func understood
        by ``kfp.Client.create_run_from_pipeline_package``).
    client:
        A ``kfp.Client``-like object. Created lazily from ``host``/``namespace``
        when omitted.
    experiment_name:
        Kubeflow experiment to group runs under.
    arguments_fn:
        Maps an :class:`Experiment` to pipeline arguments. Defaults to the
        experiment's hyperparameters.
    metrics_fn:
        Reads ``{metric: value}`` for a finished run. Defaults to none (the run's
        status is still tracked, just without scalar metrics).
    submit_fn / status_fn / terminate_fn:
        Overridable hooks for cluster interaction; sensible ``kfp`` defaults are
        provided.
    """

    name = "kubeflow"

    def __init__(
        self,
        pipeline: Any = None,
        *,
        client: Any | None = None,
        host: str | None = None,
        namespace: str | None = None,
        experiment_name: str | None = None,
        arguments_fn: Callable[[Experiment], Mapping[str, Any]] | None = None,
        metrics_fn: Callable[[Any, str], Mapping[str, float]] | None = None,
        submit_fn: Callable[..., str] | None = None,
        status_fn: Callable[[Any, str], Any] | None = None,
        terminate_fn: Callable[[Any, str], None] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self._client = client
        self._host = host
        self._namespace = namespace
        self.experiment_name = experiment_name
        self._arguments_fn = arguments_fn or (lambda exp: dict(exp.hyperparameters))
        self._metrics_fn = metrics_fn
        self._submit_fn = submit_fn or _default_submit
        self._status_fn = status_fn or _default_status
        self._terminate_fn = terminate_fn or _default_terminate
        self._run_ids: dict[str, str] = {}  # job id -> kfp run id
        self._raw_state: dict[str, Any] = {}  # job id -> last raw state

    @property
    def client(self) -> Any:
        """The underlying kfp client (created lazily on first access)."""
        if self._client is None:
            self._client = _default_client(self._host, self._namespace)
        return self._client

    def run_id_for(self, job: TrainingJob) -> str | None:
        """Return the Kubeflow run id backing ``job``, if known."""
        return self._run_ids.get(job.id)

    # -- lifecycle ---------------------------------------------------------

    def launch(self, experiment: Experiment) -> TrainingJob:
        arguments = self._arguments_fn(experiment)
        run_id = self._submit_fn(
            self.client,
            self.pipeline,
            arguments=arguments,
            run_name=experiment.name or experiment.id,
            experiment_name=self.experiment_name,
        )
        job = TrainingJob(
            experiment_id=experiment.id,
            backend=self.name,
            status=ExperimentStatus.RUNNING,
            handle=run_id,
            started_at=_utcnow(),
        )
        self._run_ids[job.id] = run_id
        return job

    def poll(self, job: TrainingJob) -> ExperimentStatus:
        if job.is_terminal:
            return job.status
        run_id = self._require_run(job)
        raw = self._status_fn(self.client, run_id)
        self._raw_state[job.id] = raw
        status = _map_state(raw)
        job.status = status
        if status in TERMINAL_STATUSES:
            job.finished_at = _utcnow()
        return status

    def result(self, job: TrainingJob) -> TrainingResult:
        if self.poll(job) not in TERMINAL_STATUSES:
            raise KubeflowError(f"pipeline run for job {job.id} is not yet terminal")
        run_id = self._require_run(job)
        metrics: list[MetricValue] = []
        if self._metrics_fn is not None:
            raw_metrics = self._metrics_fn(self.client, run_id) or {}
            metrics = [MetricValue(name=str(k), value=float(v)) for k, v in raw_metrics.items()]
        error = None
        if job.status is ExperimentStatus.FAILED:
            error = f"kubeflow run {run_id} ended in state {self._raw_state.get(job.id)!r}"
        return TrainingResult(
            experiment_id=job.experiment_id,
            status=job.status,
            metrics=metrics,
            error=error,
        )

    def cancel(self, job: TrainingJob) -> None:
        run_id = self._run_ids.get(job.id)
        if run_id is None:
            return
        self._terminate_fn(self.client, run_id)
        if not job.is_terminal:
            job.status = ExperimentStatus.CANCELLED
            job.finished_at = _utcnow()

    # -- internals ---------------------------------------------------------

    def _require_run(self, job: TrainingJob) -> str:
        run_id = self._run_ids.get(job.id)
        if run_id is None:
            raise KubeflowError(f"unknown job {job.id}; was it launched by this backend?")
        return run_id

"""Training job launchers (pipeline stage 3: *launch training*).

A *launcher* takes a generated :class:`~orchestrator.core.models.Experiment` and
runs its training job on some backend -- the local process, an MLflow project, a
Kubeflow pipeline, and so on. Backends differ wildly (synchronous in-process
calls vs. asynchronous remote schedulers), so the abstraction is built around a
small lifecycle that fits both:

``launch`` -> submit the job and get a :class:`TrainingJob` handle
``poll``   -> check the handle's current :class:`~orchestrator.core.models.ExperimentStatus`
``result`` -> retrieve the terminal :class:`TrainingResult` (metrics + status)
``cancel`` -> best-effort stop

:meth:`TrainingLauncher.run` ties these together into a blocking convenience for
synchronous callers, and :func:`apply_result` writes a result back onto the
originating experiment.

This module ships only the interface, the :class:`TrainingJob` / :class:`TrainingResult`
value types, and a registry. Concrete backends -- the local backend in
:mod:`orchestrator.core.local_backend`, and remote backends under
:mod:`orchestrator.integrations` -- register themselves against it.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any, ClassVar
from uuid import uuid4

from pydantic import BaseModel, Field

from orchestrator.core.models import Experiment, ExperimentStatus, MetricValue

#: Statuses from which a job will not progress further.
TERMINAL_STATUSES = frozenset(
    {ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED}
)

#: A training callable: given an experiment, return a mapping of metric -> value.
TrainFn = Callable[[Experiment], Mapping[str, float]]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LauncherError(Exception):
    """Raised for invalid launcher usage, registry misses, or timeouts."""


class TrainingJob(BaseModel):
    """A handle to a launched training job.

    The ``handle`` field carries backend-specific data (a remote run id, a
    subprocess handle, etc.) that the owning launcher knows how to interpret.
    """

    id: str = Field(default_factory=lambda: uuid4().hex)
    experiment_id: str
    backend: str
    status: ExperimentStatus = ExperimentStatus.PENDING
    handle: Any = None
    created_at: datetime = Field(default_factory=_utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def is_terminal(self) -> bool:
        """Whether the job has reached a state it will not leave."""
        return self.status in TERMINAL_STATUSES


class TrainingResult(BaseModel):
    """The outcome of a training job: final status, metrics, and any error."""

    experiment_id: str
    status: ExperimentStatus
    metrics: list[MetricValue] = Field(default_factory=list)
    error: str | None = None


class TrainingLauncher(ABC):
    """Interface every training backend implements.

    Subclasses set a unique :attr:`name` and implement :meth:`launch`,
    :meth:`poll`, :meth:`result`, and :meth:`cancel`.
    """

    #: Stable identifier used for registry lookup and reporting.
    name: ClassVar[str] = ""

    @abstractmethod
    def launch(self, experiment: Experiment) -> TrainingJob:
        """Submit a training job for ``experiment`` and return its handle."""

    @abstractmethod
    def poll(self, job: TrainingJob) -> ExperimentStatus:
        """Return the current status of ``job``."""

    @abstractmethod
    def result(self, job: TrainingJob) -> TrainingResult:
        """Return the terminal result of ``job``.

        Raises
        ------
        LauncherError
            If the job has not reached a terminal state.
        """

    @abstractmethod
    def cancel(self, job: TrainingJob) -> None:
        """Best-effort cancellation of ``job``."""

    def run(
        self,
        experiment: Experiment,
        *,
        poll_interval: float = 0.0,
        timeout: float | None = None,
    ) -> TrainingResult:
        """Launch ``experiment`` and block until it reaches a terminal state.

        Parameters
        ----------
        poll_interval:
            Seconds to sleep between status polls.
        timeout:
            Maximum seconds to wait before raising :class:`LauncherError`.
            ``None`` waits indefinitely.
        """
        job = self.launch(experiment)
        waited = 0.0
        while self.poll(job) not in TERMINAL_STATUSES:
            if timeout is not None and waited >= timeout:
                self.cancel(job)
                raise LauncherError(f"training job {job.id} timed out after {timeout}s")
            time.sleep(poll_interval)
            waited += poll_interval if poll_interval else 0.0
            if not poll_interval and timeout is not None:
                # Avoid a busy-spin that can never time out.
                waited = timeout
        return self.result(job)


def apply_result(experiment: Experiment, result: TrainingResult) -> Experiment:
    """Write a :class:`TrainingResult` back onto its experiment, in place.

    Appends the result's metrics, sets the terminal status and error, and stamps
    ``finished_at`` (and ``started_at`` if it was never set).
    """
    if result.experiment_id != experiment.id:
        raise LauncherError(
            f"result for {result.experiment_id} does not match experiment {experiment.id}"
        )
    now = _utcnow()
    if experiment.started_at is None:
        experiment.started_at = now
    experiment.metrics.extend(result.metrics)
    experiment.status = result.status
    experiment.error = result.error
    experiment.finished_at = now
    return experiment


_REGISTRY: dict[str, type[TrainingLauncher]] = {}


def register_launcher(cls: type[TrainingLauncher]) -> type[TrainingLauncher]:
    """Class decorator that registers a launcher under its :attr:`name`."""
    name = getattr(cls, "name", "")
    if not name:
        raise LauncherError(f"{cls.__name__} must define a non-empty 'name' to be registered")
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise LauncherError(f"launcher name {name!r} already registered to {existing.__name__}")
    _REGISTRY[name] = cls
    return cls


def get_launcher(name: str, **kwargs: Any) -> TrainingLauncher:
    """Instantiate a registered launcher by ``name``, forwarding ``kwargs``."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise LauncherError(f"unknown launcher {name!r}; registered: {known}") from None
    return cls(**kwargs)


def available_launchers() -> list[str]:
    """Return the sorted names of all registered launchers."""
    return sorted(_REGISTRY)

"""Local training backend (concrete launcher for pipeline stage 3).

:class:`LocalLauncher` runs training jobs on the local machine using a thread
pool. Unlike a toy synchronous runner, it implements the full asynchronous
launcher lifecycle, which is what makes the rest of the pipeline meaningful:

* **Asynchronous** -- :meth:`launch` submits work to a pool and returns
  immediately; :meth:`poll` reflects live status; :meth:`run` blocks until done.
* **Concurrent** -- up to ``max_workers`` jobs run at once, matching an
  objective's ``max_concurrency``.
* **Streaming** -- training functions may report intermediate metrics through a
  :class:`TrainContext`, which the monitoring stage can read live via
  :meth:`LocalLauncher.live_metrics`.
* **Cancellable** -- :meth:`cancel` signals a cooperative stop that well-behaved
  training functions observe through :meth:`TrainContext.should_stop`.

Training functions come in two shapes; the launcher detects which by arity:

.. code-block:: python

    def simple(experiment) -> dict[str, float]: ...
    def streaming(experiment, ctx: TrainContext) -> dict[str, float] | None: ...

Either may return a final metric mapping; streaming functions may also report
metrics incrementally and should check ``ctx.should_stop()`` periodically.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, ClassVar

from orchestrator.core.launcher import (
    LauncherError,
    TrainingJob,
    TrainingLauncher,
    TrainingResult,
    register_launcher,
)
from orchestrator.core.models import Experiment, ExperimentStatus, MetricValue


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TrainContext:
    """Handed to streaming training functions to report progress and observe cancellation.

    Instances are created by the launcher; training code does not construct them.
    """

    def __init__(self, experiment: Experiment, cancel_event: threading.Event) -> None:
        self.experiment = experiment
        self._cancel = cancel_event
        self._metrics: list[MetricValue] = []
        self._lock = threading.Lock()

    def report(self, name: str, value: float, step: int = 0) -> None:
        """Record an intermediate metric observation."""
        with self._lock:
            self._metrics.append(MetricValue(name=str(name), value=float(value), step=step))

    def should_stop(self) -> bool:
        """Return ``True`` once cancellation has been requested."""
        return self._cancel.is_set()

    def snapshot(self) -> list[MetricValue]:
        """Return a copy of the metrics reported so far."""
        with self._lock:
            return list(self._metrics)


@dataclass
class _JobState:
    """Internal per-job bookkeeping held by the launcher."""

    cancel: threading.Event
    context: TrainContext
    future: Future[Mapping[str, float] | None] | None = None
    status: ExperimentStatus = ExperimentStatus.RUNNING
    error: str | None = None
    final_metrics: list[MetricValue] = field(default_factory=list)


# A training callable taking just the experiment, or the experiment plus context.
SimpleTrainFn = Callable[[Experiment], "Mapping[str, float] | None"]
StreamingTrainFn = Callable[[Experiment, TrainContext], "Mapping[str, float] | None"]
LocalTrainFn = "SimpleTrainFn | StreamingTrainFn"


def _wants_context(train_fn: Callable[..., Any]) -> bool:
    """Return ``True`` if ``train_fn`` accepts a second (context) argument."""
    try:
        sig = inspect.signature(train_fn)
    except (TypeError, ValueError):
        return False
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    has_var_positional = any(p.kind is p.VAR_POSITIONAL for p in sig.parameters.values())
    return len(positional) >= 2 or has_var_positional


@register_launcher
class LocalLauncher(TrainingLauncher):
    """Run training jobs locally on a background thread pool.

    Parameters
    ----------
    train_fn:
        The training callable. Either ``train_fn(experiment)`` or
        ``train_fn(experiment, ctx)``; the launcher detects the signature.
    max_workers:
        Maximum number of jobs to run concurrently.
    """

    name: ClassVar[str] = "local"

    def __init__(self, train_fn: Callable[..., Any], *, max_workers: int = 1) -> None:
        if max_workers < 1:
            raise LauncherError("max_workers must be a positive integer")
        self._train_fn = train_fn
        self._wants_context = _wants_context(train_fn)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, _JobState] = {}

    # -- lifecycle ---------------------------------------------------------

    def launch(self, experiment: Experiment) -> TrainingJob:
        job = TrainingJob(
            experiment_id=experiment.id,
            backend=self.name,
            status=ExperimentStatus.RUNNING,
            started_at=_utcnow(),
        )
        cancel = threading.Event()
        state = _JobState(cancel=cancel, context=TrainContext(experiment, cancel))
        state.future = self._executor.submit(self._run_training, experiment, state)
        self._jobs[job.id] = state
        return job

    def poll(self, job: TrainingJob) -> ExperimentStatus:
        state = self._state(job)
        future = state.future
        if future is not None and future.done() and not job.is_terminal:
            # Surface any unexpected worker crash that bypassed our try/except.
            exc = future.exception()
            if exc is not None and state.status is ExperimentStatus.RUNNING:
                state.status = ExperimentStatus.FAILED
                state.error = repr(exc)
            job.status = state.status
            job.finished_at = _utcnow()
        return job.status

    def result(self, job: TrainingJob) -> TrainingResult:
        state = self._state(job)
        if self.poll(job) not in (
            ExperimentStatus.COMPLETED,
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED,
        ):
            raise LauncherError(f"job {job.id} is not yet terminal")
        metrics = state.context.snapshot() + state.final_metrics
        return TrainingResult(
            experiment_id=job.experiment_id,
            status=state.status,
            metrics=metrics,
            error=state.error,
        )

    def cancel(self, job: TrainingJob) -> None:
        state = self._state(job)
        state.cancel.set()
        if state.future is not None:
            # Cancels only if the job has not started running yet.
            state.future.cancel()

    # -- monitoring hook ---------------------------------------------------

    def live_metrics(self, job: TrainingJob) -> list[MetricValue]:
        """Return metrics reported so far, without waiting for completion."""
        return self._state(job).context.snapshot()

    # -- teardown ----------------------------------------------------------

    def close(self, *, wait: bool = True) -> None:
        """Shut down the underlying thread pool."""
        self._executor.shutdown(wait=wait)

    def __enter__(self) -> LocalLauncher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _state(self, job: TrainingJob) -> _JobState:
        try:
            return self._jobs[job.id]
        except KeyError:
            raise LauncherError(f"unknown job {job.id}") from None

    def _run_training(
        self, experiment: Experiment, state: _JobState
    ) -> Mapping[str, float] | None:
        if state.cancel.is_set():
            state.status = ExperimentStatus.CANCELLED
            return None
        try:
            if self._wants_context:
                raw = self._train_fn(experiment, state.context)
            else:
                raw = self._train_fn(experiment)
            if raw:
                state.final_metrics = [
                    MetricValue(name=str(k), value=float(v)) for k, v in raw.items()
                ]
            state.status = (
                ExperimentStatus.CANCELLED if state.cancel.is_set() else ExperimentStatus.COMPLETED
            )
            return raw
        except Exception as exc:  # noqa: BLE001 - captured into the job result
            state.status = ExperimentStatus.FAILED
            state.error = repr(exc)
            return None

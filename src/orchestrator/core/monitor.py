"""Live metric monitoring and streaming (pipeline stage 4: *monitor metrics*).

While training runs, the orchestrator needs to watch metrics as they arrive --
to drive dashboards, feed the hyperparameter-adjustment stage, and detect early
stopping. :class:`MetricMonitor` provides that view in a backend-agnostic way.

Given a launcher and one or more :class:`~orchestrator.core.launcher.TrainingJob`
handles, the monitor periodically reads each job's live metrics (via the
backend's ``live_metrics`` hook when available), diffs them against what it has
already seen, and emits a :class:`MetricEvent` for every new observation. When a
job reaches a terminal state it also pulls the final result, emits any remaining
metrics, and reports the status.

Two consumption modes share the same diffing engine:

* **Push** -- register :class:`MetricListener` objects and call :meth:`start`
  to monitor on a background thread (or :meth:`poll_once` for a single pass).
* **Pull** -- iterate :meth:`stream`, which yields events as they occur and
  returns when every tracked job has finished.

Backends that do not implement ``live_metrics`` still work; their metrics simply
surface in one batch when the job completes.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable, Iterator
from datetime import datetime

from pydantic import BaseModel, Field

from orchestrator.core.launcher import (
    TERMINAL_STATUSES,
    TrainingJob,
    TrainingLauncher,
    TrainingResult,
)
from orchestrator.core.models import ExperimentStatus, Goal, MetricValue


class MonitorError(Exception):
    """Raised for invalid monitor usage (e.g. a timeout while waiting)."""


class MetricEvent(BaseModel):
    """A single metric observation surfaced by the monitor."""

    experiment_id: str
    name: str
    value: float
    step: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)

    @classmethod
    def from_value(cls, experiment_id: str, metric: MetricValue) -> MetricEvent:
        """Build an event from a recorded :class:`MetricValue`."""
        return cls(
            experiment_id=experiment_id,
            name=metric.name,
            value=metric.value,
            step=metric.step,
            timestamp=metric.timestamp,
        )


class MetricListener:
    """Base class for monitor subscribers. Override the hooks you care about."""

    def on_metric(self, event: MetricEvent) -> None:
        """Called for every new metric observation."""

    def on_status(
        self,
        experiment_id: str,
        status: ExperimentStatus,
        result: TrainingResult | None,
    ) -> None:
        """Called once when a tracked job reaches a terminal state."""


class CallbackListener(MetricListener):
    """Adapt plain callables into a :class:`MetricListener`."""

    def __init__(
        self,
        on_metric=None,
        on_status=None,
    ) -> None:
        self._on_metric = on_metric
        self._on_status = on_status

    def on_metric(self, event: MetricEvent) -> None:
        if self._on_metric is not None:
            self._on_metric(event)

    def on_status(
        self, experiment_id: str, status: ExperimentStatus, result: TrainingResult | None
    ) -> None:
        if self._on_status is not None:
            self._on_status(experiment_id, status, result)


class MetricHistory(MetricListener):
    """Accumulates every observed event, queryable per experiment."""

    def __init__(self) -> None:
        self._events: dict[str, list[MetricEvent]] = {}
        self._statuses: dict[str, ExperimentStatus] = {}
        self._lock = threading.Lock()

    def on_metric(self, event: MetricEvent) -> None:
        with self._lock:
            self._events.setdefault(event.experiment_id, []).append(event)

    def on_status(
        self, experiment_id: str, status: ExperimentStatus, result: TrainingResult | None
    ) -> None:
        with self._lock:
            self._statuses[experiment_id] = status

    def events(self, experiment_id: str) -> list[MetricEvent]:
        """All events recorded for ``experiment_id``, in arrival order."""
        with self._lock:
            return list(self._events.get(experiment_id, []))

    def status(self, experiment_id: str) -> ExperimentStatus | None:
        """Terminal status of ``experiment_id``, if it has finished."""
        with self._lock:
            return self._statuses.get(experiment_id)

    def latest(self, experiment_id: str, name: str) -> float | None:
        """Most recently observed value of ``name`` for ``experiment_id``."""
        matching = [e for e in self.events(experiment_id) if e.name == name]
        if not matching:
            return None
        return max(matching, key=lambda e: (e.step, e.timestamp)).value

    def best(self, experiment_id: str, name: str, goal: Goal) -> float | None:
        """Best observed value of ``name`` under ``goal`` for ``experiment_id``."""
        values = [e.value for e in self.events(experiment_id) if e.name == name]
        if not values:
            return None
        return max(values) if goal is Goal.MAXIMIZE else min(values)


class BestMetricTracker(MetricListener):
    """Tracks the best value of one metric and fires a callback on improvement."""

    def __init__(self, metric: str, goal: Goal, on_improve=None) -> None:
        self.metric = metric
        self.goal = goal
        self._on_improve = on_improve
        self.best_value: float | None = None
        self.best_experiment_id: str | None = None
        self._lock = threading.Lock()

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        return value > self.best_value if self.goal is Goal.MAXIMIZE else value < self.best_value

    def on_metric(self, event: MetricEvent) -> None:
        if event.name != self.metric:
            return
        improved = False
        with self._lock:
            if self._is_better(event.value):
                self.best_value = event.value
                self.best_experiment_id = event.experiment_id
                improved = True
        if improved and self._on_improve is not None:
            self._on_improve(event)


class MetricMonitor:
    """Watches training jobs and emits their metrics as they arrive.

    Parameters
    ----------
    launcher:
        The launcher whose jobs are being monitored. If it exposes a
        ``live_metrics(job)`` method, intermediate metrics stream in real time;
        otherwise metrics surface when each job completes.
    interval:
        Seconds between polling passes in :meth:`start` / :meth:`stream`.
    """

    def __init__(self, launcher: TrainingLauncher, *, interval: float = 0.1) -> None:
        self._launcher = launcher
        self.interval = interval
        self._tracked: dict[str, TrainingJob] = {}
        self._seen: dict[str, int] = {}
        self._listeners: list[MetricListener] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- registration ------------------------------------------------------

    def add_listener(self, listener: MetricListener) -> MetricListener:
        """Register a listener; returns it for convenient inline construction."""
        with self._lock:
            self._listeners.append(listener)
        return listener

    def track(self, job: TrainingJob) -> None:
        """Begin monitoring ``job``."""
        with self._lock:
            self._tracked[job.id] = job
            self._seen.setdefault(job.id, 0)

    def track_all(self, jobs: Iterable[TrainingJob]) -> None:
        """Begin monitoring every job in ``jobs``."""
        for job in jobs:
            self.track(job)

    @property
    def active(self) -> bool:
        """Whether any tracked jobs remain unfinished."""
        with self._lock:
            return bool(self._tracked)

    # -- consumption: pull -------------------------------------------------

    def stream(self, *, timeout: float | None = None) -> Iterator[MetricEvent]:
        """Yield metric events until every tracked job is terminal.

        Raises
        ------
        MonitorError
            If ``timeout`` seconds elapse before all jobs finish.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.active:
            for event in self._collect():
                yield event
            if not self.active:
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise MonitorError("monitor timed out before all jobs finished")
            time.sleep(self.interval)
        # Final drain in case the last pass completed jobs.
        for event in self._collect():
            yield event

    # -- consumption: push -------------------------------------------------

    def poll_once(self) -> list[MetricEvent]:
        """Run a single polling pass, notifying listeners. Returns new events."""
        return self._collect()

    def start(self) -> None:
        """Begin monitoring on a background thread."""
        if self._thread is not None and self._thread.is_alive():
            raise MonitorError("monitor already running")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="metric-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop after its current pass."""
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the background thread to finish."""
        if self._thread is not None:
            self._thread.join(timeout)

    def __enter__(self) -> MetricMonitor:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
        self.join()

    def _run(self) -> None:
        while not self._stop.is_set() and self.active:
            self._collect()
            self._stop.wait(self.interval)
        # One last pass to flush metrics from jobs that just finished.
        if not self._stop.is_set():
            self._collect()

    # -- diffing engine ----------------------------------------------------

    def _live_metrics(self, job: TrainingJob) -> list[MetricValue]:
        hook = getattr(self._launcher, "live_metrics", None)
        if callable(hook):
            return list(hook(job))
        return []

    def _collect(self) -> list[MetricEvent]:
        """Poll every tracked job once; emit new metrics and terminal statuses."""
        with self._lock:
            jobs = list(self._tracked.values())

        events: list[MetricEvent] = []
        statuses: list[tuple[str, ExperimentStatus, TrainingResult | None]] = []

        for job in jobs:
            seen = self._seen.get(job.id, 0)
            live = self._live_metrics(job)
            for metric in live[seen:]:
                events.append(MetricEvent.from_value(job.experiment_id, metric))
            new_seen = max(seen, len(live))

            status = self._launcher.poll(job)
            if status in TERMINAL_STATUSES:
                result: TrainingResult | None = None
                try:
                    result = self._launcher.result(job)
                except Exception:  # noqa: BLE001 - result may be unavailable
                    result = None
                if result is not None:
                    for metric in result.metrics[new_seen:]:
                        events.append(MetricEvent.from_value(job.experiment_id, metric))
                    new_seen = max(new_seen, len(result.metrics))
                statuses.append((job.experiment_id, status, result))

            with self._lock:
                self._seen[job.id] = new_seen
                if status in TERMINAL_STATUSES:
                    self._tracked.pop(job.id, None)

        # Notify outside the lock so listeners can safely call back in.
        with self._lock:
            listeners = list(self._listeners)
        for event in events:
            for listener in listeners:
                listener.on_metric(event)
        for experiment_id, status, result in statuses:
            for listener in listeners:
                listener.on_status(experiment_id, status, result)

        return events

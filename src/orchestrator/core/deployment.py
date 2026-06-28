"""Best-model selection and deployment (pipeline stage 8: *deploy best model*).

The final stage promotes the winning experiment to a serving target. It splits
into two concerns:

* **Selection** -- :func:`select_best` reuses the ranking engine to pick the top
  experiment, with guards for run status and a minimum acceptable score, so a
  failed or underwhelming search does not get promoted by accident.
* **Deployment** -- :class:`DeploymentTarget` is the abstraction for "make this
  model serve somewhere". Targets register themselves (like launchers and
  strategies) and are resolved by name. A dependency-free
  :class:`LocalDeploymentTarget` records the promotion and optionally writes a
  manifest, which is enough for local workflows and tests; cloud/registry targets
  implement the same interface.

:func:`deploy_best` ties the two together: rank, gate, and deploy in one call.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Sequence
from uuid import uuid4

from orchestrator.core.models import Experiment, ExperimentStatus, Goal
from orchestrator.core.ranking import MetricKey, RankedExperiment, rank_result

if TYPE_CHECKING:
    from orchestrator.core.loop import TuningResult


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DeploymentError(Exception):
    """Raised for invalid deployment usage or target-registry misses."""


class DeploymentStatus(str, Enum):
    """Lifecycle of a deployment."""

    PENDING = "pending"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class Deployment:
    """A record of promoting one experiment's model to a target."""

    experiment_id: str
    target: str
    status: DeploymentStatus
    uri: str | None = None
    metric: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view of the deployment record."""
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "target": self.target,
            "status": self.status.value,
            "uri": self.uri,
            "metric": self.metric,
            "score": self.score,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


class DeploymentTarget(ABC):
    """Interface for promoting a model to a serving target."""

    #: Stable identifier for registry lookup and reporting.
    name: ClassVar[str] = ""

    @abstractmethod
    def deploy(
        self,
        experiment: Experiment,
        *,
        metric: str | None = None,
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Deployment:
        """Promote ``experiment``'s model and return a :class:`Deployment`."""


class LocalDeploymentTarget(DeploymentTarget):
    """Reference target that records the promotion locally.

    Keeps deployments in memory and, when ``directory`` is given, writes a JSON
    manifest per deployment. Useful for local workflows and as a test double for
    the deployment flow.
    """

    name: ClassVar[str] = "local"

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory) if directory is not None else None
        self.deployments: list[Deployment] = []
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

    def deploy(
        self,
        experiment: Experiment,
        *,
        metric: str | None = None,
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Deployment:
        model_name = experiment.name or experiment.id
        deployment = Deployment(
            experiment_id=experiment.id,
            target=self.name,
            status=DeploymentStatus.DEPLOYED,
            uri=f"local://models/{model_name}",
            metric=metric,
            score=score,
            metadata={"hyperparameters": dict(experiment.hyperparameters), **(metadata or {})},
        )
        self.deployments.append(deployment)
        if self.directory is not None:
            manifest = self.directory / f"{deployment.id}.json"
            manifest.write_text(json.dumps(deployment.to_dict(), indent=2, default=str), encoding="utf-8")
        return deployment

    @property
    def current(self) -> Deployment | None:
        """The most recent deployment, if any."""
        return self.deployments[-1] if self.deployments else None


_REGISTRY: dict[str, type[DeploymentTarget]] = {}


def register_target(cls: type[DeploymentTarget]) -> type[DeploymentTarget]:
    """Class decorator registering a deployment target under its :attr:`name`."""
    name = getattr(cls, "name", "")
    if not name:
        raise DeploymentError(f"{cls.__name__} must define a non-empty 'name' to be registered")
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise DeploymentError(f"target name {name!r} already registered to {existing.__name__}")
    _REGISTRY[name] = cls
    return cls


def get_target(name: str, **kwargs: Any) -> DeploymentTarget:
    """Instantiate a registered deployment target by ``name``."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise DeploymentError(f"unknown target {name!r}; registered: {known}") from None
    return cls(**kwargs)


def available_targets() -> list[str]:
    """Return the sorted names of all registered deployment targets."""
    return sorted(_REGISTRY)


register_target(LocalDeploymentTarget)


def select_best(
    result: TuningResult,
    *,
    statuses: Sequence[ExperimentStatus] | None = (ExperimentStatus.COMPLETED,),
    secondary: Sequence[MetricKey] = (),
    method: str = "best",
    min_score: float | None = None,
) -> RankedExperiment | None:
    """Select the best experiment from a tuning result, or ``None``.

    Parameters
    ----------
    statuses:
        Only experiments in these statuses are eligible. Defaults to completed
        runs only. Pass ``None`` to consider every experiment with a score.
    secondary:
        Tie-break ``(metric, goal)`` keys passed through to ranking.
    min_score:
        If set, the winner must meet this score (``>=`` for maximize, ``<=`` for
        minimize) to be selected.
    """
    board = rank_result(
        result,
        secondary=secondary,
        method=method,
        statuses=list(statuses) if statuses is not None else None,
    )
    best = board.best
    if best is None:
        return None
    if min_score is not None:
        meets = best.score >= min_score if result.goal is Goal.MAXIMIZE else best.score <= min_score
        if not meets:
            return None
    return best


def deploy_best(
    result: TuningResult,
    target: DeploymentTarget | str,
    *,
    statuses: Sequence[ExperimentStatus] | None = (ExperimentStatus.COMPLETED,),
    secondary: Sequence[MetricKey] = (),
    method: str = "best",
    min_score: float | None = None,
    require_target_reached: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Deployment | None:
    """Select the best experiment and deploy it to ``target``.

    Returns the :class:`Deployment`, or ``None`` if nothing qualified for
    promotion (no eligible runs, score gate not met, or ``require_target_reached``
    set while the objective's target was missed).

    ``target`` may be a :class:`DeploymentTarget` instance or a registered name.
    """
    if require_target_reached and not result.target_reached:
        return None

    best = select_best(
        result, statuses=statuses, secondary=secondary, method=method, min_score=min_score
    )
    if best is None:
        return None

    resolved = get_target(target) if isinstance(target, str) else target
    return resolved.deploy(
        best.experiment,
        metric=result.primary_metric,
        score=best.score,
        metadata=metadata,
    )

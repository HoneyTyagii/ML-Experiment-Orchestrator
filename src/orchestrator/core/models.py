"""Core domain models for the ML Experiment Orchestrator.

These models define the shared vocabulary used across every stage of the
pipeline: from objective intake through experiment generation, training,
evaluation, and deployment.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid4().hex


class Goal(str, Enum):
    """Optimization direction for the primary metric."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ExperimentStatus(str, Enum):
    """Lifecycle states for a single experiment run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HyperparameterType(str, Enum):
    """Supported hyperparameter value types for search spaces."""

    FLOAT = "float"
    INT = "int"
    CATEGORICAL = "categorical"
    BOOL = "bool"


class HyperparameterSpec(BaseModel):
    """Definition of a single tunable hyperparameter.

    For ``FLOAT`` and ``INT`` types, provide ``low`` and ``high`` bounds.
    For ``CATEGORICAL`` and ``BOOL`` types, provide ``choices``.
    """

    name: str
    type: HyperparameterType
    low: float | None = None
    high: float | None = None
    choices: list[Any] | None = None
    log_scale: bool = False

    @field_validator("choices")
    @classmethod
    def _non_empty_choices(cls, value: list[Any] | None) -> list[Any] | None:
        if value is not None and len(value) == 0:
            raise ValueError("choices must not be empty when provided")
        return value

    def validate_space(self) -> None:
        """Ensure the spec is internally consistent for its type."""
        if self.type in (HyperparameterType.FLOAT, HyperparameterType.INT):
            if self.low is None or self.high is None:
                raise ValueError(f"{self.name}: numeric params require 'low' and 'high'")
            if self.low > self.high:
                raise ValueError(f"{self.name}: 'low' must not exceed 'high'")
        elif self.type is HyperparameterType.CATEGORICAL:
            if not self.choices:
                raise ValueError(f"{self.name}: categorical params require 'choices'")


class MetricValue(BaseModel):
    """A single recorded metric observation."""

    name: str
    value: float
    step: int = 0
    timestamp: datetime = Field(default_factory=_utcnow)


class Objective(BaseModel):
    """A high-level experimentation objective supplied by the user.

    This is the entry point of the pipeline: it captures what the agent is
    trying to achieve and the constraints it must operate within.
    """

    id: str = Field(default_factory=_new_id)
    name: str
    description: str = ""
    primary_metric: str
    goal: Goal = Goal.MAXIMIZE
    dataset: str | None = None
    search_space: list[HyperparameterSpec] = Field(default_factory=list)
    max_experiments: int = 10
    max_concurrency: int = 1
    target_metric_value: float | None = None
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("max_experiments", "max_concurrency")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be a positive integer")
        return value

    def validate_search_space(self) -> None:
        """Validate every hyperparameter spec in the search space."""
        for spec in self.search_space:
            spec.validate_space()


class Experiment(BaseModel):
    """A single experiment: a concrete hyperparameter configuration and its run.

    Experiments are generated from an :class:`Objective`, executed by a training
    backend, and ranked during comparison.
    """

    id: str = Field(default_factory=_new_id)
    objective_id: str
    name: str = ""
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    status: ExperimentStatus = ExperimentStatus.PENDING
    metrics: list[MetricValue] = Field(default_factory=list)
    run_uri: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def record_metric(self, name: str, value: float, step: int = 0) -> None:
        """Append a metric observation to this experiment."""
        self.metrics.append(MetricValue(name=name, value=value, step=step))

    def latest_metric(self, name: str) -> float | None:
        """Return the most recently recorded value for ``name``, if any."""
        observations = [m for m in self.metrics if m.name == name]
        if not observations:
            return None
        return max(observations, key=lambda m: (m.step, m.timestamp)).value

    def best_metric(self, name: str, goal: Goal) -> float | None:
        """Return the best observed value for ``name`` under ``goal``."""
        values = [m.value for m in self.metrics if m.name == name]
        if not values:
            return None
        return max(values) if goal is Goal.MAXIMIZE else min(values)

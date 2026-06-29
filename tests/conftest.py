"""Shared fixtures for the orchestrator test suite."""

from __future__ import annotations

import pytest

from orchestrator.core import (
    Experiment,
    ExperimentStatus,
    Goal,
    Objective,
    intake_objective,
)


@pytest.fixture
def objective() -> Objective:
    """A small, fully-typed objective covering every hyperparameter type."""
    return intake_objective(
        {
            "name": "test-obj",
            "primary_metric": "val_accuracy",
            "goal": "maximize",
            "dataset": "toy",
            "max_experiments": 6,
            "max_concurrency": 2,
            "search_space": [
                {"name": "learning_rate", "type": "float", "low": 0.001, "high": 0.1, "log_scale": True},
                {"name": "batch_size", "type": "int", "low": 16, "high": 64},
                {"name": "optimizer", "type": "categorical", "choices": ["adam", "sgd"]},
                {"name": "augment", "type": "bool"},
            ],
        }
    )


@pytest.fixture
def categorical_objective() -> Objective:
    """An objective whose search space is a small, fully-enumerable grid (2x2)."""
    return intake_objective(
        {
            "name": "tiny",
            "primary_metric": "val_accuracy",
            "goal": "maximize",
            "max_experiments": 50,
            "max_concurrency": 4,
            "search_space": [
                {"name": "opt", "type": "categorical", "choices": ["a", "b"]},
                {"name": "flag", "type": "bool"},
            ],
        }
    )


def make_experiment(name: str, objective_id: str = "obj", **metric_series) -> Experiment:
    """Build a COMPLETED experiment with the given metric series.

    Each keyword is ``metric_name=[v0, v1, ...]`` recorded at increasing steps.
    """
    exp = Experiment(objective_id=objective_id, name=name)
    for metric, values in metric_series.items():
        for step, value in enumerate(values):
            exp.record_metric(metric, value, step=step)
    exp.status = ExperimentStatus.COMPLETED
    return exp


@pytest.fixture
def make_exp():
    """Expose :func:`make_experiment` as a fixture."""
    return make_experiment

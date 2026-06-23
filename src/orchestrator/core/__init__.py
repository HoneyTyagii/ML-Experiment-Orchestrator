"""Core domain logic for the orchestrator."""

from orchestrator.core.config import ConfigError, load_objective, objective_from_dict
from orchestrator.core.intake import (
    IntakeError,
    Severity,
    ValidationIssue,
    ValidationReport,
    intake_objective,
    validate_objective,
)
from orchestrator.core.models import (
    Experiment,
    ExperimentStatus,
    Goal,
    HyperparameterSpec,
    HyperparameterType,
    MetricValue,
    Objective,
)
from orchestrator.core.search import GridSearch, RandomSearch
from orchestrator.core.strategy import (
    ExperimentStrategy,
    StrategyError,
    available_strategies,
    get_strategy,
    register_strategy,
)

__all__ = [
    "ConfigError",
    "Experiment",
    "ExperimentStatus",
    "ExperimentStrategy",
    "Goal",
    "GridSearch",
    "HyperparameterSpec",
    "HyperparameterType",
    "IntakeError",
    "MetricValue",
    "Objective",
    "RandomSearch",
    "Severity",
    "StrategyError",
    "ValidationIssue",
    "ValidationReport",
    "available_strategies",
    "get_strategy",
    "intake_objective",
    "load_objective",
    "objective_from_dict",
    "register_strategy",
    "validate_objective",
]

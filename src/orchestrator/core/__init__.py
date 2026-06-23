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
from orchestrator.core.launcher import (
    LauncherError,
    TrainingJob,
    TrainingLauncher,
    TrainingResult,
    apply_result,
    available_launchers,
    get_launcher,
    register_launcher,
)
from orchestrator.core.local_backend import LocalLauncher, TrainContext
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
    "LauncherError",
    "LocalLauncher",
    "MetricValue",
    "Objective",
    "RandomSearch",
    "Severity",
    "StrategyError",
    "TrainContext",
    "TrainingJob",
    "TrainingLauncher",
    "TrainingResult",
    "ValidationIssue",
    "ValidationReport",
    "apply_result",
    "available_launchers",
    "available_strategies",
    "get_launcher",
    "get_strategy",
    "intake_objective",
    "load_objective",
    "objective_from_dict",
    "register_launcher",
    "register_strategy",
    "validate_objective",
]

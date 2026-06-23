"""Objective intake and validation.

This module is the front door of the pipeline (stage 1: *receive objective*).
It accepts an objective from any source -- a YAML/JSON file, a raw mapping, or
an already-constructed :class:`~orchestrator.core.models.Objective` -- and runs
a battery of *semantic* checks on top of the structural validation performed by
the pydantic models and :func:`Objective.validate_search_space`.

The structural layer (in :mod:`orchestrator.core.config` and the models) answers
"is this well-formed?". The intake layer answers "does this make sense to run?":
it catches duplicate hyperparameter names, log-scale ranges that include
non-positive values, a concurrency budget larger than the experiment budget, and
similar issues that are valid as data but problematic as an experiment plan.

Validation produces a :class:`ValidationReport` carrying ``error`` and
``warning`` issues. Errors block intake; warnings are surfaced but do not.

Example
-------
.. code-block:: python

    from orchestrator.core.intake import intake_objective

    objective = intake_objective("examples/objectives/tune_resnet.yaml")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from orchestrator.core.config import load_objective, objective_from_dict
from orchestrator.core.models import HyperparameterSpec, HyperparameterType, Objective


class Severity(str, Enum):
    """Severity of a validation issue."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    """A single problem found while validating an objective.

    Parameters
    ----------
    severity:
        Whether the issue blocks intake (:attr:`Severity.ERROR`) or is merely
        advisory (:attr:`Severity.WARNING`).
    field:
        Dotted path to the offending field, e.g. ``"search_space.learning_rate"``.
    message:
        Human-readable description of the problem.
    """

    severity: Severity
    field: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.field}: {self.message}"


@dataclass
class ValidationReport:
    """The collected result of validating an objective."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        """All issues with :attr:`Severity.ERROR`."""
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """All issues with :attr:`Severity.WARNING`."""
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        """``True`` when there are no error-level issues."""
        return not self.errors

    def add_error(self, field: str, message: str) -> None:
        """Record an error-level issue."""
        self.issues.append(ValidationIssue(Severity.ERROR, field, message))

    def add_warning(self, field: str, message: str) -> None:
        """Record a warning-level issue."""
        self.issues.append(ValidationIssue(Severity.WARNING, field, message))

    def raise_if_errors(self) -> None:
        """Raise :class:`IntakeError` if any error-level issues are present."""
        if self.errors:
            raise IntakeError(self)

    def __str__(self) -> str:
        if not self.issues:
            return "objective valid: no issues"
        return "\n".join(str(issue) for issue in self.issues)


class IntakeError(Exception):
    """Raised when an objective fails intake validation.

    The originating :class:`ValidationReport` is attached as :attr:`report` so
    callers can inspect every issue, not just the summary message.
    """

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        errors = report.errors
        summary = "; ".join(f"{i.field}: {i.message}" for i in errors) or "unknown error"
        super().__init__(f"objective failed validation ({len(errors)} error(s)): {summary}")


def _is_integral(value: float) -> bool:
    return float(value).is_integer()


def _validate_spec(spec: HyperparameterSpec, report: ValidationReport) -> None:
    """Run semantic checks for a single hyperparameter spec."""
    loc = f"search_space.{spec.name}"

    if not spec.name.strip():
        report.add_error("search_space", "hyperparameter name must not be blank")

    # Structural consistency (bounds present and ordered, choices present).
    try:
        spec.validate_space()
    except ValueError as exc:
        report.add_error(loc, str(exc))
        return

    if spec.type in (HyperparameterType.FLOAT, HyperparameterType.INT):
        # validate_space guarantees low/high are set and low <= high here.
        if spec.low == spec.high:
            report.add_warning(loc, "'low' equals 'high'; this parameter is effectively constant")
        if spec.log_scale and spec.low is not None and spec.low <= 0:
            report.add_error(loc, "log_scale requires 'low' to be greater than 0")
        if spec.type is HyperparameterType.INT:
            if spec.low is not None and not _is_integral(spec.low):
                report.add_warning(loc, "int param 'low' is not a whole number; it will be rounded")
            if spec.high is not None and not _is_integral(spec.high):
                report.add_warning(loc, "int param 'high' is not a whole number; it will be rounded")

    if spec.choices is not None:
        if len(spec.choices) != len(set(map(_hashable, spec.choices))):
            report.add_warning(loc, "duplicate values in 'choices'")
        if spec.type is HyperparameterType.BOOL and not set(spec.choices) <= {True, False}:
            report.add_warning(loc, "bool param 'choices' should only contain true/false")


def _hashable(value: Any) -> Any:
    """Best-effort conversion of a choice value into something hashable."""
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def validate_objective(objective: Objective) -> ValidationReport:
    """Validate an objective and return a :class:`ValidationReport`.

    This performs semantic checks beyond the structural validation already done
    by the pydantic models. It never raises for validation problems; inspect the
    returned report (or call :meth:`ValidationReport.raise_if_errors`).
    """
    report = ValidationReport()

    if not objective.name.strip():
        report.add_error("name", "objective name must not be blank")

    if not objective.primary_metric.strip():
        report.add_error("primary_metric", "primary_metric must not be blank")

    if objective.max_concurrency > objective.max_experiments:
        report.add_warning(
            "max_concurrency",
            f"max_concurrency ({objective.max_concurrency}) exceeds max_experiments "
            f"({objective.max_experiments}); it will be capped at max_experiments",
        )

    if not objective.search_space:
        report.add_warning(
            "search_space",
            "search space is empty; only a single default experiment can be generated",
        )

    seen: dict[str, int] = {}
    for spec in objective.search_space:
        seen[spec.name] = seen.get(spec.name, 0) + 1
        _validate_spec(spec, report)

    for name, count in seen.items():
        if count > 1:
            report.add_error("search_space", f"duplicate hyperparameter name: {name!r}")

    return report


def _coerce_to_objective(source: Any) -> Objective:
    """Load an :class:`Objective` from any supported source type."""
    if isinstance(source, Objective):
        return source
    if isinstance(source, dict):
        return objective_from_dict(source)
    if isinstance(source, (str, Path)):
        return load_objective(source)
    raise TypeError(
        "objective source must be an Objective, mapping, or file path, "
        f"got {type(source).__name__}"
    )


def intake_objective(source: Any, *, strict: bool = False) -> Objective:
    """Receive and validate an objective from any supported source.

    Parameters
    ----------
    source:
        An :class:`Objective`, a raw mapping, or a path to a YAML/JSON file.
    strict:
        When ``True``, warnings are promoted to errors and also block intake.

    Returns
    -------
    Objective
        The validated objective, ready to feed into experiment generation.

    Raises
    ------
    ConfigError
        If the source cannot be loaded or is structurally invalid.
    IntakeError
        If semantic validation finds error-level issues (or any issue when
        ``strict`` is set).
    """
    objective = _coerce_to_objective(source)
    report = validate_objective(objective)

    if strict and report.warnings:
        promoted = ValidationReport(
            issues=[ValidationIssue(Severity.ERROR, i.field, i.message) for i in report.issues]
        )
        promoted.raise_if_errors()

    report.raise_if_errors()
    return objective

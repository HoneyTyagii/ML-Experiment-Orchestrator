"""Configuration loading for objectives and constraints.

Loads experiment objectives from YAML or JSON files into validated
:class:`~orchestrator.core.models.Objective` instances. This is what turns a
human-authored spec file into the structured input the pipeline consumes.

Example YAML
------------
.. code-block:: yaml

    name: tune-resnet
    description: Tune a ResNet classifier on CIFAR-10
    primary_metric: val_accuracy
    goal: maximize
    dataset: cifar10
    max_experiments: 25
    max_concurrency: 4
    target_metric_value: 0.93
    search_space:
      - name: learning_rate
        type: float
        low: 0.0001
        high: 0.1
        log_scale: true
      - name: optimizer
        type: categorical
        choices: [adam, sgd, adamw]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from orchestrator.core.models import HyperparameterSpec, Objective


class ConfigError(Exception):
    """Raised when a configuration file is missing, malformed, or invalid."""


def _read_raw(path: Path) -> dict[str, Any]:
    """Read a YAML or JSON file into a plain dictionary."""
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    try:
        if suffix in (".yaml", ".yml"):
            data = yaml.safe_load(text)
        elif suffix == ".json":
            data = json.loads(text)
        else:
            raise ConfigError(f"unsupported config format: {suffix or '<none>'}")
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ConfigError(f"failed to parse {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping, got {type(data).__name__}")

    return data


def objective_from_dict(data: dict[str, Any]) -> Objective:
    """Build and validate an :class:`Objective` from a raw mapping."""
    raw_space = data.get("search_space", []) or []
    if not isinstance(raw_space, list):
        raise ConfigError("'search_space' must be a list")

    specs: list[HyperparameterSpec] = []
    for entry in raw_space:
        if not isinstance(entry, dict):
            raise ConfigError("each search_space entry must be a mapping")
        try:
            specs.append(HyperparameterSpec(**entry))
        except Exception as exc:  # noqa: BLE001 - re-raised as ConfigError
            raise ConfigError(f"invalid hyperparameter spec {entry!r}: {exc}") from exc

    payload = {**data, "search_space": specs}
    try:
        objective = Objective(**payload)
    except Exception as exc:  # noqa: BLE001 - re-raised as ConfigError
        raise ConfigError(f"invalid objective configuration: {exc}") from exc

    try:
        objective.validate_search_space()
    except ValueError as exc:
        raise ConfigError(f"invalid search space: {exc}") from exc

    return objective


def load_objective(path: str | Path) -> Objective:
    """Load a single objective from a YAML or JSON file."""
    return objective_from_dict(_read_raw(Path(path)))

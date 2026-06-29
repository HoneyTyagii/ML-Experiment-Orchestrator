# Usage Guide

This guide walks through using the ML Experiment Orchestrator, from a one-call
end-to-end run down to composing the individual stages yourself.

- [Install](#install)
- [Quickstart](#quickstart)
- [Defining an objective](#defining-an-objective)
- [The training function](#the-training-function)
- [Running the full pipeline](#running-the-full-pipeline)
- [Using the stages directly](#using-the-stages-directly)
- [Search strategies](#search-strategies)
- [Early stopping](#early-stopping)
- [Comparison & reports](#comparison--reports)
- [Deployment](#deployment)
- [Integrations](#integrations)

---

## Install

```bash
pip install -e .                 # core (pydantic + pyyaml)
pip install -e ".[mlflow]"       # + MLflow tracking
pip install -e ".[wandb]"        # + Weights & Biases
pip install -e ".[kubeflow]"     # + Kubeflow Pipelines backend
pip install -e ".[dev]"          # + pytest / ruff
```

All third-party integrations are **optional** — importing the package never
requires them, and a clear error is raised only if you use a backend that isn't
installed.

## Quickstart

```python
from orchestrator.core import run_pipeline

def train(experiment):
    hp = experiment.hyperparameters
    # ... train a model with these hyperparameters ...
    return {"val_accuracy": 0.91, "loss": 0.2}

result = run_pipeline("examples/objectives/tune_resnet.yaml", train, strategy="random")

print(result.report.to_markdown())
print("best:", result.best_value, "->", result.best_experiment.hyperparameters)
print("deployed at:", result.deployment.uri)
```

`run_pipeline` runs every stage: intake → generate → launch → monitor → adjust →
compare → report → deploy.

## Defining an objective

An objective is the high-level goal. Author it as YAML/JSON or build it in code.

```yaml
# examples/objectives/tune_resnet.yaml
name: tune-resnet
description: Tune a ResNet classifier on CIFAR-10
primary_metric: val_accuracy
goal: maximize                 # or: minimize
dataset: cifar10
max_experiments: 25            # total budget
max_concurrency: 4             # how many run at once
target_metric_value: 0.93      # optional early-finish target
search_space:
  - {name: learning_rate, type: float, low: 0.0001, high: 0.1, log_scale: true}
  - {name: batch_size,    type: int,   low: 32,     high: 256}
  - {name: optimizer,     type: categorical, choices: [adam, sgd, adamw]}
  - {name: use_augmentation, type: bool}
```

Load and validate it (intake checks for blank names, duplicate hyperparameters,
log-scale ranges that include zero, concurrency > budget, etc.):

```python
from orchestrator.core import intake_objective, validate_objective

objective = intake_objective("examples/objectives/tune_resnet.yaml")

# inspect issues without raising:
report = validate_objective(objective)
for issue in report.issues:
    print(issue)          # e.g. "[warning] max_concurrency: ... will be capped"
```

## The training function

You supply a callable that trains one configuration and returns its metrics.
Two shapes are supported; the local backend detects which by its signature.

```python
# 1. simple: return final metrics
def train(experiment):
    return {"val_accuracy": 0.9, "loss": 0.1}

# 2. streaming: report intermediate metrics and honor cancellation
def train(experiment, ctx):
    for epoch in range(10):
        if ctx.should_stop():          # set by early stopping / cancel
            return None
        acc = train_one_epoch(experiment.hyperparameters)
        ctx.report("val_accuracy", acc, step=epoch)
    return {"val_accuracy": acc}
```

## Running the full pipeline

For more control than `run_pipeline`, configure an `Orchestrator`:

```python
from orchestrator.core import Orchestrator, GridSearch, EarlyStopper, LocalDeploymentTarget

orch = Orchestrator(
    strategy=GridSearch(points_per_float=5),     # or "random" / "grid"
    stopper=None,                                # default derived from objective
    deploy=True,
    deploy_target=LocalDeploymentTarget("deployments/"),
    secondary=[("loss", "minimize")],            # tie-breakers for ranking/selection
    require_target_for_deploy=False,
    min_score=0.9,                               # only deploy if best >= 0.9
)

result = orch.run(objective, train)
```

You can also pass your own launcher (e.g. Kubeflow) instead of a `train_fn`:

```python
result = orch.run(objective, launcher=my_launcher)
```

## Using the stages directly

The pipeline is just these pieces wired together — use them à la carte:

```python
from orchestrator.core import (
    LocalLauncher, TuningLoop, EarlyStopper,
    rank_result, build_report, deploy_best, LocalDeploymentTarget,
)

with LocalLauncher(train, max_workers=objective.max_concurrency) as launcher:
    loop = TuningLoop(objective, launcher,
                      strategy="random",
                      stopper=EarlyStopper.from_objective(objective, patience=5))
    tuning = loop.run()

leaderboard = rank_result(tuning, secondary=[("loss", "minimize")])
report = build_report(tuning, objective=objective)
deployment = deploy_best(tuning, LocalDeploymentTarget())
```

## Search strategies

| Name     | Class          | Notes                                                   |
|----------|----------------|---------------------------------------------------------|
| `random` | `RandomSearch` | Independent uniform samples; seedable (`RandomSearch(seed=0)`). |
| `grid`   | `GridSearch`   | Cartesian product; floats discretized (`points_per_float`); lazily enumerated. |

Resolve by name with `get_strategy("random")`, or list with
`available_strategies()`. Implement your own by subclassing
`ExperimentStrategy` and decorating it with `@register_strategy`; `propose`
receives the run history, so adaptive strategies can steer the search.

## Early stopping

Compose policies and feed them to the loop:

```python
from orchestrator.core import (
    EarlyStopper, TargetThresholdPolicy, PatiencePolicy,
    DivergencePolicy, FloorThresholdPolicy, MaxStepsPolicy, Goal,
)

stopper = EarlyStopper([
    TargetThresholdPolicy("val_accuracy", Goal.MAXIMIZE, 0.93),  # stop on success
    PatiencePolicy("val_accuracy", Goal.MAXIMIZE, patience=5),   # stop on plateau
    FloorThresholdPolicy("val_accuracy", Goal.MAXIMIZE, 0.4, after_step=2),  # prune
    DivergencePolicy("val_accuracy"),                            # stop on NaN/inf
])
# or: EarlyStopper.from_objective(objective, patience=5)
```

## Comparison & reports

```python
leaderboard = rank_result(tuning)
print(leaderboard.best.experiment.hyperparameters)
for entry in leaderboard.top(5):
    print(entry.rank, entry.score, entry.experiment.name)

report = build_report(tuning, objective=objective, top_n=10)
report.to_markdown(); report.to_text(); report.to_json()
# or in one call, written to disk (format inferred from extension):
from orchestrator.core import write_report
write_report(tuning, "REPORT.md", objective=objective)
```

## Deployment

```python
from orchestrator.core import select_best, deploy_best, LocalDeploymentTarget

best = select_best(tuning, min_score=0.9)          # None if nothing qualifies
deployment = deploy_best(tuning, "local")          # by registered name
deployment = deploy_best(tuning, LocalDeploymentTarget("deployments/"))
```

Implement a custom serving target by subclassing `DeploymentTarget` and
decorating with `@register_target`.

## Integrations

All optional. Wire trackers as monitor listeners (live) or log a finished result.

```python
# MLflow
from orchestrator.integrations.mlflow import MlflowTracker, MlflowListener
tracker = MlflowTracker.for_objective(objective, tracking_uri="http://localhost:5000")
orch = Orchestrator(listeners=[MlflowListener(tracker, [])])   # live logging
# ... or after a run:
tracker.log_result(tuning)

# Weights & Biases
from orchestrator.integrations.wandb import WandbTracker
WandbTracker.for_objective(objective, entity="my-team").log_result(tuning)

# Kubeflow (a launcher backend)
from orchestrator.integrations.kubeflow import KubeflowLauncher
launcher = KubeflowLauncher(pipeline="train_pipeline.yaml", experiment_name="resnet")
result = Orchestrator().run(objective, launcher=launcher)
```

See [architecture.md](architecture.md) for how the pieces fit together.

# ML Experiment Orchestrator

An autonomous agent that orchestrates the full machine learning experimentation lifecycle: from objective intake to deploying the best-performing model.

## What it does

Given a high-level objective, the agent:

1. **Receives objective** — parses the goal, dataset, and constraints
2. **Generates experiments** — proposes hyperparameter configurations and model variants
3. **Launches training** — schedules and runs training jobs
4. **Monitors metrics** — streams live metrics during training
5. **Adjusts hyperparameters** — adapts the search based on intermediate results
6. **Compares results** — ranks runs against the objective
7. **Generates reports** — produces human-readable summaries and visualizations
8. **Deploys best model** — promotes the top model to a serving target

## Architecture

```
Receive objective
      ↓
Generate experiments
      ↓
Launch training
      ↓
Monitor metrics
      ↓
Adjust hyperparameters
      ↓
Compare results
      ↓
Generate reports
      ↓
Deploy best model
```

## Integrations

- **MLflow** — experiment tracking and model registry
- **Weights & Biases** — metric logging and visualization
- **Kubeflow** — distributed training orchestration

All integrations are optional extras; the core has no third-party runtime
dependencies beyond `pydantic` and `pyyaml`.

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
```

## Documentation

- **[Usage guide](docs/usage.md)** — install, objectives, training functions, the
  pipeline, each stage à la carte, strategies, early stopping, reports,
  deployment, and integrations.
- **[Architecture](docs/architecture.md)** — the eight-stage lifecycle, module
  map, and design principles (registries, launcher lifecycle, the feedback loop,
  optional dependencies).

## Status

Stages 1–8 implemented (intake → generate → launch → monitor → adjust → compare
→ report → deploy), with MLflow, Weights & Biases, and Kubeflow integrations and
a unit-test suite. Install dev extras and run the tests with:

```bash
pip install -e ".[dev]"
pytest
```


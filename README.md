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

## Status

Early scaffold. See the project board for progress.

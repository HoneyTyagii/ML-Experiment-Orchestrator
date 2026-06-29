"""Unit tests for the adaptive tuning loop and end-to-end pipeline."""

from __future__ import annotations

from orchestrator.core import (
    DeploymentStatus,
    ExperimentStatus,
    Goal,
    GridSearch,
    LocalLauncher,
    Orchestrator,
    TuningLoop,
    run_local,
    run_pipeline,
)


def _score_by_batch(experiment):
    """Deterministic metric: higher batch_size -> higher accuracy."""
    bs = experiment.hyperparameters.get("batch_size", 16)
    return {"val_accuracy": bs / 64.0, "loss": 1.0 - bs / 64.0}


def test_loop_runs_exactly_the_budget(objective):
    obj = objective.model_copy(update={"max_experiments": 6, "max_concurrency": 2})
    res = run_local(obj, _score_by_batch, strategy="random")
    assert len(res.experiments) == 6
    assert res.rounds == 3  # 6 experiments / concurrency 2
    assert all(e.status is ExperimentStatus.COMPLETED for e in res.experiments)


def test_loop_selects_the_best_experiment(objective):
    obj = objective.model_copy(update={"max_experiments": 5, "max_concurrency": 1})
    res = run_local(obj, _score_by_batch, strategy="random")
    expected = max(e.best_metric("val_accuracy", Goal.MAXIMIZE) for e in res.experiments)
    assert res.best_value == expected
    assert res.best_experiment is not None
    assert res.best_experiment.best_metric("val_accuracy", Goal.MAXIMIZE) == expected


def test_loop_stops_when_target_reached(objective):
    # target is trivially reachable; loop should stop before exhausting a big budget
    obj = objective.model_copy(
        update={"max_experiments": 50, "max_concurrency": 1, "target_metric_value": 0.1}
    )
    res = run_local(obj, _score_by_batch, strategy="random")
    assert res.target_reached
    assert len(res.experiments) < 50
    assert res.best_value >= 0.1


def test_loop_stops_when_grid_exhausted(categorical_objective):
    # 2x2 grid => 4 experiments even though budget is 50
    res = run_local(categorical_objective, lambda e: {"val_accuracy": 0.5}, strategy="grid")
    assert len(res.experiments) == 4


def test_loop_survives_training_failures(objective):
    def flaky(experiment):
        if experiment.hyperparameters.get("optimizer") == "sgd":
            raise RuntimeError("bad config")
        return {"val_accuracy": 0.7}

    obj = objective.model_copy(update={"max_experiments": 6, "max_concurrency": 2})
    res = run_local(obj, flaky, strategy="random")
    assert len(res.experiments) == 6  # loop did not crash
    statuses = {e.status for e in res.experiments}
    assert statuses <= {ExperimentStatus.COMPLETED, ExperimentStatus.FAILED}
    # best ignores failed runs
    assert res.best_value is None or res.best_value == 0.7


def test_explicit_loop_with_launcher_and_grid(objective):
    obj = objective.model_copy(update={"max_experiments": 4, "max_concurrency": 2})
    rounds = []
    with LocalLauncher(_score_by_batch, max_workers=2) as launcher:
        loop = TuningLoop(
            obj, launcher, strategy=GridSearch(points_per_float=2),
            on_round=lambda i, exps: rounds.append((i, len(exps))),
        )
        res = loop.run()
    assert len(res.experiments) == 4
    assert rounds == [(1, 2), (2, 2)]


def test_pipeline_end_to_end_deploys_best(objective):
    obj = objective.model_copy(update={"max_experiments": 4, "max_concurrency": 2})
    result = run_pipeline(obj, _score_by_batch, strategy="grid")
    # every stage produced output
    assert result.objective.name == "test-obj"
    assert len(result.tuning.experiments) > 0
    assert result.leaderboard.best is not None
    assert "# Experiment Report" in result.report.to_markdown()
    assert result.deployment is not None
    assert result.deployment.status is DeploymentStatus.DEPLOYED
    assert result.deployment.experiment_id == result.best_experiment.id


def test_pipeline_can_skip_deployment(objective):
    obj = objective.model_copy(update={"max_experiments": 3, "max_concurrency": 1})
    orch = Orchestrator(strategy="random", deploy=False)
    result = orch.run(obj, _score_by_batch)
    assert result.deployment is None
    assert result.leaderboard.best is not None

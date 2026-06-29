"""Unit tests for the run comparison and ranking engine."""

from __future__ import annotations

import pytest

from orchestrator.core import (
    ExperimentStatus,
    Goal,
    compare_experiments,
    rank_experiments,
    score_experiment,
)

from .conftest import make_experiment


def test_ranks_best_first_for_maximize():
    a = make_experiment("a", val_accuracy=[0.7, 0.8])
    b = make_experiment("b", val_accuracy=[0.9, 0.85])
    c = make_experiment("c", val_accuracy=[0.6])
    board = rank_experiments([a, b, c], "val_accuracy", Goal.MAXIMIZE)
    assert [e.experiment.name for e in board] == ["b", "a", "c"]
    assert [e.rank for e in board] == [1, 2, 3]
    assert board.best.experiment.name == "b"
    assert board.best.is_best
    assert board.best.score == 0.9


def test_ranks_best_first_for_minimize():
    a = make_experiment("a", loss=[0.3, 0.2])
    b = make_experiment("b", loss=[0.5, 0.1])
    board = rank_experiments([a, b], "loss", Goal.MINIMIZE)
    assert board.best.experiment.name == "b"  # best (lowest) loss 0.1
    assert board.best.score == 0.1


def test_secondary_breaks_ties():
    x = make_experiment("x", val_accuracy=[0.9], loss=[0.5])
    y = make_experiment("y", val_accuracy=[0.9], loss=[0.1])
    board = rank_experiments(
        [x, y], "val_accuracy", Goal.MAXIMIZE, secondary=[("loss", Goal.MINIMIZE)]
    )
    assert [e.experiment.name for e in board] == ["y", "x"]


def test_full_tie_preserves_input_order():
    p = make_experiment("p", val_accuracy=[0.5])
    q = make_experiment("q", val_accuracy=[0.5])
    board = rank_experiments([p, q], "val_accuracy", Goal.MAXIMIZE)
    assert [e.experiment.name for e in board] == ["p", "q"]


def test_experiments_without_metric_are_unranked():
    a = make_experiment("a", val_accuracy=[0.7])
    d = make_experiment("d")  # no metrics
    board = rank_experiments([a, d], "val_accuracy", Goal.MAXIMIZE)
    assert [e.experiment.name for e in board] == ["a"]
    assert [e.name for e in board.unranked] == ["d"]


def test_status_filter_excludes_non_matching():
    a = make_experiment("a", val_accuracy=[0.7])
    b = make_experiment("b", val_accuracy=[0.9])
    b.status = ExperimentStatus.FAILED
    board = rank_experiments(
        [a, b], "val_accuracy", Goal.MAXIMIZE, statuses=[ExperimentStatus.COMPLETED]
    )
    assert [e.experiment.name for e in board] == ["a"]
    assert [e.name for e in board.unranked] == ["b"]


def test_method_last_uses_latest_value():
    a = make_experiment("a", val_accuracy=[0.9, 0.5])  # best 0.9, last 0.5
    b = make_experiment("b", val_accuracy=[0.6, 0.7])  # best 0.7, last 0.7
    by_best = rank_experiments([a, b], "val_accuracy", Goal.MAXIMIZE, method="best")
    by_last = rank_experiments([a, b], "val_accuracy", Goal.MAXIMIZE, method="last")
    assert by_best.best.experiment.name == "a"
    assert by_last.best.experiment.name == "b"


def test_summary_statistics():
    a = make_experiment("a", val_accuracy=[0.8])
    b = make_experiment("b", val_accuracy=[0.9])
    c = make_experiment("c", val_accuracy=[0.7])
    summary = rank_experiments([a, b, c], "val_accuracy", Goal.MAXIMIZE).summary()
    assert summary.count == 3
    assert summary.best == 0.9
    assert summary.worst == 0.7
    assert summary.mean == pytest.approx(0.8)
    assert summary.median == pytest.approx(0.8)


def test_rows_are_serializable():
    a = make_experiment("a", val_accuracy=[0.8])
    a.hyperparameters = {"lr": 0.01}
    rows = rank_experiments([a], "val_accuracy", Goal.MAXIMIZE).rows()
    assert rows[0]["rank"] == 1
    assert rows[0]["score"] == 0.8
    assert rows[0]["hp.lr"] == 0.01


def test_compare_experiments_three_way():
    a = make_experiment("a", val_accuracy=[0.8])
    b = make_experiment("b", val_accuracy=[0.9])
    assert compare_experiments(a, b, "val_accuracy", Goal.MAXIMIZE) == 1   # a worse
    assert compare_experiments(b, a, "val_accuracy", Goal.MAXIMIZE) == -1  # b better
    assert compare_experiments(a, a, "val_accuracy", Goal.MAXIMIZE) == 0


def test_score_experiment_missing_returns_none():
    d = make_experiment("d")
    assert score_experiment(d, "val_accuracy", Goal.MAXIMIZE) is None

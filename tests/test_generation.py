"""Unit tests for experiment generation strategies (random + grid search)."""

from __future__ import annotations

import pytest

from orchestrator.core import (
    GridSearch,
    Objective,
    RandomSearch,
    StrategyError,
    available_strategies,
    get_strategy,
    intake_objective,
)


def test_strategies_are_registered():
    assert {"random", "grid"} <= set(available_strategies())
    assert isinstance(get_strategy("random"), RandomSearch)
    assert isinstance(get_strategy("grid"), GridSearch)


def test_unknown_strategy_raises():
    with pytest.raises(StrategyError):
        get_strategy("does-not-exist")


def test_random_is_reproducible_with_seed(objective):
    a = RandomSearch(seed=123).propose(objective, count=5)
    b = RandomSearch(seed=123).propose(objective, count=5)
    assert [e.hyperparameters for e in a] == [e.hyperparameters for e in b]


def test_random_differs_across_seeds(objective):
    a = RandomSearch(seed=1).propose(objective, count=5)
    b = RandomSearch(seed=2).propose(objective, count=5)
    assert [e.hyperparameters for e in a] != [e.hyperparameters for e in b]


def test_random_respects_search_space(objective):
    for exp in RandomSearch(seed=0).propose(objective, count=25):
        hp = exp.hyperparameters
        assert 0.001 <= hp["learning_rate"] <= 0.1
        assert isinstance(hp["batch_size"], int)
        assert 16 <= hp["batch_size"] <= 64
        assert hp["optimizer"] in ("adam", "sgd")
        assert isinstance(hp["augment"], bool)


def test_random_count_validation(objective):
    with pytest.raises(StrategyError):
        RandomSearch().propose(objective, count=0)


def test_random_links_experiments_to_objective(objective):
    exps = RandomSearch(seed=0).propose(objective, count=3)
    assert all(e.objective_id == objective.id for e in exps)
    assert [e.name for e in exps] == ["test-obj-0000", "test-obj-0001", "test-obj-0002"]


def test_random_history_continues_numbering(objective):
    first = RandomSearch(seed=0).propose(objective, count=2)
    nxt = RandomSearch(seed=0).propose(objective, count=2, history=first)
    assert [e.name for e in nxt] == ["test-obj-0002", "test-obj-0003"]


def test_grid_is_deterministic_and_continues(categorical_objective):
    grid = GridSearch()
    first = grid.propose(categorical_objective, count=2)
    nxt = grid.propose(categorical_objective, count=2, history=first)
    # stable order: same inputs -> same first batch
    assert [e.hyperparameters for e in first] == [
        e.hyperparameters for e in GridSearch().propose(categorical_objective, count=2)
    ]
    assert [e.name for e in nxt] == ["tiny-0002", "tiny-0003"]


def test_grid_enumerates_full_product_then_stops(categorical_objective):
    grid = GridSearch()
    seen = []
    history: list = []
    while True:
        batch = grid.propose(categorical_objective, count=10, history=history)
        if not batch:
            break
        seen.extend(batch)
        history = seen
    combos = {(e.hyperparameters["opt"], e.hyperparameters["flag"]) for e in seen}
    assert len(seen) == 4
    assert combos == {("a", True), ("a", False), ("b", True), ("b", False)}
    # exhausted -> empty
    assert grid.propose(categorical_objective, count=5, history=seen) == []


def test_grid_discretizes_floats(objective):
    grid = GridSearch(points_per_float=3)
    values = grid._float_points(objective.search_space[0])  # learning_rate, log-scaled
    assert len(values) == 3
    assert values[0] == pytest.approx(0.001)
    assert values[-1] == pytest.approx(0.1)
    # geometric spacing for log scale -> midpoint is the geometric mean
    assert values[1] == pytest.approx((0.001 * 0.1) ** 0.5)


@pytest.mark.parametrize("strategy_cls", [RandomSearch, GridSearch])
def test_empty_search_space_yields_single_default(strategy_cls):
    obj = Objective(name="empty", primary_metric="acc")
    strat = strategy_cls()
    first = strat.propose(obj, count=5)
    assert len(first) == 1
    assert first[0].hyperparameters == {}
    # already produced -> nothing more
    assert strat.propose(obj, count=5, history=first) == []


def test_invalid_search_space_raises(objective):
    bad = intake_objective(objective.model_dump())
    # corrupt a spec post-validation: low > high
    bad.search_space[1].low = 100.0
    bad.search_space[1].high = 1.0
    with pytest.raises(StrategyError):
        RandomSearch().propose(bad, count=1)

"""Decision and cost logic in src/scoring.py."""

import numpy as np

from src.scoring import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    HIGH_RISK,
    LOW_RISK,
    choose_threshold,
    cost_table,
    decide,
    expected_cost,
)


def test_decide_uses_threshold_inclusively():
    assert decide(0.10, 0.32) == LOW_RISK
    assert decide(0.32, 0.32) == HIGH_RISK  # at the threshold counts as high risk
    assert decide(0.90, 0.32) == HIGH_RISK


def test_expected_cost_counts_false_negatives_and_false_positives():
    y_true = [1, 1, 0, 0]
    y_pred = [0, 1, 1, 0]  # one missed bad loan, one wrongly rejected good applicant
    assert expected_cost(y_true, y_pred) == COST_FALSE_NEGATIVE + COST_FALSE_POSITIVE
    assert expected_cost(y_true, y_pred, cost_fn=10, cost_fp=2) == 12
    assert expected_cost(y_true, y_true) == 0


def test_cost_table_covers_the_grid():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.6, 0.4])
    table = cost_table(y, p)
    assert list(table.columns) == ["threshold", "false_negatives", "false_positives", "rejected_share", "cost"]
    assert table["threshold"].min() == 0.05 and table["threshold"].max() == 0.95
    assert (table["cost"] >= 0).all()


def test_choose_threshold_lowers_cost_below_default():
    """Missing a default costs 5x a false alarm, so the best threshold sits below 0.5 here."""
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.45, 0.60, 0.30, 0.20, 0.10, 0.35, 0.05, 0.15, 0.25, 0.40])
    threshold, table = choose_threshold(y, p)
    cost_default = table.loc[table["threshold"] == 0.5, "cost"].iloc[0]
    cost_chosen = table.loc[table["threshold"] == threshold, "cost"].iloc[0]
    assert threshold < 0.5
    assert cost_chosen < cost_default
    assert cost_chosen == table["cost"].min()


def test_choose_threshold_breaks_ties_towards_fewer_rejections():
    # Perfect separation: every threshold in (0.2, 0.8] costs 0, and the highest one wins.
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    threshold, _ = choose_threshold(y, p)
    assert threshold == 0.8

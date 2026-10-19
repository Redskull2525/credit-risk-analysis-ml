"""Turning a predicted probability into a lending decision.

This module is shared by training (src/train.py), the Streamlit app (app.py)
and the tests, so the threshold logic exists in exactly one place.

Cost assumptions (ASSUMPTIONS, not measured values)
---------------------------------------------------
The German Credit dataset ships with a cost matrix from its authors:
"It is worse to class a customer as good when they are bad (5), than it is
to class a customer as bad when they are good (1)."

* False negative  = a bad applicant approved  -> the bank loses the money lent.  Cost 5.
* False positive  = a good applicant rejected -> the bank loses some interest.   Cost 1.

Real costs depend on loan size, interest margin, recovery rates and so on.
The 5:1 ratio is only a reasonable, documented starting point.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "credit_pipeline.pkl"

COST_FALSE_NEGATIVE = 5  # bad applicant approved (missed default)
COST_FALSE_POSITIVE = 1  # good applicant rejected (lost business)

HIGH_RISK = "high_risk"
LOW_RISK = "low_risk"

# Candidate thresholds to search when choosing the operating point.
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def expected_cost(
    y_true,
    y_pred,
    cost_fn: float = COST_FALSE_NEGATIVE,
    cost_fp: float = COST_FALSE_POSITIVE,
) -> float:
    """Total cost of a set of decisions under the cost assumptions above.

    y_true / y_pred use 1 = bad (default), 0 = good.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    false_negatives = int(((y_true == 1) & (y_pred == 0)).sum())
    false_positives = int(((y_true == 0) & (y_pred == 1)).sum())
    return float(false_negatives * cost_fn + false_positives * cost_fp)


def cost_table(y_true, probabilities, thresholds=THRESHOLD_GRID) -> pd.DataFrame:
    """Cost, false negatives and false positives for every candidate threshold."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    rows = []
    for t in thresholds:
        y_pred = (probabilities >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "false_negatives": int(((y_true == 1) & (y_pred == 0)).sum()),
                "false_positives": int(((y_true == 0) & (y_pred == 1)).sum()),
                "rejected_share": float(y_pred.mean()),
                "cost": expected_cost(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(y_true, probabilities, thresholds=THRESHOLD_GRID) -> tuple[float, pd.DataFrame]:
    """Pick the threshold with the lowest expected cost.

    If several thresholds tie, the highest one wins: same cost, fewer
    rejected applicants. Returns (threshold, full cost table).
    """
    table = cost_table(y_true, probabilities, thresholds)
    best_cost = table["cost"].min()
    best = table.loc[table["cost"] == best_cost, "threshold"].max()
    return float(best), table


def decide(probability: float, threshold: float) -> str:
    """HIGH_RISK when the probability of default reaches the threshold, else LOW_RISK."""
    return HIGH_RISK if probability >= threshold else LOW_RISK


def load_bundle(path: Path = MODEL_PATH) -> dict:
    """Load the saved pipeline + metadata written by src/train.py."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run `python src/train.py` first.")
    return joblib.load(path)

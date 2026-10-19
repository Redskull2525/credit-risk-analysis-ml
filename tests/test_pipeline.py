"""The training pipeline: feature lists are consistent, it learns something, it does not crash on new categories."""

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, PROTECTED_ATTRIBUTES, TARGET, split_xy
from src.train import build_pipeline, candidate_models


def test_feature_lists_are_consistent(data):
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)
    assert len(FEATURES) == len(set(FEATURES)), "duplicate feature names"
    assert set(FEATURES) <= set(data.columns), "a listed feature is missing from the CSV"
    assert set(PROTECTED_ATTRIBUTES).isdisjoint(FEATURES), "protected attributes must not be model inputs"
    assert TARGET not in FEATURES


def test_data_has_expected_shape_and_labels(data):
    assert data.shape == (1000, 21)
    assert set(data[TARGET].unique()) == {"good", "bad"}
    assert data[FEATURES].isna().sum().sum() == 0


def test_pipeline_trains_and_beats_chance_on_small_fixture(data):
    """A 300-row sample is enough to learn a signal: ROC-AUC must clear 0.5 (random) with margin."""
    sample = data.sample(300, random_state=0)
    X, y = split_xy(sample)
    pipeline = build_pipeline(candidate_models()["logistic_regression"])
    scores = cross_val_score(
        pipeline, X, y, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0), scoring="roc_auc"
    )
    assert scores.mean() > 0.6, f"ROC-AUC {scores.mean():.3f} is barely better than chance"


def test_pipeline_outputs_valid_probabilities(data):
    X, y = split_xy(data.head(300))
    pipeline = build_pipeline(candidate_models()["random_forest"]).fit(X, y)
    proba = pipeline.predict_proba(X.head(10))
    assert proba.shape == (10, 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_pipeline_handles_unseen_category(data):
    """handle_unknown='ignore' means a category never seen in training does not crash the app."""
    X, y = split_xy(data.head(300))
    pipeline = build_pipeline(candidate_models()["logistic_regression"]).fit(X, y)
    row = X.head(1).copy()
    row["purpose"] = "spaceship"
    proba = pipeline.predict_proba(row)[0, 1]
    assert 0.0 <= proba <= 1.0

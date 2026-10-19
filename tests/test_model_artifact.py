"""The saved model bundle: it loads, its feature list matches the code, and its metrics are reproducible."""

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

from src.data import FEATURES, split_xy
from src.train import split_data

REQUIRED_KEYS = ["pipeline", "model_name", "feature_names", "features", "threshold", "costs", "metrics",
                 "sklearn_version", "python_version", "trained_at"]


def test_bundle_has_required_metadata(bundle):
    missing = [k for k in REQUIRED_KEYS if k not in bundle]
    assert not missing, f"bundle is missing {missing}"


def test_saved_feature_list_matches_pipeline_and_source(bundle):
    """The form (built from bundle['features']), the pipeline and src/data.py must agree exactly."""
    form_features = [f["name"] for f in bundle["features"]]
    assert form_features == bundle["feature_names"] == FEATURES
    assert list(bundle["pipeline"].feature_names_in_) == FEATURES


def test_feature_spec_is_complete(bundle):
    for feat in bundle["features"]:
        assert feat["label"], feat["name"]
        if feat["type"] == "numeric":
            assert feat["min"] <= feat["default"] <= feat["max"], feat["name"]
        else:
            assert feat["default"] in feat["categories"], feat["name"]
            assert len(feat["categories"]) >= 2, feat["name"]


def test_saved_model_loads_and_predicts(bundle, data):
    X, _ = split_xy(data.head(5))
    proba = bundle["pipeline"].predict_proba(X)
    assert proba.shape == (5, 2)
    assert np.all((proba >= 0) & (proba <= 1))


def test_saved_metrics_are_reproducible(bundle, data):
    """Rebuild the test split with the same seed; the stored ROC-AUC must match the saved model's output now."""
    _, df_test = split_data(data)
    X_test, y_test = split_xy(df_test)
    auc_now = roc_auc_score(y_test, bundle["pipeline"].predict_proba(X_test)[:, 1])
    assert abs(auc_now - bundle["metrics"]["at_chosen_threshold"]["roc_auc"]) < 1e-6


def test_threshold_is_in_search_range(bundle):
    assert 0.05 <= bundle["threshold"] <= 0.95


def test_sklearn_version_matches_artifact(bundle):
    """A version mismatch between training and serving is exactly what broke v1 of this repo."""
    assert bundle["sklearn_version"] == sklearn.__version__

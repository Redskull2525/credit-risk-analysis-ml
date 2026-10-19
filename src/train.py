"""Train, compare and evaluate credit-risk models on the German Credit data.

Run from the repository root:

    python src/train.py

What happens, in order:

1. Load data/german_credit.csv (download it first with `python src/data.py`).
2. Stratified train/test split (80/20, random_state=42). The test set is
   used exactly once, at the very end.
3. Preprocess inside a Pipeline: StandardScaler for numeric columns,
   OneHotEncoder for categorical columns.
4. Compare Logistic Regression vs Random Forest with 5-fold stratified
   cross-validation on the TRAINING set and keep the best mean ROC-AUC.
5. Choose the decision threshold that minimises expected cost on
   out-of-fold training predictions (never on the test set).
6. Fit the winner on the full training set, evaluate once on the test set.
7. Save pipeline + metadata to models/credit_pipeline.pkl, metrics to
   reports/metrics.json, a model card to reports/model_card.md and the
   logistic-regression coefficients to reports/logistic_coefficients.csv.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # so `python src/train.py` can import the src package
    sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402  (import after sys.path fix)
    CATEGORICAL_FEATURES,
    FEATURE_DESCRIPTIONS,
    FEATURES,
    NUMERIC_FEATURES,
    POSITIVE_LABEL,
    PROTECTED_ATTRIBUTES,
    PROTECTED_CATEGORICAL,
    PROTECTED_NUMERIC,
    TARGET,
    load_data,
    split_xy,
)
from src.scoring import (  # noqa: E402
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    MODEL_PATH,
    choose_threshold,
    expected_cost,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5
DEFAULT_THRESHOLD = 0.5  # what you get if you never think about thresholds

REPORTS_DIR = ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"
MODEL_CARD_PATH = REPORTS_DIR / "model_card.md"
COEFFICIENTS_PATH = REPORTS_DIR / "logistic_coefficients.csv"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def build_preprocessor(numeric=NUMERIC_FEATURES, categorical=CATEGORICAL_FEATURES) -> ColumnTransformer:
    """Scale numeric columns, one-hot encode categorical columns.

    handle_unknown="ignore" means a category never seen in training becomes
    all zeros instead of crashing the app.
    """
    return ColumnTransformer(
        [
            ("num", StandardScaler(), list(numeric)),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(categorical)),
        ]
    )


def candidate_models() -> dict[str, object]:
    """The two models we compare. Both use class_weight='balanced' because
    only 30% of applicants are bad; without it the models lean towards
    predicting 'good' for everyone."""
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,  # with only 800 training rows, fully grown trees overfit
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def build_pipeline(model, numeric=NUMERIC_FEATURES, categorical=CATEGORICAL_FEATURES) -> Pipeline:
    """Preprocessing + model in one object, so the app can call predict_proba on raw inputs."""
    return Pipeline([("preprocess", build_preprocessor(numeric, categorical)), ("model", model)])


def make_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split on the full dataframe (keeps protected columns for the ablation)."""
    return train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET])


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def cross_validate_candidates(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """5-fold stratified CV of every candidate on the training set only."""
    results = {}
    for name, model in candidate_models().items():
        scores = cross_validate(
            build_pipeline(model), X_train, y_train, cv=make_cv(),
            scoring=["roc_auc", "average_precision"],
        )
        results[name] = {
            "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "cv_roc_auc_std": float(scores["test_roc_auc"].std()),
            "cv_roc_auc_folds": [round(float(s), 4) for s in scores["test_roc_auc"]],
            "cv_pr_auc_mean": float(scores["test_average_precision"].mean()),
            "cv_pr_auc_std": float(scores["test_average_precision"].std()),
        }
        print(
            f"  {name:20s} ROC-AUC {results[name]['cv_roc_auc_mean']:.3f} "
            f"+/- {results[name]['cv_roc_auc_std']:.3f} | "
            f"PR-AUC {results[name]['cv_pr_auc_mean']:.3f}"
        )
    return results


def evaluate(y_true, probabilities, threshold: float) -> dict:
    """All the numbers we report for one operating point."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    y_pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_bad": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_bad": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_bad": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "rejected_share": float(y_pred.mean()),
        "expected_cost": expected_cost(y_true, y_pred),
        "expected_cost_per_applicant": expected_cost(y_true, y_pred) / len(y_true),
    }


def baseline_costs(y_true) -> dict:
    """What the cost model says about the two 'no model' strategies."""
    y_true = np.asarray(y_true, dtype=int)
    n_bad, n_good = int(y_true.sum()), int((y_true == 0).sum())
    return {
        "approve_everyone": {
            "accuracy": n_good / len(y_true),
            "recall_bad": 0.0,
            "expected_cost": float(n_bad * COST_FALSE_NEGATIVE),
        },
        "reject_everyone": {
            "accuracy": n_bad / len(y_true),
            "recall_bad": 1.0,
            "expected_cost": float(n_good * COST_FALSE_POSITIVE),
        },
    }


def protected_attribute_ablation(df_train: pd.DataFrame, model_name: str) -> dict:
    """CV ROC-AUC of the chosen model with and without the protected attributes.

    This quantifies the price of the fairness decision instead of hand-waving it.
    """
    y_train = (df_train[TARGET] == POSITIVE_LABEL).astype(int)
    out = {}
    for label, numeric, categorical in [
        ("without_protected", NUMERIC_FEATURES, CATEGORICAL_FEATURES),
        ("with_protected", NUMERIC_FEATURES + PROTECTED_NUMERIC, CATEGORICAL_FEATURES + PROTECTED_CATEGORICAL),
    ]:
        pipe = build_pipeline(candidate_models()[model_name], numeric, categorical)
        scores = cross_validate(pipe, df_train[numeric + categorical], y_train, cv=make_cv(), scoring="roc_auc")
        out[label] = {
            "cv_roc_auc_mean": float(scores["test_score"].mean()),
            "cv_roc_auc_std": float(scores["test_score"].std()),
            "features": numeric + categorical,
        }
    out["roc_auc_difference"] = out["with_protected"]["cv_roc_auc_mean"] - out["without_protected"]["cv_roc_auc_mean"]
    return out


def logistic_coefficients(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Fit the logistic-regression candidate and return its coefficients as odds ratios.

    Numeric features are standardised, so a coefficient is the change in log-odds
    of default for +1 standard deviation. One-hot coefficients are best compared
    within the same original feature (e.g. the four checking_status levels).
    """
    pipe = build_pipeline(candidate_models()["logistic_regression"]).fit(X_train, y_train)
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    coefs = pipe.named_steps["model"].coef_[0]
    table = pd.DataFrame(
        {
            "feature": [n.split("__", 1)[1] for n in names],  # drop the 'num__' / 'cat__' prefix
            "coefficient": coefs,
            "odds_ratio": np.exp(coefs),
        }
    )
    table["abs_coefficient"] = table["coefficient"].abs()
    return table.sort_values("abs_coefficient", ascending=False).drop(columns="abs_coefficient").reset_index(drop=True)


def describe_features(X_train: pd.DataFrame) -> list[dict]:
    """Feature spec saved with the model; the Streamlit form is generated from it."""
    spec = []
    for col in NUMERIC_FEATURES:
        s = X_train[col]
        spec.append(
            {
                "name": col,
                "type": "numeric",
                "label": FEATURE_DESCRIPTIONS[col],
                "min": int(s.min()),
                "max": int(s.max()),
                "default": int(round(float(s.median()))),
            }
        )
    for col in CATEGORICAL_FEATURES:
        counts = X_train[col].value_counts()  # most frequent first
        spec.append(
            {
                "name": col,
                "type": "categorical",
                "label": FEATURE_DESCRIPTIONS[col],
                "categories": [str(c) for c in counts.index],
                "default": str(counts.index[0]),
            }
        )
    return spec


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------
def run_training(save: bool = True, verbose: bool = True) -> dict:
    """Run the whole procedure. Returns everything the notebook needs to plot.

    save=False trains and evaluates without touching models/ or reports/.
    """
    log = print if verbose else (lambda *a, **k: None)

    # 1. Load
    df = load_data()
    log(f"Loaded {len(df):,} rows; bad rate = {(df[TARGET] == POSITIVE_LABEL).mean():.1%}")

    # 2. Split (stratified so both sets have the same 30% bad rate)
    df_train, df_test = split_data(df)
    X_train, y_train = split_xy(df_train)
    X_test, y_test = split_xy(df_test)
    log(f"Train {len(X_train)} rows / test {len(X_test)} rows; {len(FEATURES)} features")

    # 3-4. Compare candidates with CV on the training set only
    log("\n5-fold stratified cross-validation on the training set:")
    cv_results = cross_validate_candidates(X_train, y_train)
    best_name = max(cv_results, key=lambda n: cv_results[n]["cv_roc_auc_mean"])
    log(f"Selected: {best_name}")

    # 5. Threshold from out-of-fold predictions (still training data only)
    pipeline = build_pipeline(candidate_models()[best_name])
    oof_proba = cross_val_predict(pipeline, X_train, y_train, cv=make_cv(), method="predict_proba")[:, 1]
    threshold, cost_tbl = choose_threshold(y_train, oof_proba)
    oof_cost_default = float(cost_tbl.loc[cost_tbl["threshold"] == DEFAULT_THRESHOLD, "cost"].iloc[0])
    oof_cost_chosen = float(cost_tbl.loc[cost_tbl["threshold"] == threshold, "cost"].iloc[0])
    log(f"\nThreshold chosen on out-of-fold training predictions: {threshold:.2f} "
        f"(cost {oof_cost_chosen:.0f} vs {oof_cost_default:.0f} at 0.50)")

    # 6. Final fit and the ONE evaluation on the test set
    pipeline.fit(X_train, y_train)
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_at_default = evaluate(y_test, test_proba, DEFAULT_THRESHOLD)
    test_at_chosen = evaluate(y_test, test_proba, threshold)
    baselines = baseline_costs(y_test)
    log("\nTest set (200 rows, evaluated once):")
    for label, m in [("threshold 0.50", test_at_default), (f"threshold {threshold:.2f}", test_at_chosen)]:
        log(f"  {label}: ROC-AUC {m['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f} | "
            f"precision(bad) {m['precision_bad']:.3f} | recall(bad) {m['recall_bad']:.3f} | "
            f"F1(bad) {m['f1_bad']:.3f} | accuracy {m['accuracy']:.3f} | cost {m['expected_cost']:.0f} | "
            f"confusion {m['confusion_matrix']}")
    log(f"  approve-everyone baseline: accuracy {baselines['approve_everyone']['accuracy']:.3f}, "
        f"cost {baselines['approve_everyone']['expected_cost']:.0f}")

    # Extras for the report: fairness ablation and interpretability
    log("\nProtected-attribute ablation (CV on training set):")
    ablation = protected_attribute_ablation(df_train, best_name)
    log(f"  without age/personal_status/foreign_worker: {ablation['without_protected']['cv_roc_auc_mean']:.3f}")
    log(f"  with them:                                  {ablation['with_protected']['cv_roc_auc_mean']:.3f}")
    coef_table = logistic_coefficients(X_train, y_train)

    # 7. Assemble metadata
    calibration = {
        "mean_predicted_probability_test": float(test_proba.mean()),
        "actual_bad_rate_test": float(np.mean(y_test)),
        "note": "class_weight='balanced' shifts probabilities upwards; compare the score to the threshold "
                "rather than reading it as a calibrated default rate.",
    }
    cost_rows = cost_tbl[(np.isclose(cost_tbl["threshold"] * 100 % 5, 0)) | (cost_tbl["threshold"] == threshold)]
    metrics = {
        "dataset": {
            "name": "German Credit (OpenML credit-g v1 = UCI Statlog German Credit Data)",
            "n_rows": int(len(df)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "bad_rate": float((df[TARGET] == POSITIVE_LABEL).mean()),
            "features_used": list(FEATURES),
            "protected_attributes_excluded": list(PROTECTED_ATTRIBUTES),
        },
        "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE, "stratified": True},
        "cross_validation": {"n_folds": N_FOLDS, "candidates": cv_results, "selected_model": best_name,
                             "selection_rule": "highest mean CV ROC-AUC"},
        "threshold": {
            "method": "minimise expected cost on out-of-fold training predictions",
            "costs": {"false_negative": COST_FALSE_NEGATIVE, "false_positive": COST_FALSE_POSITIVE,
                      "source": "cost matrix published with the UCI Statlog German Credit dataset (an assumption)"},
            "chosen": float(threshold),
            "oof_cost_at_default_threshold": oof_cost_default,
            "oof_cost_at_chosen_threshold": oof_cost_chosen,
            "cost_table_oof": cost_rows.to_dict(orient="records"),
        },
        "test": {"at_default_threshold": test_at_default, "at_chosen_threshold": test_at_chosen,
                 "baselines": baselines, "calibration": calibration},
        "protected_attribute_ablation": ablation,
        "logistic_regression_top_coefficients": coef_table.head(15).round(4).to_dict(orient="records"),
        "environment": {"sklearn_version": sklearn.__version__, "python_version": platform.python_version(),
                        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
    }
    bundle = {
        "pipeline": pipeline,
        "model_name": best_name,
        "feature_names": list(FEATURES),
        "features": describe_features(X_train),
        "target": {"name": TARGET, "positive_label": POSITIVE_LABEL, "meaning": "1 = bad (defaulted), 0 = good"},
        "threshold": float(threshold),
        "costs": {"false_negative": COST_FALSE_NEGATIVE, "false_positive": COST_FALSE_POSITIVE},
        "metrics": metrics["test"],
        "cv_results": cv_results,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "random_state": RANDOM_STATE,
        "trained_at": metrics["environment"]["trained_at"],
    }

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH, compress=3)  # 500 trees compress from ~8 MB to a few MB
        METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        coef_table.round(6).to_csv(COEFFICIENTS_PATH, index=False)
        MODEL_CARD_PATH.write_text(render_model_card(metrics, coef_table), encoding="utf-8")
        log(f"\nSaved {MODEL_PATH.relative_to(ROOT)}, {METRICS_PATH.relative_to(ROOT)}, "
            f"{MODEL_CARD_PATH.relative_to(ROOT)}, {COEFFICIENTS_PATH.relative_to(ROOT)}")

    return {
        "metrics": metrics,
        "bundle": bundle,
        "pipeline": pipeline,
        "coefficients": coef_table,
        "cost_table": cost_tbl,
        "X_train": X_train, "y_train": y_train, "oof_proba": oof_proba,
        "X_test": X_test, "y_test": y_test, "test_proba": test_proba,
    }


# ---------------------------------------------------------------------------
# Model card (generated from the numbers, so it cannot drift from the run)
# ---------------------------------------------------------------------------
def _metric_row(label: str, m: dict) -> str:
    cm = m["confusion_matrix"]
    return (f"| {label} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} | {m['precision_bad']:.3f} | "
            f"{m['recall_bad']:.3f} | {m['f1_bad']:.3f} | {m['accuracy']:.3f} | "
            f"TN {cm['tn']} / FP {cm['fp']} / FN {cm['fn']} / TP {cm['tp']} | {m['expected_cost']:.0f} |")


def render_model_card(metrics: dict, coef_table: pd.DataFrame) -> str:
    d, cv, th, te, ab = (metrics["dataset"], metrics["cross_validation"], metrics["threshold"],
                         metrics["test"], metrics["protected_attribute_ablation"])
    best = cv["selected_model"]
    cand_rows = "\n".join(
        f"| {name} | {r['cv_roc_auc_mean']:.3f} +/- {r['cv_roc_auc_std']:.3f} | {r['cv_pr_auc_mean']:.3f} |"
        + (" selected |" if name == best else " |")
        for name, r in cv["candidates"].items()
    )
    coef_rows = "\n".join(
        f"| {r.feature} | {r.coefficient:+.3f} | {r.odds_ratio:.2f} |" for r in coef_table.head(10).itertuples()
    )
    base = te["baselines"]
    cal = te["calibration"]
    return f"""# Model card: German Credit risk classifier

Generated by `src/train.py` on {metrics['environment']['trained_at']}
(scikit-learn {metrics['environment']['sklearn_version']}, Python {metrics['environment']['python_version']}).
Every number here comes from that run. Do not edit by hand; re-run the script.

## What the model is

* **Task**: binary classification. Input: one loan application. Output: a score between 0 and 1
  for the applicant turning out *bad* (not repaying), and a HIGH / LOW risk decision at a threshold.
* **Model**: `{best}` inside a scikit-learn `Pipeline`
  (`StandardScaler` for {len(NUMERIC_FEATURES)} numeric columns, `OneHotEncoder` for
  {len(CATEGORICAL_FEATURES)} categorical columns). Both candidates used `class_weight="balanced"`.
* **Selection**: {cv['n_folds']}-fold stratified cross-validation on the training set, highest mean ROC-AUC.

| Candidate | CV ROC-AUC | CV PR-AUC | |
|---|---|---|---|
{cand_rows}

* **Decision threshold**: **{th['chosen']:.2f}** (instead of the default 0.50). Chosen by minimising
  expected cost on out-of-fold training predictions, assuming a missed bad loan costs
  {th['costs']['false_negative']} and a rejected good applicant costs {th['costs']['false_positive']}
  (the cost matrix published with the dataset; **an assumption, not a measured number**).
  On the training data this lowered the out-of-fold cost from {th['oof_cost_at_default_threshold']:.0f}
  to {th['oof_cost_at_chosen_threshold']:.0f}.

## Data

* {d['name']}: {d['n_rows']:,} loan applications at a German bank, donated by Hans Hofmann in 1994.
  Licence CC BY 4.0 (see `data/README.md`). Bad rate {d['bad_rate']:.0%}.
* Split: {d['n_train']} train / {d['n_test']} test, stratified, `random_state={metrics['split']['random_state']}`.
* Inputs ({len(d['features_used'])}): {', '.join(d['features_used'])}.
* **Deliberately excluded**: {', '.join(d['protected_attributes_excluded'])} (see fairness note).

## Test-set results (evaluated once)

"bad" is the positive class. Costs use the 5:1 assumption above.

| Operating point | ROC-AUC | PR-AUC | Precision (bad) | Recall (bad) | F1 (bad) | Accuracy | Confusion matrix | Cost |
|---|---|---|---|---|---|---|---|---|
{_metric_row('threshold 0.50', te['at_default_threshold'])}
{_metric_row(f"threshold {th['chosen']:.2f}", te['at_chosen_threshold'])}

Baselines on the same test set: approving everyone scores accuracy {base['approve_everyone']['accuracy']:.3f}
with recall 0 and cost {base['approve_everyone']['expected_cost']:.0f}; rejecting everyone costs
{base['reject_everyone']['expected_cost']:.0f}. Accuracy alone is therefore a poor yardstick here.

Calibration: mean predicted probability on the test set is {cal['mean_predicted_probability_test']:.3f} against an
actual bad rate of {cal['actual_bad_rate_test']:.3f}. {cal['note']}

## Fairness note and protected-attribute ablation

The dataset contains `personal_status` (encodes sex and marital status), `foreign_worker`
(a proxy for national origin) and `age`. Deciding who gets credit on the basis of sex, national
origin or age is discriminatory and illegal in many places (for example the US Equal Credit
Opportunity Act and EU equal-treatment rules), so these columns are **not model inputs**.

What that decision costs, measured with the same {cv['n_folds']}-fold CV on the training set
(`{best}`):

| Inputs | CV ROC-AUC |
|---|---|
| without protected attributes (the saved model) | {ab['without_protected']['cv_roc_auc_mean']:.3f} +/- {ab['without_protected']['cv_roc_auc_std']:.3f} |
| with age, personal_status, foreign_worker | {ab['with_protected']['cv_roc_auc_mean']:.3f} +/- {ab['with_protected']['cv_roc_auc_std']:.3f} |

Difference: {ab['roc_auc_difference']:+.3f} ROC-AUC. Removing the columns is not a full fairness fix:
other features (job, housing, purpose) can act as proxies. A real audit would compare error rates
across groups; this learning project does not do that.

## Interpretation (logistic-regression coefficients)

Top 10 coefficients by absolute size (positive = raises the odds of default). Numeric features are
standardised, so their coefficients are per +1 standard deviation. Full table: `reports/logistic_coefficients.csv`.

| Feature | Coefficient | Odds ratio |
|---|---|---|
{coef_rows}

## Limitations

* 1,000 rows from a German bank in the early 1990s, amounts in Deutsche Mark. It says nothing
  reliable about any current population or lender.
* The test set has only {d['n_test']} rows ({int(round(d['n_test'] * d['bad_rate']))} bad), so every test
  metric carries sizeable sampling noise; differences of a few hundredths are not meaningful.
* No hyperparameter tuning was done; both candidates use sensible fixed settings.
* The 5:1 cost ratio is an assumption. Change it in `src/scoring.py` and the threshold moves.
* Probabilities from a class-weighted model are not calibrated to the real default rate.

## Intended use

A learning and portfolio project. **Not for real lending decisions.**
"""


if __name__ == "__main__":
    run_training(save=True)

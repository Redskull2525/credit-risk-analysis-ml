"""Credit Risk Scorer - Streamlit app.

The input form is generated from the feature list saved inside the model
bundle (models/credit_pipeline.pkl). If training adds, removes or renames a
feature, the form follows automatically, so the app cannot drift away from
the model again (the first version of this repo sent 5 inputs to a model
that expected 11).

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:  # make `src` importable wherever Streamlit is launched from
    sys.path.insert(0, str(ROOT))

from src.scoring import HIGH_RISK, decide, load_bundle  # noqa: E402

REPO_URL = "https://github.com/Redskull2525/credit-risk-analysis-ml"

st.set_page_config(page_title="Credit Risk Scorer", page_icon=":bank:", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_bundle() -> dict:
    """Load the pipeline + metadata once per server process."""
    return load_bundle()


def render_sidebar(bundle: dict) -> None:
    metrics = bundle["metrics"]["at_chosen_threshold"]
    st.sidebar.title("About me")
    st.sidebar.markdown(
        "**Abhishek Shelke** - aspiring Data Analyst  \n"
        "MSc Computer Science, Pune, India  \n"
        "[GitHub](https://github.com/Redskull2525) | "
        "[LinkedIn](https://www.linkedin.com/in/abhishek-s-b98895249)"
    )
    st.sidebar.markdown("---")
    st.sidebar.title("Model")
    st.sidebar.markdown(
        f"- Model: `{bundle['model_name']}` (scikit-learn {bundle['sklearn_version']})\n"
        f"- Inputs: {len(bundle['feature_names'])} features\n"
        f"- Test ROC-AUC: {metrics['roc_auc']:.3f}\n"
        f"- Test PR-AUC: {metrics['pr_auc']:.3f}\n"
        f"- Decision threshold: {bundle['threshold']:.2f}\n"
        f"- Cost assumption: missed bad loan = {bundle['costs']['false_negative']}, "
        f"rejected good applicant = {bundle['costs']['false_positive']}\n"
    )
    st.sidebar.markdown(f"[Source code, model card and tests]({REPO_URL})")


def render_form(features: list[dict]) -> tuple[dict, bool]:
    """One widget per saved feature. Returns (values, submitted)."""
    values: dict = {}
    with st.form("applicant_form"):
        left, right = st.columns(2)
        for i, feat in enumerate(features):
            with left if i % 2 == 0 else right:
                if feat["type"] == "numeric":
                    values[feat["name"]] = st.number_input(
                        feat["label"],
                        min_value=int(feat["min"]),
                        max_value=int(feat["max"]),
                        value=int(feat["default"]),
                        step=1,
                        key=feat["name"],
                        help=f"Range seen in the training data: {feat['min']} to {feat['max']}",
                    )
                else:
                    options = list(feat["categories"])
                    values[feat["name"]] = st.selectbox(
                        feat["label"],
                        options=options,
                        index=options.index(feat["default"]),
                        key=feat["name"],
                    )
        submitted = st.form_submit_button("Score this applicant", type="primary")
    return values, submitted


def render_result(bundle: dict, values: dict) -> None:
    row = pd.DataFrame([values])[bundle["feature_names"]]  # same column order as training
    probability = float(bundle["pipeline"].predict_proba(row)[0, 1])
    threshold = float(bundle["threshold"])
    decision = decide(probability, threshold)
    test = bundle["metrics"]["at_chosen_threshold"]
    costs = bundle["costs"]

    st.subheader("Result")
    col1, col2 = st.columns(2)
    col1.metric("Probability of default (risk score)", f"{probability:.1%}")
    col2.metric("Decision threshold", f"{threshold:.0%}")
    st.progress(min(max(probability, 0.0), 1.0))

    if decision == HIGH_RISK:
        st.error(
            f"HIGH RISK: the score {probability:.1%} is at or above the threshold {threshold:.0%}. "
            "A lender would decline this application or send it to manual review."
        )
    else:
        st.success(
            f"LOW RISK: the score {probability:.1%} is below the threshold {threshold:.0%}. "
            "A lender would approve."
        )

    with st.expander("How to read this", expanded=True):
        st.markdown(
            f"""
- **The score** is the model's estimate that this applicant will *not* repay, from a
  `{bundle['model_name']}` trained on 800 historical applications.
- **The threshold ({threshold:.2f}) is not the default 0.5.** It was chosen to minimise expected cost,
  assuming a missed bad loan costs {costs['false_negative']} and a wrongly rejected good applicant costs
  {costs['false_positive']} (the cost matrix published with the dataset - an assumption, not a measurement).
  A lower threshold rejects more people in order to catch more defaults.
- **Scores are not calibrated probabilities.** The model was trained with class weighting, so a score of
  40% does not mean 40 in 100 such applicants default. Compare the score with the threshold instead.
- **On the held-out test set** ({bundle['metrics']['at_chosen_threshold']['confusion_matrix']['tp'] + bundle['metrics']['at_chosen_threshold']['confusion_matrix']['fn']} bad loans among 200 applications)
  the model reached ROC-AUC {test['roc_auc']:.3f}; at this threshold it caught {test['recall_bad']:.0%} of the bad loans
  while rejecting {test['rejected_share']:.0%} of all applicants.
"""
        )


def render_model_details(bundle: dict) -> None:
    with st.expander("What is inside the model"):
        st.markdown(
            "Inputs are the 17 columns below. `age`, `personal_status` (sex and marital status) and "
            "`foreign_worker` exist in the data but are **deliberately not used**: deciding credit on sex, "
            "nationality or age is discriminatory and illegal in many jurisdictions."
        )
        st.dataframe(
            pd.DataFrame(
                [{"feature": f["name"], "meaning": f["label"], "type": f["type"]} for f in bundle["features"]]
            ),
            hide_index=True,
            width="stretch",
        )
        rows = []
        for label, m in [("threshold 0.50", bundle["metrics"]["at_default_threshold"]),
                         (f"threshold {bundle['threshold']:.2f}", bundle["metrics"]["at_chosen_threshold"])]:
            rows.append(
                {"operating point": label, "ROC-AUC": round(m["roc_auc"], 3), "PR-AUC": round(m["pr_auc"], 3),
                 "precision (bad)": round(m["precision_bad"], 3), "recall (bad)": round(m["recall_bad"], 3),
                 "F1 (bad)": round(m["f1_bad"], 3), "accuracy": round(m["accuracy"], 3),
                 "expected cost": m["expected_cost"]}
            )
        st.markdown("Test-set metrics (200 applications, evaluated once):")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def main() -> None:
    bundle = get_bundle()
    render_sidebar(bundle)

    st.title("Credit Risk Scorer")
    st.caption("Will this applicant repay? A scikit-learn pipeline trained on the German Credit dataset.")
    st.warning(
        "**Learning project - not for real lending decisions.** The model learned from 1,000 loan "
        "applications at a German bank in 1994 (amounts in Deutsche Mark). It knows nothing about "
        "today's borrowers, and a real lender would need calibration, fairness testing and regulatory review."
    )

    st.subheader("Applicant details")
    st.caption("Defaults are typical values from the training data. Change them and press the button.")
    values, submitted = render_form(bundle["features"])
    if submitted:
        render_result(bundle, values)
    render_model_details(bundle)


if __name__ == "__main__":
    main()

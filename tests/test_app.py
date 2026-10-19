"""Streamlit smoke tests with AppTest: the app renders, the form matches the model, scoring works."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def run_app() -> AppTest:
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def test_app_renders_without_exception():
    at = run_app()
    assert "Credit Risk Scorer" in at.title[0].value
    assert at.warning, "the learning-project disclaimer must be visible"


def test_form_is_generated_from_saved_feature_list():
    at = run_app()
    numeric_keys = {w.key for w in at.number_input}
    categorical_keys = {w.key for w in at.selectbox}
    assert numeric_keys == set(NUMERIC_FEATURES)
    assert categorical_keys == set(CATEGORICAL_FEATURES)


def test_fill_form_and_score_applicant():
    at = run_app()
    at.number_input(key="duration").set_value(48)
    at.number_input(key="credit_amount").set_value(9000)
    at.selectbox(key="checking_status").select("<0")
    at.selectbox(key="credit_history").select("no credits/all paid")
    submit = [b for b in at.button if "Score" in b.label]
    assert submit, "submit button not found"
    submit[0].click().run()
    assert not at.exception, at.exception
    labels = [m.label for m in at.metric]
    assert any("Probability of default" in label for label in labels)
    # Exactly one decision banner is shown.
    assert len(at.error) + len(at.success) == 1

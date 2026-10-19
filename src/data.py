"""Data access for the German Credit dataset.

Source: OpenML dataset 31 "credit-g" (version 1), which is the UCI
"Statlog (German Credit Data)" dataset donated by Hans Hofmann (1994).
Licence: CC BY 4.0. See data/README.md for the full citation.

Download it once with:

    python src/data.py

This writes data/german_credit.csv (1,000 rows), which is committed to the
repository so that training is reproducible without network access.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "german_credit.csv"

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
TARGET = "class"            # raw labels in the file: "good" or "bad"
POSITIVE_LABEL = "bad"      # we model P(bad): the event the bank wants to avoid

# ---------------------------------------------------------------------------
# Model inputs
# ---------------------------------------------------------------------------
# All of these are integers in the source data.
NUMERIC_FEATURES = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "existing_credits",
    "num_dependents",
]

# All of these are short text codes (e.g. checking_status = "<0", "no checking").
CATEGORICAL_FEATURES = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Protected attributes: present in the data, deliberately NOT used as inputs.
# ---------------------------------------------------------------------------
# * personal_status encodes sex (and marital status).
# * foreign_worker is a proxy for national origin.
# * age is a protected characteristic under most fair-lending rules.
# Using them to decide who gets credit is discriminatory in many jurisdictions
# (e.g. the US Equal Credit Opportunity Act, EU equal-treatment law).
# src/train.py measures what excluding them costs in ROC-AUC (see the model card).
PROTECTED_NUMERIC = ["age"]
PROTECTED_CATEGORICAL = ["personal_status", "foreign_worker"]
PROTECTED_ATTRIBUTES = PROTECTED_NUMERIC + PROTECTED_CATEGORICAL

# Plain-English labels, used by the Streamlit form and the docs.
FEATURE_DESCRIPTIONS = {
    "checking_status": "Checking account balance (DM)",
    "duration": "Loan duration (months)",
    "credit_history": "Credit history",
    "purpose": "Purpose of the loan",
    "credit_amount": "Loan amount (DM)",
    "savings_status": "Savings account / bonds balance (DM)",
    "employment": "Years in present job",
    "installment_commitment": "Installment rate code (1-4, share of disposable income)",
    "other_parties": "Other debtors / guarantors",
    "residence_since": "Years at present residence (1-4)",
    "property_magnitude": "Most valuable property owned",
    "other_payment_plans": "Other installment plans",
    "housing": "Housing situation",
    "existing_credits": "Number of existing credits at this bank",
    "job": "Job type",
    "num_dependents": "Number of dependents",
    "own_telephone": "Telephone registered in applicant's name",
    # protected attributes (not model inputs)
    "age": "Age in years",
    "personal_status": "Sex and marital status",
    "foreign_worker": "Foreign worker",
    # target
    "class": "Outcome: good = repaid, bad = defaulted",
}


def download_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Fetch credit-g from OpenML (no credentials needed) and save it as CSV."""
    from sklearn.datasets import fetch_openml  # imported here: only needed for the download

    dataset = fetch_openml("credit-g", version=1, as_frame=True)
    df: pd.DataFrame = dataset.frame.copy()

    # Numeric columns arrive as floats (6.0, 1169.0 ...) but are whole numbers.
    for col in NUMERIC_FEATURES + PROTECTED_NUMERIC:
        df[col] = df[col].astype(int)
    # Categorical columns arrive as pandas Categorical; plain strings are simpler.
    for col in CATEGORICAL_FEATURES + PROTECTED_CATEGORICAL + [TARGET]:
        df[col] = df[col].astype(str)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the committed CSV. Categorical columns are read as plain strings."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run `python src/data.py` to download it.")
    text_cols = CATEGORICAL_FEATURES + PROTECTED_CATEGORICAL + [TARGET]
    return pd.read_csv(path, dtype={col: str for col in text_cols})


def split_xy(df: pd.DataFrame, features: list[str] = FEATURES) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) where y is 1 for a bad loan (default) and 0 for a good one."""
    X = df[features].copy()
    y = (df[TARGET] == POSITIVE_LABEL).astype(int).rename("is_bad")
    return X, y


if __name__ == "__main__":
    frame = download_data()
    print(f"Saved {len(frame):,} rows x {frame.shape[1]} columns to {DATA_PATH}")
    print(frame[TARGET].value_counts())

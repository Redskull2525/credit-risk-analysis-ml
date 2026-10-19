"""Shared fixtures. Adds the repo root to sys.path so `src` imports work from any cwd."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_data  # noqa: E402
from src.scoring import load_bundle  # noqa: E402


@pytest.fixture(scope="session")
def data():
    return load_data()


@pytest.fixture(scope="session")
def bundle():
    return load_bundle()

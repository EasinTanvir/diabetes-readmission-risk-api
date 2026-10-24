import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_splits  # noqa: E402
from src.scoring import Scorer  # noqa: E402


def to_records(df):
    """DataFrame -> JSON-like dicts, as an API client would send them (NaN -> None)."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


@pytest.fixture(scope="session")
def val_df():
    _, _, val, _ = load_splits()
    return val.drop(columns=["readmitted", "target"])


@pytest.fixture(scope="session")
def val_records(val_df):
    return to_records(val_df)


@pytest.fixture(scope="session")
def scorer():
    return Scorer()


@pytest.fixture
def base(val_records):
    """One clean validation row (fresh copy for each test)."""
    return dict(val_records[0])

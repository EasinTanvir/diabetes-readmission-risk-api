import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import DATA_CSV, load_splits  # noqa: E402
from src.scoring import Scorer  # noqa: E402

EXAMPLE_JSON = ROOT / "examples" / "predict_request.json"
NO_DATA = ("diabetic_data.csv not found - download it (see README 'Data') "
           "to run the data-dependent tests")


def to_records(df):
    """DataFrame -> JSON-like dicts, as an API client would send them (NaN -> None)."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


def frame_sha256(df):
    """Exact fingerprint of a DataFrame: columns, dtypes and every value bit (index ignored)."""
    h = hashlib.sha256()
    h.update(json.dumps(df.columns.tolist()).encode())
    h.update(json.dumps(df.dtypes.astype(str).tolist()).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()


@pytest.fixture(scope="session")
def val_df():
    if not DATA_CSV.exists():
        pytest.skip(NO_DATA)
    _, _, val, _ = load_splits()
    return val.drop(columns=["readmitted", "target"])


@pytest.fixture(scope="session")
def val_records(val_df):
    return to_records(val_df)


@pytest.fixture(scope="session")
def scorer():
    return Scorer()


@pytest.fixture
def base():
    """One clean synthetic patient from examples/ (fresh copy for each test, no data needed)."""
    return dict(json.loads(EXAMPLE_JSON.read_text())[0])

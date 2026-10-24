"""Clean data: the hardened path must give exactly the same features and scores as before."""
import hashlib
from collections import Counter

import numpy as np

from conftest import ROOT, frame_sha256
from src.prep import prepare
from src.schema import validate_patient

EXPECTED_WARNINGS = ("not seen in training -> Other", "above the training maximum",
                     "gender: 'Unknown/Invalid'")


def test_pipeline_pkl_unchanged():
    sha = hashlib.sha256((ROOT / "model" / "pipeline.pkl").read_bytes()).hexdigest()
    assert sha == (ROOT / "eval" / "reference_pipeline_sha256.txt").read_text().strip()


def test_all_validation_rows_identical(scorer, val_records):
    results = scorer.score(val_records)
    assert len(results) == 14911
    assert [r.error for r in results if r.error] == []

    scores = np.array([r.risk_score for r in results])
    reference = np.load(ROOT / "eval" / "reference_val_scores.npy")
    assert np.array_equal(scores, reference)

    flagged = np.array([r.flagged for r in results])
    assert np.array_equal(flagged, reference >= scorer.cutoff)

    warnings = Counter(w for r in results for w in r.warnings)
    print("\nwarnings on clean validation rows:", dict(warnings))
    assert all(any(e in w for e in EXPECTED_WARNINGS) for w in warnings), warnings


def test_prepared_table_identical(scorer, val_records):
    raw = [validate_patient(r)[0] for r in val_records]
    X = prepare(scorer.to_frame(raw), scorer.cfg, scorer.columns)
    # exact fingerprint of eval/reference_val_X.pkl (the pickle itself is not in git)
    reference = (ROOT / "eval" / "reference_val_X_sha256.txt").read_text().strip()
    assert frame_sha256(X) == reference

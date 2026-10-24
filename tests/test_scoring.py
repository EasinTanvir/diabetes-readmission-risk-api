"""Every robustness case gives a clear error or a logged, deliberate fallback - never a silent wrong score."""
import copy
import logging

import numpy as np
import pytest

from cases import CASES, apply


@pytest.mark.parametrize("name,change,expect", CASES, ids=[c[0] for c in CASES])
def test_case_outcome(scorer, base, caplog, name, change, expect):
    kind, text = expect
    with caplog.at_level(logging.WARNING, logger="readmission"):
        (r,) = scorer.score([apply(base, change)])

    if kind == "error":
        assert r.risk_score is None and r.flagged is None
        assert r.error and text in r.error, r.error
        assert not r.error.startswith("internal"), r.error
        return

    assert r.error is None, r.error
    assert 0.0 <= r.risk_score <= 1.0
    assert r.flagged == (r.risk_score >= scorer.cutoff)
    if kind == "warning":
        assert any(text in w for w in r.warnings), r.warnings
        assert any(text in rec.getMessage() for rec in caplog.records), "warning was not logged"
    else:
        assert r.warnings == [], r.warnings


def test_base_row_is_clean(scorer, base):
    (r,) = scorer.score([base])
    assert r.error is None and r.warnings == []


def test_bad_row_does_not_break_batch(scorer, val_records):
    good = [dict(r) for r in val_records[:5]]
    bad = [apply(good[0], {"diag_1": "abc", "age": "[100-110)"}), "not an object", {}]
    batch = good[:2] + bad + good[2:]
    results = scorer.score(batch)

    alone = scorer.score(good)
    scored = [r for r in results if r.risk_score is not None]
    assert [r.risk_score for r in scored] == [r.risk_score for r in alone]
    assert results[2].error.startswith("age:")
    assert results[3].error.startswith("row: must be a JSON object")
    assert "gender: required field is missing" in results[4].error


def test_one_row_vs_many(scorer, val_records):
    rows = val_records[:300]
    batch = np.array([r.risk_score for r in scorer.score(rows)])
    single = np.array([scorer.score([r])[0].risk_score for r in rows])
    assert np.array_equal(batch, single)


def test_input_not_changed(scorer, val_records):
    rows = [dict(r) for r in val_records[:50]] + [apply(val_records[0], {"diag_1": " v45 ", "payer_code": "ZZ"})]
    before = copy.deepcopy(rows)
    scorer.score(rows)
    assert rows == before


def test_ids_never_reach_model(scorer, base):
    a = scorer.score([base])[0].risk_score
    b = scorer.score([apply(base, {"patient_nbr": 1, "encounter_id": 2})])[0].risk_score
    assert a == b

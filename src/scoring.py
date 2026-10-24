"""Safe scoring: validate each row on its own, score the valid rows, never return a silent wrong score.

risk_score is for ranking only (class_weight="balanced" inflates it). It is not a probability.
"""
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from src.prep import CAT_COLS, prepare
from src.schema import INPUT_FIELDS, INT_FIELDS, validate_patient

log = logging.getLogger("readmission")

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "pipeline.pkl"
SCORE_NOTE = "risk_score is for ranking only; it is not a probability"


@dataclass
class Result:
    id: Optional[Any] = None
    risk_score: Optional[float] = None
    flagged: Optional[bool] = None
    warnings: list = field(default_factory=list)
    error: Optional[str] = None


class Scorer:
    def __init__(self, model_path=MODEL_PATH):
        model_path = Path(model_path)
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.cutoff = float(bundle["cutoff"])
        self.columns = bundle["columns"]
        self.cfg = bundle["cfg"]
        self.versions = bundle["versions"]
        self.model_version = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
        self.groups = {c: [k for k in self.columns if k.startswith(c + "_")] for c in CAT_COLS}

    def to_frame(self, raw_rows):
        """Validated raw rows -> DataFrame with the same dtypes as diabetic_data.csv."""
        df = pd.DataFrame(raw_rows, columns=INPUT_FIELDS)
        df[INT_FIELDS] = df[INT_FIELDS].astype("int64")
        other = [c for c in INPUT_FIELDS if c not in INT_FIELDS]
        df[other] = df[other].astype(object)
        return df

    def bad_features(self, X):
        """Per row: why the prepared features cannot be trusted (empty string = fine)."""
        reasons = pd.Series("", index=X.index)
        nan_or_inf = ~np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
        reasons[nan_or_inf] = "NaN/inf in features"
        for c, ks in self.groups.items():
            empty = X[ks].sum(axis=1) == 0
            reasons[empty & (reasons == "")] = f"no {c} category matched"
        return reasons

    def score(self, records):
        results, valid_pos, raw_rows = [], [], []
        for i, rec in enumerate(records):
            raw, warnings, errors, row_id = validate_patient(rec)
            results.append(Result(id=row_id, warnings=warnings, error="; ".join(errors) or None))
            if raw is not None:
                valid_pos.append(i)
                raw_rows.append(raw)
            else:
                log.warning("row %s rejected: %s", row_id, results[-1].error)

        if raw_rows:
            X = prepare(self.to_frame(raw_rows), self.cfg, self.columns)
            reasons = self.bad_features(X)
            scores = self.model.predict_proba(X)[:, 1]
            for j, i in enumerate(valid_pos):
                if reasons.iloc[j]:
                    # guard: should never happen after validation; refuse rather than score silently
                    results[i].error = f"internal: unexpected features ({reasons.iloc[j]})"
                    log.error("row %s: %s", results[i].id, results[i].error)
                    continue
                results[i].risk_score = float(scores[j])
                results[i].flagged = bool(scores[j] >= self.cutoff)
        return results

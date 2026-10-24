"""Shared metric helpers for the eval scripts."""
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "model" / "pipeline.pkl"


def load_bundle():
    return joblib.load(MODEL_PATH)


def at_cutoff(y, scores, cutoff):
    """Flag rate, recall, precision, lift and confusion matrix for a fixed cutoff."""
    y = np.asarray(y)
    flagged = scores >= cutoff
    caught = int((flagged & (y == 1)).sum())
    n_flag, n_pos = int(flagged.sum()), int(y.sum())
    precision = caught / n_flag if n_flag else float("nan")
    return {
        "n": len(y),
        "positives": n_pos,
        "flag_rate": flagged.mean(),
        "recall": caught / n_pos if n_pos else float("nan"),
        "precision": precision,
        "lift": precision / y.mean() if y.mean() else float("nan"),
        "confusion": confusion_matrix(y, flagged.astype(int), labels=[0, 1]),
    }


def ranking(y, scores):
    return {"roc_auc": roc_auc_score(y, scores), "avg_precision": average_precision_score(y, scores)}


def check(name, got, expected, tol=0.01):
    ok = abs(got - expected) <= tol
    print(f"  {name:<14} got {got:.4f} | expected {expected} | {'OK' if ok else 'DIFFERS > ' + str(tol)}")
    return ok

"""Step 0 of the API work: freeze the current validation scores and prepared table as the reference.

Run once, BEFORE any change to src/prep.py. Later checks compare against these files.
"""
import hashlib

import numpy as np

from metrics import MODEL_PATH, ROOT, load_bundle

from src.data import load_splits
from src.prep import prepare

bundle = load_bundle()
_, _, val_df, _ = load_splits()
X_val = prepare(val_df, bundle["cfg"], bundle["columns"])
scores = bundle["model"].predict_proba(X_val)[:, 1]

np.save(ROOT / "eval" / "reference_val_scores.npy", scores)
X_val.to_pickle(ROOT / "eval" / "reference_val_X.pkl")
sha = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
(ROOT / "eval" / "reference_pipeline_sha256.txt").write_text(sha + "\n")

print("val rows:", len(scores), "| score mean:", round(float(scores.mean()), 6))
print("pipeline.pkl sha256:", sha)

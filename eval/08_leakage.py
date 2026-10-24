"""Step 8: leakage and logic audit - programmatic checks on the bundle, plus a scan of the notebook."""
import json
import re

import numpy as np
import pandas as pd

from metrics import ROOT, load_bundle

from src.data import load_splits
from src.prep import fit_cfg, prepare

bundle = load_bundle()
cfg, columns, model = bundle["cfg"], bundle["columns"], bundle["model"]
df, train_df, val_df, test_df = load_splits()

print("1. Forbidden columns among model features")
forbidden = ["readmitted", "target", "encounter_id", "patient_nbr"]
hits = [c for c in columns if any(f in c for f in forbidden)]
print("   hits:", hits or "none", "| n features:", len(columns))

print("\n2. cfg learned from TRAINING only")
print("   bundle cfg == fit_cfg(train):", cfg == fit_cfg(train_df))
for name, rows in [("train+val", pd.concat([train_df, val_df])), ("all rows", df)]:
    other = fit_cfg(rows)
    diffs = [k for k in cfg if cfg[k] != other[k]]
    print(f"   fit_cfg({name}) differs in: {diffs or 'nothing'}")
    for k in diffs:
        if k == "keep":
            for c in cfg["keep"]:
                a, b = set(cfg["keep"][c]), set(other["keep"][c])
                if a != b:
                    print(f"      keep[{c}]: train-only {sorted(a - b)} | extra with {name} {sorted(b - a)}")

print("\n3. One-hot column list built from training only")
train_cols = prepare(train_df, cfg).columns.tolist()
val_only = sorted(set(prepare(val_df, cfg).columns) - set(train_cols))
test_only = sorted(set(prepare(test_df, cfg).columns) - set(train_cols))
print("   bundle columns == train dummies:", columns == train_cols)
print("   dummies seen only in val:", val_only or "none", "| only in test:", test_only or "none")

print("\n4. Cutoff came from validation")
scores = {n: model.predict_proba(prepare(d, cfg, columns))[:, 1]
          for n, d in [("train", train_df), ("val", val_df), ("test", test_df)]}
for n, s in scores.items():
    print(f"   90th pct of {n:<5} scores: {np.quantile(s, 0.9):.4f}")
print(f"   bundle cutoff:              {bundle['cutoff']:.4f}")

print("\n5. Early stopping inside HGB (validation_fraction=%s, not grouped by patient)" % model.validation_fraction)
stays = train_df["patient_nbr"].map(train_df["patient_nbr"].value_counts())
print(f"   training rows from patients with >1 stay: {(stays > 1).mean():.1%}")
print("   -> the random 10% early-stop holdout shares patients with the fitting rows")
print("   iterations used:", model.n_iter_, "of max_iter", model.max_iter)

print("\n6. Notebook scan: where test data is touched")
nb = json.loads((ROOT / "Diabetes.ipynb").read_text(encoding="utf-8"))
test_pat = re.compile(r"\b(X_test\w*|y_test|test_df|test_probs|test_idx)\b")
metric_pat = re.compile(r"(roc_auc_score|average_precision_score|confusion_matrix|recall_score|precision_score|accuracy_score|\.fit\(|quantile)")
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    lines = [l for l in src.splitlines() if test_pat.search(l) and not l.lstrip().startswith("#")]
    if not lines:
        continue
    used = [l.strip() for l in lines if metric_pat.search(l)]
    kind = "METRIC/FIT ON TEST" if used else "same transform as train/val"
    print(f"   cell [{i}]: {kind}")
    for l in used:
        print("      ", l[:110])

print("\n7. Notebook scan: cross-validation uses patient groups")
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if re.search(r"cross_validate|SearchCV", src):
        print(f"   cell [{i}]: GroupKFold={'GroupKFold' in src} | groups=g_train passed={'groups=g_train' in src}")

print("\n8. Notebook scan: decisions made on validation (cutoff and model choice)")
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if re.search(r"(X_val\w*|y_val|val_probs)", src) and metric_pat.search(src):
        print(f"   cell [{i}] uses validation for metrics/quantile")

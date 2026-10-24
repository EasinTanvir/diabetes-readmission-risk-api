"""Step 7: robustness of prepare() + model on single RAW rows, as the future API will send them.

Each case changes one copy of a real validation row. Status:
  OK     runs, features look sane
  SILENT runs, but the features are wrong or empty (NaN/inf, a whole one-hot group all 0,
         a value quietly turned into a default) -> a score comes back with no warning
  CRASH  raises an exception
"""
import copy

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from metrics import load_bundle

from src.data import load_splits
from src.prep import CAT_COLS, prepare

bundle = load_bundle()
model, cfg, columns = bundle["model"], bundle["cfg"], bundle["columns"]
_, _, val_df, _ = load_splits()
RAW_COLS = [c for c in val_df.columns if c not in ("readmitted", "target")]
api_val = val_df[RAW_COLS]                  # what an API would receive: no label columns
base = api_val.iloc[[0]].reset_index(drop=True)
base_X = prepare(base, cfg, columns)
base_score = model.predict_proba(base_X)[0, 1]

# one-hot groups that exist in the model's columns
GROUPS = {c: [k for k in columns if k.startswith(c + "_")] for c in CAT_COLS}


def problems(X):
    """Things that make a row's features wrong without raising."""
    out = []
    num = X.select_dtypes(include="number")
    if X.isnull().any().any():
        out.append("NaN in " + ",".join(X.columns[X.isnull().any()].tolist()))
    if np.isinf(num.to_numpy(dtype=float)).any():
        out.append("inf in " + ",".join(num.columns[np.isinf(num).any()].tolist()))
    empty = [c for c, ks in GROUPS.items() if ks and (X[ks].sum(axis=1) == 0).any()]
    if empty:
        out.append("all-zero one-hot: " + ",".join(empty))
    obj = X.select_dtypes(exclude="number").columns.tolist()
    if obj:
        out.append("non-numeric cols: " + ",".join(obj))
    return out


def run(name, row, expect_default=None):
    """expect_default: (feature, expected_value) - flags a quiet fallback the caller may not want."""
    try:
        X = prepare(row, cfg, columns)
        score = model.predict_proba(X)[0, 1]
    except Exception as e:
        return name, "CRASH", f"{type(e).__name__}: {str(e)[:90]}"
    notes = problems(X)
    # model columns that prepare() never built and reindex filled with 0
    built = prepare(row, cfg, None).columns
    filled = [c for c in columns if c not in built and not any(c in ks for ks in GROUPS.values())]
    if filled:
        notes.append("filled with 0 (field absent): " + ",".join(filled))
    if expect_default and X[expect_default[0]].iloc[0] == expect_default[1]:
        notes.append(f"quietly mapped to {expect_default[0]}")
    status = "SILENT" if notes else "OK"
    return name, status, f"score {score:.3f} (base {base_score:.3f})" + (" | " + "; ".join(notes) if notes else "")


def with_value(col, value):
    r = base.copy()
    # numbers keep a numeric dtype, like pd.DataFrame([json_dict]) would give
    is_num = isinstance(value, (int, float)) and value is not None
    r[col] = pd.Series([value], dtype=None if is_num else object)
    return r


results = []

# 1. unseen category values
results += [
    run("race='Martian'", with_value("race", "Martian")),
    run("medical_specialty='Astrology'", with_value("medical_specialty", "Astrology"), ("medical_specialty_Other", 1)),
    run("payer_code='ZZ'", with_value("payer_code", "ZZ"), ("payer_code_Other", 1)),
    run("gender='M'", with_value("gender", "M")),
    run("gender='Unknown/Invalid'", with_value("gender", "Unknown/Invalid")),
    run("admission_type_id=99", with_value("admission_type_id", 99), ("admission_type_id_Other", 1)),
    run("admission_type_id='1' (string)", with_value("admission_type_id", "1"), ("admission_type_id_Other", 1)),
    run("A1Cresult='8'", with_value("A1Cresult", "8")),
    run("insulin='Increased'", with_value("insulin", "Increased")),
    run("metformin='up' (lowercase)", with_value("metformin", "up")),
    run("diabetesMed='yes' (lowercase)", with_value("diabetesMed", "yes"), ("diabetesMed", 0)),
    run("change='Yes' (not 'Ch')", with_value("change", "Yes"), ("change", 0)),
]

# 2. glucose / A1C "not tested" spellings
for v in ["None", None, np.nan, "?"]:
    results.append(run(f"max_glu_serum={v!r}", with_value("max_glu_serum", v)))

# 3. odd diagnosis codes
for v in ["V45", "E880", "v45", "250.83", "", "abc", None, 428, "?"]:
    results.append(run(f"diag_1={v!r}", with_value("diag_1", v)))

# 4. out-of-range and wrong-type numbers
results += [
    run("age='[100-110)'", with_value("age", "[100-110)")),
    run("age=55 (number)", with_value("age", 55)),
    run("number_inpatient=-1", with_value("number_inpatient", -1)),
    run("number_inpatient=-3", with_value("number_inpatient", -3)),
    run("time_in_hospital=-5", with_value("time_in_hospital", -5)),
    run("num_medications=10000", with_value("num_medications", 10000)),
    run("number_emergency='2' (string)", with_value("number_emergency", "2")),
    run("time_in_hospital='3' (string)", with_value("time_in_hospital", "3")),
]

# 5. '?', None, missing - one field at a time, over every raw input field
sweep = {}
for col in RAW_COLS:
    for label, make in [("'?'", lambda c: with_value(c, "?")),
                        ("None", lambda c: with_value(c, None)),
                        ("missing", lambda c: base.drop(columns=[c]))]:
        _, status, detail = run(f"{col}={label}", make(col))
        sweep.setdefault((label, status), []).append(col if status == "OK" else f"{col} [{detail.split('|')[-1].strip()}]")

# 6. extra / reordered columns
results += [
    run("extra unknown column", base.assign(foo="bar")),
    run("columns in reverse order", base[base.columns[::-1]]),
]

print(f"{'case':<36} {'status':<7} detail")
for name, status, detail in results:
    print(f"{name:<36} {status:<7} {detail}")

print("\nOne field at a time: '?', None, missing")
for (label, status), cols in sorted(sweep.items()):
    print(f"\n[{label} -> {status}] {len(cols)} fields")
    for c in cols:
        print("   ", c)

# 7. one row vs many rows: same scores?
many = api_val.iloc[:300].reset_index(drop=True)
batch = model.predict_proba(prepare(many, cfg, columns))[:, 1]
single = np.array([model.predict_proba(prepare(many.iloc[[i]], cfg, columns))[0, 1] for i in range(len(many))])
print(f"\n1 row vs 300 rows: max |score diff| = {np.abs(batch - single).max():.2e}")

# mixed batch: one bad row breaks or poisons the whole call?
mixed = pd.concat([many.iloc[:5], with_value("diag_1", "abc")], ignore_index=True)
try:
    prepare(mixed, cfg, columns)
    print("batch with one bad diag row: runs")
except Exception as e:
    print(f"batch with one bad diag row: whole batch CRASH ({type(e).__name__})")

# 8. does prepare() change its input?
before = copy.deepcopy(many)
prepare(many, cfg, columns)
try:
    assert_frame_equal(before, many)
    print("prepare() leaves its input unchanged: True")
except AssertionError as e:
    print("prepare() CHANGES its input:", str(e)[:200])

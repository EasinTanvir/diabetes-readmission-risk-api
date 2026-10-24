"""Step 9: fairness check on VALIDATION only, by race, gender and age group (raw values).

lift = group precision / group positive rate (how much better than random flagging inside that group).
Groups with fewer than 200 distinct patients are marked UNRELIABLE.
"""
import pandas as pd

from metrics import at_cutoff, load_bundle

from src.data import load_splits
from src.prep import prepare

MIN_PATIENTS = 200

bundle = load_bundle()
_, _, val_df, _ = load_splits()
val = val_df.copy()
val["score"] = bundle["model"].predict_proba(prepare(val_df, bundle["cfg"], bundle["columns"]))[:, 1]

overall = at_cutoff(val["target"], val["score"].values, bundle["cutoff"])
print(f"ALL: rows {overall['n']} | pos rate {val['target'].mean():.3f} | flag {overall['flag_rate']:.3f} | "
      f"recall {overall['recall']:.3f} | precision {overall['precision']:.3f} | lift {overall['lift']:.2f}")

pd.set_option("display.width", 200)
for col in ["race", "gender", "age"]:
    rows = []
    for g, part in val.groupby(col):
        m = at_cutoff(part["target"], part["score"].values, bundle["cutoff"])
        patients = part["patient_nbr"].nunique()
        rows.append({col: g, "rows": m["n"], "patients": patients, "positives": m["positives"],
                     "pos_rate": part["target"].mean(), "flag_rate": m["flag_rate"],
                     "recall": m["recall"], "precision": m["precision"], "lift": m["lift"],
                     "note": "UNRELIABLE (<200 patients)" if patients < MIN_PATIENTS else ""})
    print(f"\nBy {col}")
    print(pd.DataFrame(rows).set_index(col).round(3).to_string())

"""Build model/schema_values.json: the allowed input values for src/schema.py.

Category lists come from the TRAINING split only. ID codes come from IDS_mapping.csv.
Does not touch model/pipeline.pkl.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import ROOT, load_splits
from src.prep import fit_cfg

OUT_PATH = ROOT / "model" / "schema_values.json"
IDS_CSV = ROOT / "IDS_mapping.csv"
ID_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
COUNT_COLS = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
              "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]


def read_ids_mapping(path=IDS_CSV):
    """IDS_mapping.csv holds 3 tables one after another, each starting with its own header row."""
    codes, current = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        first = line.split(",", 1)[0].strip()
        if first in ID_COLS:
            current = first
            codes[current] = []
        elif first and current:
            codes[current].append(int(first))
    return {c: sorted(v) for c, v in codes.items()}


def main():
    _, train_df, _, _ = load_splits()

    def train_values(col):
        return sorted(v for v in train_df[col].dropna().unique().tolist() if v != "?")

    values = {
        "source": "training split (GroupShuffleSplit, random_state=42) + IDS_mapping.csv",
        "race": train_values("race"),
        "medical_specialty": train_values("medical_specialty"),
        "payer_code": train_values("payer_code"),
        "id_codes": read_ids_mapping(),
        "id_codes_seen_in_training": {c: sorted(int(v) for v in train_df[c].unique()) for c in ID_COLS},
        # counts above the training maximum are scored, with a warning
        "count_max_in_training": {c: int(train_df[c].max()) for c in COUNT_COLS},
        # raw CSV columns the model drops (near-constant drugs + weight): accepted and ignored quietly
        "unused_raw_columns": fit_cfg(train_df)["drop_cols"],
    }
    OUT_PATH.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    print("saved:", OUT_PATH)
    for k in ["race", "medical_specialty", "payer_code"]:
        print(f"  {k}: {len(values[k])} values")
    for c in ID_COLS:
        print(f"  {c}: IDS_mapping {values['id_codes'][c]}")
        print(f"  {'':{len(c)}}  training   {values['id_codes_seen_in_training'][c]}")


if __name__ == "__main__":
    main()

"""Turn RAW rows of diabetic_data.csv into the feature table the model uses.

Rebuilt from Diabetes.ipynb cells 9-31. Fixed rules are module constants.
Values learned from data come from fit_cfg(train_raw), on TRAINING rows only.
prepare() follows the notebook exactly; it does not add extra input checks.
"""
import logging
import math

import numpy as np
import pandas as pd

log = logging.getLogger("readmission")

# Columns that must not be features: the two ids and the two labels
ID_LABEL_COLS = ["encounter_id", "patient_nbr", "readmitted", "target"]

# Columns where "?" means missing
Q_COLS = ["race", "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3"]
# Columns where NaN (or the text "None") means "test was not done"
TEST_COLS = ["max_glu_serum", "A1Cresult"]
DIAG_COLS = ["diag_1", "diag_2", "diag_3"]


def make_map(groups):
    lookup = {}
    for group, codes in groups.items():
        for c in codes:
            lookup[c] = group
    return lookup


# ID code -> group, using IDS_mapping.csv. Any code not listed becomes "Other"
ID_MAPS = {
    "admission_type_id": make_map({
        "Emergency": [1], "Urgent": [2], "Elective": [3],
        "Unknown": [5, 6, 8],
    }),
    "discharge_disposition_id": make_map({
        "Home": [1], "HomeHealth": [6, 8], "LeftAMA": [7],
        "Facility": [2, 3, 4, 5, 22, 23, 24, 29, 30],
        "Unknown": [18, 25, 26],
    }),
    "admission_source_id": make_map({
        "Emergency": [7], "Referral": [1, 2, 3],
        "Transfer": [4, 5, 6, 10, 18, 22, 25, 26],
        "Unknown": [9, 15, 17, 20, 21],
    }),
}

AGE_MAP = {"[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
           "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95}

# The 7 drugs where Up and Down are rare (insulin is kept as it is)
RARE_MEDS = ["metformin", "repaglinide", "glimepiride", "glipizide",
             "glyburide", "pioglitazone", "rosiglitazone"]

VISIT_COLS = ["number_outpatient", "number_emergency", "number_inpatient"]

CAT_COLS = ["race", "gender", "payer_code", "medical_specialty",
            "diag_1", "diag_2", "diag_3",
            "admission_type_id", "discharge_disposition_id", "admission_source_id",
            "max_glu_serum", "A1Cresult",
            "metformin", "repaglinide", "glimepiride", "glipizide",
            "glyburide", "pioglitazone", "rosiglitazone", "insulin"]

KEEP_COLS = ["medical_specialty", "payer_code"]
NEAR_CONSTANT_SHARE = 0.99
MIN_COUNT = 500


def group_diag(code):
    """Turn a diagnosis code into a broad disease group (fixed ICD-9 ranges).

    Never raises: None, NaN and codes that are not a number become "Unknown" (logged).
    A number such as 428 is read as the code "428". Clean codes give the same group as the notebook.
    """
    if code is None or (isinstance(code, float) and math.isnan(code)):
        log.warning("diagnosis code missing -> Unknown")
        return "Unknown"
    code = str(code).strip().upper()
    if code in ("UNKNOWN", "?"):
        return "Unknown"
    if code.startswith(("V", "E")):          # special codes, not diseases
        return "Other"
    try:
        n = float(code)
    except ValueError:
        log.warning("diagnosis code %r is not a number -> Unknown", code)
        return "Unknown"
    if not math.isfinite(n):
        log.warning("diagnosis code %r is not finite -> Unknown", code)
        return "Unknown"
    if 250 <= n < 251:                       return "Diabetes"
    if 390 <= n <= 459 or n == 785:          return "Circulatory"
    if 460 <= n <= 519 or n == 786:          return "Respiratory"
    if 520 <= n <= 579 or n == 787:          return "Digestive"
    if 580 <= n <= 629 or n == 788:          return "Genitourinary"
    if 710 <= n <= 739:                      return "Musculoskeletal"
    if 800 <= n <= 999:                      return "Injury"
    if 140 <= n <= 239:                      return "Neoplasms"
    return "Other"


def fit_cfg(train_raw):
    """Learn the data-dependent values from TRAINING rows only (cells 10, 11, 25)."""
    X = train_raw.drop(columns=ID_LABEL_COLS, errors="ignore")

    top_share = X.apply(lambda c: c.value_counts(normalize=True, dropna=False).iloc[0])
    drop_cols = top_share[top_share >= NEAR_CONSTANT_SHARE].index.tolist() + ["weight"]

    # keep-lists are counted after "?" -> "Unknown", as in the notebook
    keep = {}
    for c in KEEP_COLS:
        counts = X[c].replace("?", "Unknown").value_counts()
        keep[c] = counts[counts >= MIN_COUNT].index.tolist()

    return {
        "drop_cols": drop_cols,
        "keep": keep,
        "gender_mode": X["gender"].mode()[0],
    }


def prepare(raw_df, cfg, columns=None):
    """RAW rows -> feature table. Does not change raw_df. columns=None skips the reindex."""
    X = raw_df.copy()
    X = X.drop(columns=ID_LABEL_COLS + cfg["drop_cols"], errors="ignore")

    X[Q_COLS] = X[Q_COLS].replace("?", "Unknown")
    X[TEST_COLS] = X[TEST_COLS].replace("None", np.nan).fillna("NotTested")

    for c in DIAG_COLS:
        X[c] = X[c].map(group_diag)

    for c, m in ID_MAPS.items():
        X[c] = X[c].map(m).fillna("Other")

    X["age"] = X["age"].map(AGE_MAP)

    X[RARE_MEDS] = X[RARE_MEDS].replace({"Up": "Changed", "Down": "Changed"})
    X["change"] = (X["change"] == "Ch").astype(int)
    X["diabetesMed"] = (X["diabetesMed"] == "Yes").astype(int)

    for c, keep in cfg["keep"].items():
        X[c] = X[c].where(X[c].isin(keep), "Other")
    X["gender"] = X["gender"].replace("Unknown/Invalid", cfg["gender_mode"])

    X[VISIT_COLS] = np.log1p(X[VISIT_COLS])

    X = pd.get_dummies(X, columns=CAT_COLS, dtype=int)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X

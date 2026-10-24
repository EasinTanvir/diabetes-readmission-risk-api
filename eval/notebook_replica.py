"""Cell-by-cell copy of the Diabetes.ipynb preparation (cells 9-31), in place on 3 parts.

Used only as the reference for checking src/prep.py with assert_frame_equal.
"""
import numpy as np
import pandas as pd


def replica(train_df, val_df, test_df):
    # [9]
    drop_cols = ["encounter_id", "patient_nbr", "readmitted", "target"]
    X_train = train_df.drop(columns=drop_cols)
    X_val = val_df.drop(columns=drop_cols)
    X_test = test_df.drop(columns=drop_cols)

    # [10]
    info = pd.DataFrame({
        "unique": X_train.nunique(),
        "top_share": X_train.apply(lambda c: c.value_counts(normalize=True, dropna=False).iloc[0]).round(3),
        "question": (X_train == "?").mean().round(3),
    })

    # [11]
    near_constant = info[info["top_share"] >= 0.99].index.tolist()
    drop_more = near_constant + ["weight"]
    X_train = X_train.drop(columns=drop_more)
    X_val = X_val.drop(columns=drop_more)
    X_test = X_test.drop(columns=drop_more)

    # [12]
    q_cols = ["race", "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3"]
    test_cols = ["max_glu_serum", "A1Cresult"]
    for part in [X_train, X_val, X_test]:
        part[q_cols] = part[q_cols].replace("?", "Unknown")
        part[test_cols] = part[test_cols].fillna("NotTested")

    # [14]
    def group_diag(code):
        if code == "Unknown":
            return "Unknown"
        if code.startswith(("V", "E")):
            return "Other"
        n = float(code)
        if 250 <= n < 251:                       return "Diabetes"
        if 390 <= n <= 459 or n == 785:          return "Circulatory"
        if 460 <= n <= 519 or n == 786:          return "Respiratory"
        if 520 <= n <= 579 or n == 787:          return "Digestive"
        if 580 <= n <= 629 or n == 788:          return "Genitourinary"
        if 710 <= n <= 739:                      return "Musculoskeletal"
        if 800 <= n <= 999:                      return "Injury"
        if 140 <= n <= 239:                      return "Neoplasms"
        return "Other"

    for part in [X_train, X_val, X_test]:
        for c in ["diag_1", "diag_2", "diag_3"]:
            part[c] = part[c].map(group_diag)

    # [17]
    def make_map(groups):
        lookup = {}
        for group, codes in groups.items():
            for c in codes:
                lookup[c] = group
        return lookup

    admission_type_map = make_map({
        "Emergency": [1], "Urgent": [2], "Elective": [3],
        "Unknown": [5, 6, 8],
    })
    discharge_map = make_map({
        "Home": [1], "HomeHealth": [6, 8], "LeftAMA": [7],
        "Facility": [2, 3, 4, 5, 22, 23, 24, 29, 30],
        "Unknown": [18, 25, 26],
    })
    source_map = make_map({
        "Emergency": [7], "Referral": [1, 2, 3],
        "Transfer": [4, 5, 6, 10, 18, 22, 25, 26],
        "Unknown": [9, 15, 17, 20, 21],
    })
    for part in [X_train, X_val, X_test]:
        part["admission_type_id"] = part["admission_type_id"].map(admission_type_map).fillna("Other")
        part["discharge_disposition_id"] = part["discharge_disposition_id"].map(discharge_map).fillna("Other")
        part["admission_source_id"] = part["admission_source_id"].map(source_map).fillna("Other")

    # [20]
    age_map = {"[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
               "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95}
    for part in [X_train, X_val, X_test]:
        part["age"] = part["age"].map(age_map)

    # [22]
    rare_meds = ["metformin", "repaglinide", "glimepiride", "glipizide",
                 "glyburide", "pioglitazone", "rosiglitazone"]
    for part in [X_train, X_val, X_test]:
        part[rare_meds] = part[rare_meds].replace({"Up": "Changed", "Down": "Changed"})
        part["change"] = (part["change"] == "Ch").astype(int)
        part["diabetesMed"] = (part["diabetesMed"] == "Yes").astype(int)

    # [25]
    min_count = 500
    keep = {}
    for c in ["medical_specialty", "payer_code"]:
        counts = X_train[c].value_counts()
        keep[c] = counts[counts >= min_count].index.tolist()
    gender_mode = X_train["gender"].mode()[0]
    for part in [X_train, X_val, X_test]:
        for c in keep:
            part[c] = part[c].where(part[c].isin(keep[c]), "Other")
        part["gender"] = part["gender"].replace("Unknown/Invalid", gender_mode)

    # [29]
    visit_cols = ["number_outpatient", "number_emergency", "number_inpatient"]
    for part in [X_train, X_val, X_test]:
        part[visit_cols] = np.log1p(part[visit_cols])

    # [31]
    cat_cols = ["race", "gender", "payer_code", "medical_specialty",
                "diag_1", "diag_2", "diag_3",
                "admission_type_id", "discharge_disposition_id", "admission_source_id",
                "max_glu_serum", "A1Cresult",
                "metformin", "repaglinide", "glimepiride", "glipizide",
                "glyburide", "pioglitazone", "rosiglitazone", "insulin"]
    X_train_enc = pd.get_dummies(X_train, columns=cat_cols, dtype=int)
    X_val_enc = pd.get_dummies(X_val, columns=cat_cols, dtype=int).reindex(columns=X_train_enc.columns, fill_value=0)
    X_test_enc = pd.get_dummies(X_test, columns=cat_cols, dtype=int).reindex(columns=X_train_enc.columns, fill_value=0)

    nb_cfg = {"drop_cols": drop_more, "keep": keep, "gender_mode": gender_mode}
    return X_train_enc, X_val_enc, X_test_enc, nb_cfg

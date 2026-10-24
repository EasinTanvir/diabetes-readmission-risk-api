"""Load the raw data, remove died/hospice rows, add the target and split by patient.

Same order and settings as Diabetes.ipynb cells 1, 5-7.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "diabetic_data.csv"

# Discharge codes for died (11, 19, 20, 21) and hospice (13, 14)
DIED_OR_HOSPICE = [11, 13, 14, 19, 20, 21]
SEED = 42


def load(path=DATA_CSV):
    return pd.read_csv(path)


def remove_died_hospice(df):
    return df[~df["discharge_disposition_id"].isin(DIED_OR_HOSPICE)].copy()


def make_target(df):
    # 1 = readmitted within 30 days, 0 = everything else
    df["target"] = (df["readmitted"] == "<30").astype(int)
    return df


def split(df):
    """70/15/15 by patient: first 70/30, then the 30% in half (both random_state=42)."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, temp_idx = next(gss1.split(df, df["target"], groups=df["patient_nbr"]))
    train_df, temp_df = df.iloc[train_idx], df.iloc[temp_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_idx, test_idx = next(gss2.split(temp_df, temp_df["target"], groups=temp_df["patient_nbr"]))
    return train_df, temp_df.iloc[val_idx], temp_df.iloc[test_idx]


def load_splits():
    df = make_target(remove_died_hospice(load()))
    return (df, *split(df))

"""Step 3: rebuild the data, check the split, check prepare() against the notebook replica."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pandas.testing import assert_frame_equal

from src.data import load, load_splits
from src.prep import fit_cfg, prepare
from eval.notebook_replica import replica

raw = load()
df, train_df, val_df, test_df = load_splits()
print("raw rows:", len(raw), "| after died/hospice removal:", len(df))
for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"{name}: rows {len(part)} | positives {part['target'].mean():.1%} | patients {part['patient_nbr'].nunique()}")

p = {n: set(d["patient_nbr"]) for n, d in [("train", train_df), ("val", val_df), ("test", test_df)]}
print("overlap train&val:", len(p["train"] & p["val"]),
      "| train&test:", len(p["train"] & p["test"]),
      "| val&test:", len(p["val"] & p["test"]))

assert len(df) == 99343
assert (len(train_df), len(val_df), len(test_df)) == (69519, 14911, 14913)
assert not (p["train"] & p["val"]) and not (p["train"] & p["test"]) and not (p["val"] & p["test"])

# prepare() on raw rows vs the notebook's step-by-step version
cfg = fit_cfg(train_df)
X_train = prepare(train_df, cfg)
columns = X_train.columns.tolist()
X_val = prepare(val_df, cfg, columns)
X_test = prepare(test_df, cfg, columns)

nb_train, nb_val, nb_test, nb_cfg = replica(train_df, val_df, test_df)
print("cfg equal to notebook:", cfg == nb_cfg)
print("drop_cols:", len(cfg["drop_cols"]), cfg["drop_cols"])
print("keep:", cfg["keep"])
print("gender_mode:", cfg["gender_mode"])

for name, mine, nb in [("train", X_train, nb_train), ("val", X_val, nb_val), ("test", X_test, nb_test)]:
    assert_frame_equal(mine, nb)
    print(f"{name}: shape {mine.shape} | NaN {int(mine.isnull().sum().sum())} | "
          f"object cols {mine.select_dtypes(include='object').columns.tolist()} | equal to notebook: True")

assert len(columns) == 122
print("ALL STEP 3 CHECKS PASSED")

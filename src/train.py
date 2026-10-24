"""Step 4: train the final model with the notebook's chosen settings (no search) and save it.

Cutoff = 90th percentile of VALIDATION scores (flags the top 10%).
The score is for ranking only: class_weight="balanced" inflates it, it is not a probability.
"""
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import SEED, load_splits
from src.prep import fit_cfg, prepare

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "pipeline.pkl"

# Best settings from RandomizedSearchCV in notebook cell 39
PARAMS = dict(min_samples_leaf=50, max_leaf_nodes=31, max_iter=200, learning_rate=0.05,
              l2_regularization=10, class_weight="balanced", early_stopping=True,
              random_state=SEED)
FLAG_SHARE = 0.10


def main():
    _, train_df, val_df, _ = load_splits()

    cfg = fit_cfg(train_df)
    X_train = prepare(train_df, cfg)
    columns = X_train.columns.tolist()
    X_val = prepare(val_df, cfg, columns)

    model = HistGradientBoostingClassifier(**PARAMS).fit(X_train, train_df["target"])
    val_scores = model.predict_proba(X_val)[:, 1]
    cutoff = float(np.quantile(val_scores, 1 - FLAG_SHARE))

    bundle = {
        "model": model,
        "cutoff": cutoff,
        "columns": columns,
        "cfg": cfg,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__,
                     "joblib": joblib.__version__},
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    # Reload and re-score validation: must give the same cutoff
    loaded = joblib.load(MODEL_PATH)
    re_scores = loaded["model"].predict_proba(prepare(val_df, loaded["cfg"], loaded["columns"]))[:, 1]
    assert np.array_equal(re_scores, val_scores)
    print("saved:", MODEL_PATH)
    print("iterations used (early stopping):", model.n_iter_)
    print("cutoff:", round(cutoff, 4))
    print("versions:", bundle["versions"])
    print("reload check: identical validation scores")


if __name__ == "__main__":
    main()

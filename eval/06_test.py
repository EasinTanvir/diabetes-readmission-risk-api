"""Step 6: ONE read-only run on the test set with the fixed validation cutoff.

The test set was already used once in the notebook. Nothing is tuned or changed from this result.
"""
from metrics import at_cutoff, check, load_bundle, ranking

from src.data import load_splits
from src.prep import prepare

bundle = load_bundle()
_, _, _, test_df = load_splits()
y = test_df["target"].values
scores = bundle["model"].predict_proba(prepare(test_df, bundle["cfg"], bundle["columns"]))[:, 1]

r = ranking(y, scores)
m = at_cutoff(y, scores, bundle["cutoff"])
print("TEST (n=%d, positives=%d), cutoff %.4f from validation" % (m["n"], m["positives"], bundle["cutoff"]))
oks = [
    check("flagged", m["flag_rate"], 0.099),
    check("recall", m["recall"], 0.22),
    check("precision", m["precision"], 0.25),
    check("lift", m["lift"], 2.2, tol=0.1),
    check("ROC-AUC", r["roc_auc"], 0.671),
    check("avg precision", r["avg_precision"], 0.211),
]
print("  confusion:\n", m["confusion"])
print("  expected confusion: [[12144 1111] [1288 370]]")
print("ALL WITHIN TOLERANCE" if all(oks) else "SOME NUMBERS DIFFER - STOP")

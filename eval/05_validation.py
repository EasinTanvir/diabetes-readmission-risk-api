"""Step 5: validation report for the saved model, plus the "never flag anyone" baseline."""
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score

from metrics import at_cutoff, check, load_bundle, ranking

from src.data import load_splits
from src.prep import prepare

bundle = load_bundle()
_, _, val_df, _ = load_splits()
y = val_df["target"].values
scores = bundle["model"].predict_proba(prepare(val_df, bundle["cfg"], bundle["columns"]))[:, 1]

r = ranking(y, scores)
m = at_cutoff(y, scores, bundle["cutoff"])
print("VALIDATION (n=%d, positives=%d)" % (m["n"], m["positives"]))
oks = [
    check("ROC-AUC", r["roc_auc"], 0.67),
    check("avg precision", r["avg_precision"], 0.230),
    check("cutoff", bundle["cutoff"], 0.641),
    check("recall", m["recall"], 0.24),
    check("precision", m["precision"], 0.28),
    check("lift", m["lift"], 2.4, tol=0.1),
]
print(f"  flagged        {m['flag_rate']:.1%}")
print("  confusion:\n", m["confusion"])
print("  expected confusion: [[12100 1071] [1319 421]]")

print("\nTop-k shares on validation (cutoff re-derived per share, for reference only)")
for share in [0.05, 0.10, 0.20, 0.30]:
    k = at_cutoff(y, scores, np.quantile(scores, 1 - share))
    print(f"  top {int(share * 100):>2}%: recall {k['recall']:.2f} | precision {k['precision']:.2f} | lift {k['lift']:.1f}x")

print("\nBaseline: never flag anyone")
never = np.zeros_like(y)
oks += [
    check("accuracy", accuracy_score(y, never), 0.883),
    check("avg precision", average_precision_score(y, never), 0.117),
]
print("ALL WITHIN TOLERANCE" if all(oks) else "SOME NUMBERS DIFFER - STOP")

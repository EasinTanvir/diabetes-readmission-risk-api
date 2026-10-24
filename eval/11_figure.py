"""README figure: precision and recall against the share of patients flagged (VALIDATION only).

Writes docs/flag_tradeoff.png and prints the plotted points. Needs requirements-dev.txt (matplotlib).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics import ROOT, at_cutoff, load_bundle

from src.data import load_splits
from src.prep import prepare

# reference palette (dataviz skill), validated: categorical slots 1-2 on the light surface
SURFACE, INK, INK_2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"

bundle = load_bundle()
_, _, val_df, _ = load_splits()
y = val_df["target"].values
scores = bundle["model"].predict_proba(prepare(val_df, bundle["cfg"], bundle["columns"]))[:, 1]
base_rate = y.mean()

shares = np.arange(1, 51) / 100
pts = [at_cutoff(y, scores, np.quantile(scores, 1 - s)) for s in shares]
precision = np.array([p["precision"] for p in pts])
recall = np.array([p["recall"] for p in pts])
marks = [0.05, 0.10, 0.20, 0.30]

fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

ax.plot(shares * 100, precision, color=BLUE, lw=2, label="Precision (flagged who return)")
ax.plot(shares * 100, recall, color=ORANGE, lw=2, label="Recall (returners found)")
for s in marks:
    i = int(round(s * 100)) - 1
    # precision labels above the point, recall labels below, so they never collide
    for v, c, dy, va in [(precision[i], BLUE, 9, "bottom"), (recall[i], ORANGE, -11, "top")]:
        ax.plot(s * 100, v, "o", ms=8, color=c, mec=SURFACE, mew=2, zorder=3)
        ax.annotate(f"{v:.2f}", (s * 100, v), textcoords="offset points", xytext=(0, dy),
                    ha="center", va=va, fontsize=8.5, color=INK_2)

ax.axhline(base_rate, color=INK_2, lw=1, ls=(0, (4, 3)))
ax.text(50, base_rate - 0.025, f"random flagging precision = {base_rate:.3f}", ha="right", fontsize=8.5, color=INK_2)
ax.axvline(10, color=AXIS, lw=1)
ax.text(10.6, 0.75, "chosen: top 10%", fontsize=8.5, color=INK_2)

ax.set_xlim(0, 51)
ax.set_ylim(0, 0.8)
ax.set_xlabel("Share of patients flagged (%)", color=INK)
ax.set_ylabel("Share (0-1)", color=INK)
ax.set_title("Flagging more patients finds more returners, at lower precision (validation)",
             fontsize=11, color=INK, loc="left")
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
for side in ["left", "bottom"]:
    ax.spines[side].set_color(AXIS)
ax.tick_params(colors=INK_2, labelsize=9)
leg = ax.legend(loc="center right", frameon=False, fontsize=9)
for t in leg.get_texts():
    t.set_color(INK)

out = ROOT / "docs" / "flag_tradeoff.png"
out.parent.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(out, facecolor=SURFACE)
print("saved:", out.relative_to(ROOT).as_posix())
for s in marks:
    i = int(round(s * 100)) - 1
    print(f"  top {int(s * 100):>2}%: precision {precision[i]:.3f} | recall {recall[i]:.3f}")

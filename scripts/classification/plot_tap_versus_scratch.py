"""Figure for scripts/hub/tap_versus_scratch.py (Greatest Hits, hits and scratches on the same objects).

tap_versus_scratch.png: family score for each (trained on, tested on) pair; dots are the 5 video folds.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/classification/tap_versus_scratch"
CHANCE = 0.25

scores = pd.read_csv(OUT / "scores.csv")
pairs = [("hit", "hit"), ("scratch", "scratch"), ("hit", "scratch"), ("scratch", "hit")]
colors = ["#2a78d6", "#eb6834", "0.5", "0.5"]

fig, ax = plt.subplots(figsize=(7, 4.5))
for number, (trained_on, tested_on) in enumerate(pairs):
    values = scores.score[(scores.trained_on == trained_on) & (scores.tested_on == tested_on)]
    ax.bar(number, values.mean(), color=colors[number], width=0.6)
    ax.scatter([number] * len(values), values, color="0.15", s=14, zorder=3)
ax.axhline(CHANCE, color="0.4", linestyle=":", linewidth=1)
ax.set_xticks(range(len(pairs)), [f"{a} -> {b}" for a, b in pairs])
ax.set(
    ylim=(0, 1), ylabel="balanced accuracy, 4 families", title="Greatest Hits: a hit tells a little more than a scratch"
)
ax.grid(axis="y", color="0.9")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "tap_versus_scratch.png", dpi=150)

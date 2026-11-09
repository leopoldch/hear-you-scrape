"""Figure for scripts/classification/devillard_same_surfaces.py (same 10 textures, 10 people, 4 sensors).

devillard_same_surfaces.png
  left: texture recognition (chance 0.10) for known people, a new person, a new sensor of the same kind,
        and acceleration <-> audio, all without labels of the new person or sensor.
  right: mean cosine between texture directions, across people and across sensors.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/classification/devillard_same_surfaces"
CHANCE = 0.10

scores = pd.read_csv(OUT / "scores.csv")
cosines = pd.read_csv(OUT / "direction_cosines.csv")


def is_accel_audio_pair(row):
    one_is_mic = "microphone" in row["trained on"]
    other_is_mic = "microphone" in row["tested on"]
    return one_is_mic != other_is_mic


def group_of(row):
    if row.test == "known people, new trial":
        return "known people,\nnew trial"
    if row.test == "new person, k = 0":
        return "new person"
    if row.test == "new sensor, k = 0, same people":
        if is_accel_audio_pair(row):
            return "acceleration\n<-> audio"
        return "new sensor,\nsame kind"
    return None


scores["group"] = scores.apply(group_of, axis=1)
groups = ["known people,\nnew trial", "new person", "new sensor,\nsame kind", "acceleration\n<-> audio"]
colors = ["#2a78d6", "#2a78d6", "#eb6834", "0.5"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for number, group in enumerate(groups):
    values = scores.score[scores.group == group]
    axes[0].bar(number, values.mean(), color=colors[number], width=0.6)
    axes[0].scatter([number] * len(values), values, color="0.15", s=14, zorder=3)
axes[0].axhline(CHANCE, color="0.4", linestyle=":", linewidth=1)
axes[0].set_xticks(range(len(groups)), groups)
axes[0].set(ylim=(0, 1), ylabel="balanced accuracy, 10 textures", title="Same surfaces: transfer works without labels")

labels = {
    "across people": "across people",
    "same person, different trials": "same person,\nother trials",
}
cosines["label"] = cosines.comparison.map(labels)
cosines.loc[cosines.comparison == "across sensors", "label"] = cosines.sensor.str.replace(" vs ", "\nvs ")
per_label = cosines.groupby("label", sort=False).cosine.mean()
axes[1].bar(range(len(per_label)), per_label, color="#6250d6", width=0.6)
axes[1].set_xticks(range(len(per_label)), per_label.index, fontsize=8)
axes[1].set(
    ylim=(-0.1, 1), ylabel="cosine of texture directions", title="Directions stay aligned, except accel vs audio"
)
axes[1].axhline(0, color="0.4", linewidth=1)

for ax in axes:
    ax.grid(axis="y", color="0.9")
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "devillard_same_surfaces.png", dpi=150)

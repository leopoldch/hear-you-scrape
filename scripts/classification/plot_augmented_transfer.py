"""Figure for H6 (scripts/hub/augmented_beats_transfer.py, run on the hub; CSV downloaded here).

augmented_transfer.png: fine-tuned BEATs with random-setup augmentation. Left panel per input:
score on the new setup (no labels). Right panel: control on new surfaces of the known setups.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/classification/augmentation"
SCORES = pd.read_csv(OUT / "augmented_transfer_scores.csv")
CHANCE = 0.25
COLORS = {"none": "#2a78d6", "medium": "#eb6834", "strong": "#1baf7a"}
NEW_SETUPS = ["cluster", "lmt108", "sens3"]
INPUTS = ["audio", "audio + acceleration"]


def main():
    fig, axes = plt.subplots(len(INPUTS), 2, figsize=(12, 8), sharey=True)
    for row, input_name in enumerate(INPUTS):
        for column, on_new_setup in enumerate([True, False]):
            ax = axes[row, column]
            here = SCORES[(SCORES.input == input_name) & (SCORES.test.str.startswith("new setup") == on_new_setup)]
            for shift, level in zip([-0.25, 0, 0.25], COLORS):
                bars = here[here.augmentation == level].set_index("new setup").loc[NEW_SETUPS]
                x = np.arange(len(NEW_SETUPS)) + shift
                ax.bar(x, bars["balanced accuracy"], width=0.23, color=COLORS[level], label=level)
                ax.errorbar(
                    x,
                    bars["balanced accuracy"],
                    yerr=[bars["balanced accuracy"] - bars.low, bars.high - bars["balanced accuracy"]],
                    fmt="none",
                    ecolor="0.2",
                    capsize=3,
                    linewidth=1,
                )
            ax.axhline(CHANCE, color="0.4", linestyle=":", linewidth=1)
            ax.set_xticks(range(len(NEW_SETUPS)), [f"{s} held out" for s in NEW_SETUPS])
            title = (
                "test: the held-out setup (no labels)" if on_new_setup else "control: new surfaces of the known setups"
            )
            ax.set(title=f"{input_name}\n{title}", ylim=(0, 1))
            ax.grid(axis="y", color="0.9")
            ax.spines[["top", "right"]].set_visible(False)
            if column == 0:
                ax.set_ylabel("balanced accuracy")
    axes[0, 0].legend(title="augmentation", frameon=False, loc="upper right")
    fig.suptitle("H6: random-setup augmentation does not transfer and hurts known setups; dotted line = chance")
    fig.tight_layout()
    fig.savefig(OUT / "augmented_transfer.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()

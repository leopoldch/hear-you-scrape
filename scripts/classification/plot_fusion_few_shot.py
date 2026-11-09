"""Figure for H7 (scripts/hub/fusion_few_shot_probes.py, run on the hub; CSV downloaded here).

fusion_few_shot.png: score vs k labeled surfaces per family of the new setup, one panel per new
setup, one line per representation, for two ways of using the k surfaces: pooled with the other
setups (solid) and new setup only (dashed). Shaded band = spread over the 10 draws (min to max).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/classification/qwen_audio"
SCORES = pd.read_csv(OUT / "fusion_few_shot_scores.csv")
CHANCE = 0.25
COLORS = {
    "BEATs": "#2a78d6",
    "CLAP": "#eb6834",
    "Qwen encoder 32": "#1baf7a",
    "BEATs + CLAP + Qwen encoder 32": "#6250d6",
}
METHODS = {"pooled": "-", "new setup only": "--"}
NEW_SETUPS = ["cluster", "lmt108", "sens3"]


def main():
    fig, axes = plt.subplots(1, len(NEW_SETUPS) + 1, figsize=(18, 5), sharey=True)
    for column, new_setup in enumerate(NEW_SETUPS + ["mean of the 3"]):
        ax = axes[column]
        here = SCORES if new_setup == "mean of the 3" else SCORES[SCORES["new setup"] == new_setup]
        for representation, color in COLORS.items():
            for method, line in METHODS.items():
                rows = here[(here.representation == representation) & (here.method == method)]
                per_draw = rows.groupby(["k", "draw"])["balanced accuracy"].mean().unstack("draw")
                ax.plot(
                    per_draw.index,
                    per_draw.mean(axis=1),
                    line,
                    marker="o",
                    color=color,
                    linewidth=2,
                    markersize=6,
                    label=f"{representation}, {method}",
                )
                ax.fill_between(per_draw.index, per_draw.min(axis=1), per_draw.max(axis=1), color=color, alpha=0.08)
        ax.axhline(CHANCE, color="0.4", linestyle=":", linewidth=1)
        ax.set(
            title=f"new setup: {new_setup}",
            xlabel="k labeled surfaces per family of the new setup",
            xticks=[0, 1, 2, 4],
            ylim=(0, 1),
        )
        ax.grid(axis="y", color="0.9")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("balanced accuracy")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.suptitle(
        "H7: few-shot calibration, audio + acceleration. "
        "Dashed (new setup only) beats solid (pooled with other setups); "
        "fusion does not beat BEATs"
    )
    fig.tight_layout()
    fig.savefig(OUT / "fusion_few_shot.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()

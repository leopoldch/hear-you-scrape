"""Figure for H16 (scripts/classification/calibrated_new_setup.py).

calibrated_new_setup.png
  left: development (acceleration only), score vs k for the frozen choice (BEATs, new setup only)
        and BEATs pooled, per development setup.
  right: the locked HaTT test, same two curves, plus per-family recall at k = 2.
Lines = mean over draws; shaded band = spread over draws (min to max).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/classification/calibrated_new_setup"
DEVELOPMENT = pd.read_csv(OUT / "development_scores.csv")
HATT = pd.read_csv(OUT / "hatt_locked_test.csv")
CHANCE = 0.25
COLORS = {"cluster": "#2a78d6", "lmt108": "#eb6834", "sens3": "#1baf7a", "hatt": "#6250d6"}
FAMILIES = ["wood", "metal", "textile", "polymer"]


def curve(ax, rows, color, line, label):
    per_k = rows.groupby("k")["balanced accuracy"]
    ax.plot(per_k.mean().index, per_k.mean(), line, marker="o", color=color, linewidth=2, markersize=6, label=label)
    ax.fill_between(per_k.mean().index, per_k.min(), per_k.max(), color=color, alpha=0.1)


def style(ax, title):
    ax.axhline(CHANCE, color="0.4", linestyle=":", linewidth=1)
    ax.set(title=title, ylim=(0, 1))
    ax.grid(axis="y", color="0.9")
    ax.spines[["top", "right"]].set_visible(False)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    beats = DEVELOPMENT[DEVELOPMENT.candidate == "BEATs"]
    for setup in ["cluster", "lmt108", "sens3"]:
        here = beats[beats["new setup"] == setup]
        curve(
            axes[0],
            here[here.method == "new setup only"],
            COLORS[setup],
            "-",
            f"{setup}, new setup only (frozen choice)",
        )
        curve(axes[0], here[here.method == "pooled"], COLORS[setup], "--", f"{setup}, pooled")
    style(axes[0], "development: each setup held out in turn")

    calibrated = HATT[HATT.method == "new setup only"]
    reference = HATT[HATT.method != "new setup only"]
    curve(axes[1], pd.concat([reference, calibrated]), COLORS["hatt"], "-", "HaTT: k = 0 pooled, then HaTT only")
    for k, rows in calibrated.groupby("k"):
        axes[1].errorbar(
            k + 0.05,
            rows["balanced accuracy"].mean(),
            yerr=[
                [rows["balanced accuracy"].mean() - rows.low.mean()],
                [rows.high.mean() - rows["balanced accuracy"].mean()],
            ],
            fmt="none",
            ecolor="0.2",
            capsize=3,
        )
    style(axes[1], "locked test: Penn HaTT (run once)")

    at_two = calibrated[calibrated.k == 2][[f"recall {f}" for f in FAMILIES]].mean()
    at_zero = reference[[f"recall {f}" for f in FAMILIES]].mean()
    x = np.arange(len(FAMILIES))
    axes[2].bar(x - 0.2, at_zero, width=0.38, color="#c3c3c3", label="k = 0 (pooled)")
    axes[2].bar(x + 0.2, at_two, width=0.38, color=COLORS["hatt"], label="k = 2 (HaTT only)")
    axes[2].set_xticks(x, ["wood\n(1 test surface at k = 2)", "metal", "textile", "polymer"])
    style(axes[2], "HaTT recall per family")
    axes[2].set_ylabel("recall")

    for ax in axes[:2]:
        ax.set(xlabel="k labeled surfaces per family of the new setup", xticks=[0, 1, 2])
        ax.set_ylabel("balanced accuracy")
        ax.legend(frameon=False, loc="upper left", fontsize=8)
    axes[2].legend(frameon=False, loc="upper right")
    fig.suptitle(
        "H16: frozen BEATs + a probe trained on k labeled surfaces of the new setup, "
        "acceleration only; dotted line = chance"
    )
    fig.tight_layout()
    fig.savefig(OUT / "calibrated_new_setup.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()

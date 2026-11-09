"""Figures for scripts/hub/setup_transfer_probes.py (run on the hub; CSV downloaded here).

1. qwen_layer_scan.png: linear-probe score per Qwen2-Audio layer, BEATs and CLAP as references.
2. setup_transfer_summary.png: Qwen answering directly vs linear probes, per new setup.
3. few_shot_pretrained.png: adding k labeled surfaces per family of the new setup.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/classification/qwen_audio"
SCORES = pd.read_csv(OUT / "setup_transfer_scores.csv")
CHANCE = 0.25
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INPUTS = ["audio", "acceleration", "audio + acceleration"]
TESTS = ["new surfaces: cluster", "new setup: cluster", "new setup: lmt108", "new setup: sens3"]
NEW_SETUPS = ["cluster", "lmt108", "sens3"]


def style(ax, title):
    ax.axhline(CHANCE, color="0.4", linestyle=":", linewidth=1)
    ax.set(title=title, ylim=(0, 1))
    ax.grid(axis="y", color="0.9")
    ax.spines[["top", "right"]].set_visible(False)


def layer_scan():
    probes = SCORES[SCORES.variant == "linear probe"]
    fig, axes = plt.subplots(len(INPUTS), len(TESTS), figsize=(16, 10), sharey=True)
    for row, input_name in enumerate(INPUTS):
        for column, test in enumerate(TESTS):
            ax = axes[row, column]
            here = probes[(probes.input == input_name) & (probes.test == test)]
            for color, kind in zip(COLORS, ["Qwen encoder", "Qwen LLM last token", "Qwen LLM audio tokens"]):
                curve = here[here.representation.str.startswith(kind + " layer")].copy()
                curve["layer"] = curve.representation.str.rsplit(" ", n=1).str[-1].astype(int)
                curve = curve.sort_values("layer")
                ax.plot(
                    curve.layer, curve["balanced accuracy"], "o-", color=color, linewidth=2, markersize=6, label=kind
                )
                ax.fill_between(curve.layer, curve.low, curve.high, color=color, alpha=0.12)
            for color, name in zip(COLORS[3:], ["BEATs", "CLAP"]):
                value = here[here.representation == name]["balanced accuracy"]
                if len(value):
                    ax.axhline(value.iloc[0], color=color, linewidth=2, linestyle="--", label=name)
            style(ax, f"{input_name}\n{test}")
            if row == len(INPUTS) - 1:
                ax.set_xlabel("layer")
            if column == 0:
                ax.set_ylabel("balanced accuracy")
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.suptitle("Linear probes on frozen representations (LLM in 4 bits); dotted line = chance")
    fig.tight_layout()
    fig.savefig(OUT / "qwen_layer_scan.png", dpi=130, bbox_inches="tight")


def summary():
    chosen = [
        ("Qwen answer", "audio", "raw", "Qwen answers, raw"),
        ("Qwen answer", "audio", "centered on new setup", "Qwen answers, centered on new setup"),
        ("Qwen answer", "audio", "recalibrated on training setups", "Qwen answers, recalibrated"),
        ("Qwen encoder layer 32", "audio + acceleration", "linear probe", "Qwen encoder 32, probe"),
        ("Qwen LLM last token layer 16", "audio + acceleration", "linear probe", "Qwen LLM 16, probe"),
        ("BEATs", "audio + acceleration", "linear probe", "BEATs, probe"),
        ("CLAP", "audio + acceleration", "linear probe", "CLAP, probe"),
        (
            "Qwen encoder layer 32",
            "audio + acceleration",
            "linear probe, per-setup standardization",
            "Qwen encoder 32, per-setup std",
        ),
        ("BEATs", "audio + acceleration", "linear probe, per-setup standardization", "BEATs, per-setup std"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    for ax, new_setup in zip(axes, NEW_SETUPS):
        here = SCORES[SCORES.test == f"new setup: {new_setup}"]
        for position, (representation, input_name, variant, label) in enumerate(chosen):
            row = here[(here.representation == representation) & (here.input == input_name) & (here.variant == variant)]
            if row.empty:
                continue
            row = row.iloc[0]
            color = COLORS[0] if representation == "Qwen answer" else COLORS[1] if "std" in label else COLORS[2]
            ax.bar(position, row["balanced accuracy"], color=color, width=0.7)
            ax.errorbar(
                position,
                row["balanced accuracy"],
                [[row["balanced accuracy"] - row.low], [row.high - row["balanced accuracy"]]],
                color="0.2",
                capsize=3,
                linewidth=1,
            )
        ax.set_xticks(range(len(chosen)), [c[3] for c in chosen], rotation=60, ha="right", fontsize=8)
        style(ax, f"trained on the other two setups, tested on {new_setup}")
    axes[0].set_ylabel("balanced accuracy (95% surface interval)")
    fig.suptitle("Does anything transfer to a new setup? Probes use audio + acceleration; Qwen answers use audio")
    fig.tight_layout()
    fig.savefig(OUT / "setup_transfer_summary.png", dpi=130, bbox_inches="tight")


def few_shot():
    shots = SCORES[SCORES.variant == "linear probe + k labeled surfaces"]
    plain = SCORES[SCORES.variant == "linear probe"]
    representations = list(shots.representation.unique())
    fig, axes = plt.subplots(len(INPUTS), len(NEW_SETUPS), figsize=(14, 10), sharex=True, sharey=True)
    for row, input_name in enumerate(INPUTS):
        for column, new_setup in enumerate(NEW_SETUPS):
            ax = axes[row, column]
            test = f"new setup: {new_setup}"
            for color, representation in zip(COLORS, representations):
                curve = shots[
                    (shots.input == input_name) & (shots.test == test) & (shots.representation == representation)
                ]
                grouped = curve.groupby("k")["balanced accuracy"]
                zero = plain[
                    (plain.input == input_name) & (plain.test == test) & (plain.representation == representation)
                ]
                ks = [0, *grouped.mean().index]
                means = [zero["balanced accuracy"].iloc[0], *grouped.mean()]
                ax.plot(ks, means, "o-", color=color, linewidth=2, markersize=6, label=representation)
                ax.fill_between(
                    grouped.mean().index, grouped.quantile(0.025), grouped.quantile(0.975), color=color, alpha=0.12
                )
            style(ax, f"{input_name}, new setup: {new_setup}")
            if row == len(INPUTS) - 1:
                ax.set_xlabel("labeled surfaces per family of the new setup (k)")
            if column == 0:
                ax.set_ylabel("balanced accuracy on its other surfaces")
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.suptitle("Calibrating a new setup with a few labeled surfaces (mean of 10 draws, 95% band)")
    fig.tight_layout()
    fig.savefig(OUT / "few_shot_pretrained.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    layer_scan()
    summary()
    few_shot()
    print("saved figures in", OUT)

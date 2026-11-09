"""Figure for the simulated devices (scripts/hub/simulated_devices.py and simulated_devices_calibration.py).

simulated_devices.png
  left: inside the simulation, score on 50 unseen simulated devices with no label, against the number of
        simulated devices used for training.
  middle: the same probe (trained on all simulated devices) on the real setups, no label.
  right: k labeled surfaces per family on the real setups, acceleration: H16 logistic probe against
         prototypes with the projection learned on simulated devices.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/classification/simulated_devices"
CHANCE = 0.25
COLORS = {"cluster": "#2a78d6", "lmt108": "#eb6834", "sens3": "#1baf7a", "simulated (unseen devices)": "0.3"}
METHOD_STYLE = {"logistic probe (H16)": "-", "prototypes, learned projection": "--"}


def style(ax, title):
    ax.axhline(CHANCE, color="0.4", linestyle=":", linewidth=1)
    ax.set(title=title, ylim=(0, 1))
    ax.grid(axis="y", color="0.9")
    ax.spines[["top", "right"]].set_visible(False)


scale = pd.read_csv(OUT / "scale_curve.csv")
zero_shot = pd.read_csv(OUT / "real_zero_shot.csv")
few_shot = pd.read_csv(OUT / "real_few_shot.csv")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

per_n = scale.groupby("n_devices").score
axes[0].plot(per_n.mean().index, per_n.mean(), marker="o", color="0.3", linewidth=2)
axes[0].fill_between(per_n.mean().index, per_n.min(), per_n.max(), color="0.3", alpha=0.15)
axes[0].set(xscale="log", xlabel="simulated devices used for training", ylabel="balanced accuracy")
style(axes[0], "In the simulation: one device is almost enough")

signals = ["audio", "accel"]
for number, setup in enumerate(["cluster", "lmt108", "sens3"]):
    for offset, signal in enumerate(signals):
        row = zero_shot[(zero_shot.setup == setup) & (zero_shot.signal == signal)].iloc[0]
        x = number + (offset - 0.5) * 0.35
        hatch = "" if signal == "audio" else "//"
        axes[1].bar(x, row["balanced accuracy"], width=0.33, color=COLORS[setup], hatch=hatch, edgecolor="white")
        axes[1].errorbar(
            x,
            row["balanced accuracy"],
            [[row["balanced accuracy"] - row.low], [row.high - row["balanced accuracy"]]],
            color="0.2",
            capsize=3,
        )
axes[1].set_xticks(range(3), ["Cluster", "LMT108", "SENS3"])
axes[1].set_ylabel("balanced accuracy (plain: audio, hatched: acceleration)")
style(axes[1], "On real setups, no label: chance")

acceleration = few_shot[few_shot.signal.isin(["accel", "simulated"])]
for setup, color in COLORS.items():
    for method, line in METHOD_STYLE.items():
        rows = acceleration[(acceleration.setup == setup) & (acceleration.method == method)]
        axes[2].plot(rows.k, rows.score, line, marker="o", color=color, linewidth=2, label=f"{setup}, {method}")
axes[2].set(xlabel="labeled surfaces per family (k)", ylabel="balanced accuracy", xticks=[1, 2, 4])
axes[2].legend(fontsize=7, frameon=False, loc="lower right")
style(axes[2], "Calibration learned in simulation: helps there, hurts on real setups")

fig.tight_layout()
fig.savefig(OUT / "simulated_devices.png", dpi=150)

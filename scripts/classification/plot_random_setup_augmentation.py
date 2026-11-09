"""Check the "random setup" augmentation of H6 (docs/experiments.md) by eye.

Left: spectrum of one Cluster audio window and of 6 augmented copies, per level.
Right: spread of the augmented spectra around the original (RMS dB difference over 100 copies),
next to the measured setup filter (3.5-7 dB RMS, setup_filter.py).
Output: analysis/classification/augmentation/random_setup_augmentation.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/hub"))
from augmented_beats_transfer import AUGMENTATION_DB, RATE, random_setup  # noqa: E402

OUT = ROOT / "analysis/classification/augmentation"
OUT.mkdir(parents=True, exist_ok=True)
COLORS = {"none": "0.2", "medium": "#2a78d6", "strong": "#eb6834"}


def spectrum_db(x):
    window = torch.hann_window(len(x))
    power = torch.fft.rfft(x * window).abs() ** 2
    smoothed = torch.nn.functional.avg_pool1d(power[None, None], 15, 1, 7)[0, 0]
    return 10 * torch.log10(smoothed + 1e-10).numpy()


def main():
    cache = np.load(ROOT / "analysis/classification/waveforms.npz", mmap_mode="r")
    window = torch.from_numpy(cache["audio"][1000].astype(np.float32))
    freqs = np.fft.rfftfreq(len(window), 1 / RATE)
    generator = torch.Generator().manual_seed(0)
    original = spectrum_db(window)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, level in zip(axes[:2], ["medium", "strong"]):
        copies = random_setup(window.repeat(6, 1), AUGMENTATION_DB[level], generator)
        for copy in copies:
            ax.semilogx(freqs[1:], spectrum_db(copy)[1:], color=COLORS[level], alpha=0.5, linewidth=1)
        ax.semilogx(freqs[1:], original[1:], color=COLORS["none"], linewidth=2, label="original")
        ax.set(
            title=f"{level} (±{AUGMENTATION_DB[level]:.0f} dB): 6 virtual setups",
            xlabel="frequency (Hz)",
            ylabel="power (dB)",
            xlim=(50, RATE / 2),
        )
        ax.legend(frameon=False)
        ax.grid(color="0.9")

    in_band = (freqs > 50) & (freqs < 7000)
    for level in ["medium", "strong"]:
        copies = random_setup(window.repeat(100, 1), AUGMENTATION_DB[level], generator)
        spread = [
            np.sqrt(
                np.mean(
                    (spectrum_db(c)[in_band] - original[in_band] - np.mean(spectrum_db(c)[in_band] - original[in_band]))
                    ** 2
                )
            )
            for c in copies
        ]
        axes[2].hist(spread, bins=20, color=COLORS[level], alpha=0.7, label=level)
    axes[2].axvspan(3.5, 7, color="0.9", label="measured setup filter", zorder=0)
    axes[2].set(title="shape change vs original (RMS dB, 50 Hz-7 kHz)", xlabel="RMS dB", ylabel="augmented copies")
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "random_setup_augmentation.png", dpi=130, bbox_inches="tight")
    print("saved", OUT / "random_setup_augmentation.png")


if __name__ == "__main__":
    main()

"""Which spectral peaks come from the Cluster machine, not from the surface?

Narrow peaks of the median spectrum over all 72 surfaces (see common.machine_lines).
Plotted against f, fixed lines align vertically; plotted against f / v,
lines locked to the motion (drive mechanism) align vertically.
"""

import numpy as np
import pandas as pd
import seaborn as sns

from common import ACCEL_BAND, ACCEL_FREQ, AUDIO_BAND, AUDIO_FREQ, load, machine_lines, save

meta, psd = load(clean=False)
for modality, freq, band, filter_width, max_freq in [
    ("accel", ACCEL_FREQ, ACCEL_BAND, 15, 600),
    ("audio", AUDIO_FREQ, AUDIO_BAND, 41, 1500),
]:
    lines = machine_lines(meta, psd[modality], freq, band, filter_width)
    print(f"\n{modality} machine lines:")
    print(lines.round(1).to_string(index=False))

    shown = (freq >= band[0]) & (freq <= max_freq)
    curves = []
    for velocity in [20, 30, 40, 50, 60]:
        median_spectrum = np.median(psd[modality][meta.velocity == velocity], axis=0)
        curve = pd.DataFrame(
            {
                "velocity": velocity,
                "level (dB)": 10 * np.log10(median_spectrum[shown]),
                "f (Hz)": freq[shown],
                "f / v (cycles per mm)": freq[shown] / velocity,
            }
        )
        curves.append(curve)

    # Long format: the same curve twice, once against f and once against f / v
    curves = pd.concat(curves).melt(id_vars=["velocity", "level (dB)"], var_name="x axis", value_name="x")
    g = sns.relplot(
        curves,
        x="x",
        y="level (dB)",
        row="velocity",
        col="x axis",
        kind="line",
        lw=0.8,
        height=1.6,
        aspect=4,
        facet_kws={"sharex": "col", "sharey": False},
    )
    for ax in g.axes[:, 1]:
        for cycles_per_mm in [0.875, 1.75, 3.5, 7, 14]:
            ax.axvline(cycles_per_mm, color="red", ls=":", lw=0.8)
    g.figure.suptitle(f"Cluster {modality}: median over surfaces. Red dotted: multiples of 0.875 / mm", y=1.01)
    save(g.figure, f"setup_lines_{modality}.png")

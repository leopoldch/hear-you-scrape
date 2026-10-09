"""How does the Cluster spectrum change with sliding velocity?

Texture model: a spatial frequency k gives f = k v, so from 20 to 60 mm/s the
spectrum should move up by a factor 3. If resonances (probe, sensor) dominate,
it stays at the same frequencies. For each surface / direction / force / repeat
we correlate the 20 and 60 mm/s log spectra in two ways:
  same f        S60(f)   vs S20(f)    high if the spectrum stays put
  same f / v    S60(3 f) vs S20(f)    high if the spectrum follows f = k v
We also measure the level change per doubling of velocity.
"""

import numpy as np
import pandas as pd
import seaborn as sns

from common import ACCEL_FREQ, AUDIO_FREQ, FAMILIES, PALETTE, load, save, third_octaves

meta, clean_psd = load()
_, raw_psd = load(clean=False)
GROUP = ["surface", "family", "direction", "force", "repeat"]

examples = []
for surface in ["0", "65", "102"]:
    chosen = (meta.surface == surface) & (meta.direction == 0) & (meta.force == 500) & (meta.repeat == 0)
    for modality, freq, band in [("audio", AUDIO_FREQ, (100, 8000)), ("accel", ACCEL_FREQ, (10, 1000))]:
        levels = third_octaves(raw_psd[modality][chosen], freq, band)
        levels.index = meta.velocity[chosen]
        long = levels.stack().rename_axis(["velocity", "f (Hz)"]).reset_index(name="dB")
        long["surface"] = surface
        long["modality"] = modality
        examples.append(long)

g = sns.relplot(
    pd.concat(examples),
    x="f (Hz)",
    y="dB",
    hue="velocity",
    col="surface",
    row="modality",
    kind="line",
    palette="viridis",
    height=3,
    aspect=1.4,
    facet_kws={"sharey": False, "sharex": False},
)
g.set(xscale="log")
save(g.figure, "velocity_examples.png")


def compare_20_and_60(group, modality, freq, band):
    grid = np.geomspace(band[0], band[1] / 3, 200)
    at_20 = group[group.velocity == 20].index[0]
    at_60 = group[group.velocity == 60].index[0]
    log_psd_20 = np.log(clean_psd[modality][at_20])
    log_psd_60 = np.log(clean_psd[modality][at_60])

    s20 = np.interp(grid, freq, log_psd_20)
    s60_same_f = np.interp(grid, freq, log_psd_60)
    s60_same_f_over_v = np.interp(3 * grid, freq, log_psd_60)
    level_change_db = 20 * (meta[f"{modality}_log_rms"][at_60] - meta[f"{modality}_log_rms"][at_20])

    return {
        "corr, same f": np.corrcoef(s20, s60_same_f)[0, 1],
        "corr, same f / v": np.corrcoef(s20, s60_same_f_over_v)[0, 1],
        "dB per doubling of v": level_change_db / np.log2(3),
    }


results = []
for modality, freq, band in [("audio", AUDIO_FREQ, (100, 8000)), ("accel", ACCEL_FREQ, (10, 600))]:
    for keys, group in meta[meta.dataset == "cluster"].groupby(GROUP):
        row = dict(zip(GROUP, keys, strict=True))
        row["modality"] = modality
        row.update(compare_20_and_60(group, modality, freq, band))
        results.append(row)
results = pd.DataFrame(results)

measures = ["corr, same f", "corr, same f / v", "dB per doubling of v"]
print(results.groupby(["modality", "family"])[measures].median().round(2))

g = sns.catplot(
    results.melt(id_vars=GROUP + ["modality"], value_vars=measures, var_name="measure"),
    x="family",
    y="value",
    hue="family",
    col="measure",
    row="modality",
    kind="box",
    palette=PALETTE,
    order=FAMILIES,
    showfliers=False,
    sharey=False,
    height=3,
    aspect=1.2,
)
save(g.figure, "velocity_scaling.png")

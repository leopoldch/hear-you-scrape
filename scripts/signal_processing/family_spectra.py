"""Average spectral shape of each family, in each dataset (Cluster machine lines removed).

One curve per surface (median over its recordings, mean level removed), then the
family median and interquartile range. A second figure shows every surface.
"""

import seaborn as sns

from common import FAMILIES, PALETTE, load, save, shapes

meta, psd = load()
per_surface = shapes(psd).groupby([meta.dataset, meta.family, meta.surface]).median()

# Long format for seaborn: one row per (surface, band); band names look like "audio 125"
long = per_surface.stack().rename_axis(["dataset", "family", "surface", "band"]).reset_index(name="relative level (dB)")
long["modality"] = long.band.str.split(" ").str[0]
long["f (Hz)"] = long.band.str.split(" ").str[1].astype(float)

g = sns.relplot(
    long,
    x="f (Hz)",
    y="relative level (dB)",
    hue="family",
    row="dataset",
    col="modality",
    kind="line",
    estimator="median",
    errorbar=("pi", 50),
    palette=PALETTE,
    hue_order=FAMILIES,
    height=3.5,
    aspect=1.5,
    facet_kws={"sharex": False},
)
g.set(xscale="log")
save(g.figure, "family_spectra.png")

for modality in ["audio", "accel"]:
    g = sns.relplot(
        long[long.modality == modality],
        x="f (Hz)",
        y="relative level (dB)",
        hue="family",
        units="surface",
        estimator=None,
        row="dataset",
        col="family",
        kind="line",
        lw=0.7,
        palette=PALETTE,
        col_order=FAMILIES,
        height=2.5,
        aspect=1.3,
    )
    g.set(xscale="log")
    save(g.figure, f"all_surfaces_{modality}.png")

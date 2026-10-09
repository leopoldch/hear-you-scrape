"""Simple descriptors: which ones separate the families, and do they agree across datasets?

  centroid           geometric-mean frequency of the PSD (Hz)
  slope              line fitted to the 1/3-octave levels vs log2 f (dB per octave)
  hf_lf              energy above / below a split frequency (2 kHz audio, 200 Hz accel), dB
  flatness           geometric / arithmetic mean of the PSD (1 = white noise)
  peakiness          99th percentile of (dB PSD - its median-filtered version): sharp resonances
  log_rms, kurtosis  of the band-passed signal (computed in extract_spectra.py)

One value per surface (median over its recordings). Then, per descriptor:
  epsilon2   Kruskal-Wallis effect size of the family, per dataset (0 = none, 1 = perfect)
  rank rho   Spearman correlation between the 4 family medians of Cluster and of LMT108
             (+1 = the families are in the same order in both datasets)
"""

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import ndimage, stats

from common import ACCEL_BAND, ACCEL_FREQ, AUDIO_BAND, AUDIO_FREQ, FAMILIES, OUT, load, save, third_octaves


def spectral_descriptors(psd, freq, band, split):
    in_band = (freq >= band[0]) & (freq <= band[1])
    f = freq[in_band]
    p = psd[:, in_band]
    level = 10 * np.log10(p)
    bands = third_octaves(psd, freq, band)
    band_centers = bands.columns.astype(float)

    return {
        "centroid": np.exp((p * np.log(f)).sum(axis=1) / p.sum(axis=1)),
        "slope": np.polyfit(np.log2(band_centers), bands.T, 1)[0],
        "hf_lf": 10 * np.log10(p[:, f >= split].sum(axis=1) / p[:, f < split].sum(axis=1)),
        "flatness": stats.gmean(p, axis=1) / p.mean(axis=1),
        "peakiness": np.percentile(level - ndimage.median_filter(level, (1, 31)), 99, axis=1),
    }


def epsilon_squared(values, groups):
    h = stats.kruskal(*[values[groups == g] for g in groups.unique()]).statistic
    return (h - groups.nunique() + 1) / (len(values) - groups.nunique())


meta, psd = load()

features = meta[["dataset", "family", "surface"]].copy()
for modality, freq, band, split in [("audio", AUDIO_FREQ, AUDIO_BAND, 2000), ("accel", ACCEL_FREQ, ACCEL_BAND, 200)]:
    for name, values in spectral_descriptors(psd[modality], freq, band, split).items():
        features[f"{modality} {name}"] = values
    features[f"{modality} log_rms"] = meta[f"{modality}_log_rms"]
    features[f"{modality} kurtosis"] = meta[f"{modality}_kurtosis"]
descriptor_names = list(features.columns[3:])

per_surface = features.groupby(["dataset", "family", "surface"]).median().reset_index()
cluster = per_surface[per_surface.dataset == "cluster"]
lmt = per_surface[per_surface.dataset == "lmt108"]

table = pd.DataFrame(index=descriptor_names)
for name in descriptor_names:
    table.loc[name, "epsilon2 cluster"] = epsilon_squared(cluster[name], cluster.family)
    table.loc[name, "epsilon2 lmt108"] = epsilon_squared(lmt[name], lmt.family)
    cluster_medians = cluster.groupby("family")[name].median()[FAMILIES]
    lmt_medians = lmt.groupby("family")[name].median()[FAMILIES]
    table.loc[name, "rank rho"] = stats.spearmanr(cluster_medians, lmt_medians).statistic

weakest_dataset = table[["epsilon2 cluster", "epsilon2 lmt108"]].min(axis=1)
table = table.loc[weakest_dataset.sort_values(ascending=False).index]
table.round(2).to_csv(OUT / "descriptors.csv")
print(table.round(2))

long = per_surface.melt(id_vars=["dataset", "family", "surface"], var_name="descriptor")
g = sns.catplot(
    long,
    x="family",
    y="value",
    hue="dataset",
    col="descriptor",
    col_wrap=5,
    col_order=table.index,
    kind="strip",
    dodge=True,
    order=FAMILIES,
    sharey=False,
    height=2.6,
    s=12,
)
for ax, name in zip(g.axes, table.index, strict=True):
    scores = table.loc[name]
    ax.set_title(
        f"{name}\n"
        f"eps2 C={scores['epsilon2 cluster']:.2f} L={scores['epsilon2 lmt108']:.2f} rho={scores['rank rho']:.1f}",
        fontsize=8,
    )
save(g.figure, "descriptors.png")

# Same descriptors after a z-score inside each dataset (removes the offset between datasets)
long["z-score within dataset"] = long.groupby(["dataset", "descriptor"]).value.transform(stats.zscore)
best_five = long[long.descriptor.isin(table.index[:5])]
g = sns.catplot(
    best_five,
    x="family",
    y="z-score within dataset",
    hue="dataset",
    col="descriptor",
    kind="box",
    order=FAMILIES,
    showfliers=False,
    height=3,
)
save(g.figure, "descriptors_zscored.png")

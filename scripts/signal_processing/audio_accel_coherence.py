"""Do audio and acceleration see the same vibration?

Coherence near 1: same content at that frequency; near 0: different content,
or the two streams are not synchronous. The histogram shows the lag that
best aligned them (a sharp peak = synchronous recording).
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import COHERENCE_FREQ, FAMILIES, PALETTE, load, save

meta, psd = load(clean=False)
coherence = pd.DataFrame(psd["coherence"], columns=COHERENCE_FREQ)
median_coherence = coherence.groupby([meta.dataset, meta.family]).median()
long = median_coherence.stack().rename_axis(["dataset", "family", "f (Hz)"]).reset_index(name="coherence")

between_50_and_1000_hz = long[(long["f (Hz)"] >= 50) & (long["f (Hz)"] <= 1000)]
print("median coherence, 50-1000 Hz:")
print(between_50_and_1000_hz.groupby(["dataset", "family"]).coherence.median().round(3))
print("\nlag of maximum cross-correlation (ms):")
print(meta.groupby("dataset").lag_ms.describe().round(1))

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (dataset, curves) in zip(axes[:2], long.groupby("dataset"), strict=True):
    sns.lineplot(curves, x="f (Hz)", y="coherence", hue="family", palette=PALETTE, hue_order=FAMILIES, ax=ax)
    ax.set(xscale="log", xlim=(20, 4000), ylim=(0, 1), title=f"{dataset}: audio vs acceleration")
sns.histplot(meta, x="lag_ms", hue="dataset", bins=80, stat="density", common_norm=False, ax=axes[2])
axes[2].set_title("lag of maximum cross-correlation (ms)")
save(fig, "coherence.png")

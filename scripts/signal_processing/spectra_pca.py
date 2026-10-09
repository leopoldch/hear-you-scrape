"""PCA (week 4) and Kernel PCA (week 5) on spectral shapes.

Vectors: 1/3-octave levels of audio + acceleration, mean level removed (common.shapes).
Subset: Cluster at direction 0 / 500 mN (720 recordings) + all LMT108 (720), so both
datasets weigh the same. Is the main axis the family, the dataset or the velocity?
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA, KernelPCA

from common import FAMILIES, PALETTE, load, save, shapes

meta, psd = load()
balanced = (meta.dataset == "lmt108") | ((meta.direction == 0) & (meta.force == 500))
vectors = shapes(psd)[balanced.to_numpy()]
points = meta[balanced].reset_index(drop=True)
FAMILY_STYLE = {"hue": "family", "style": "dataset", "palette": PALETTE, "hue_order": FAMILIES, "s": 12, "alpha": 0.7}

pca = PCA(n_components=10).fit(vectors)
projection = pca.transform(vectors)
points["PC1"] = projection[:, 0]
points["PC2"] = projection[:, 1]
print("explained variance:", pca.explained_variance_ratio_.round(2))

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
sns.scatterplot(points, x="PC1", y="PC2", ax=axes[0], **FAMILY_STYLE)
sns.scatterplot(points, x="PC1", y="PC2", hue="dataset", s=12, alpha=0.5, ax=axes[1])
cluster_points = points[points.dataset == "cluster"]
sns.scatterplot(cluster_points, x="PC1", y="PC2", hue="velocity", palette="viridis", s=12, ax=axes[2])
axes[2].set_title("Cluster only, colour = velocity")
save(fig, "pca_both_datasets.png")

explained = pd.Series(pca.explained_variance_ratio_, index=range(1, 11))
components = pd.DataFrame(pca.components_[:3].T, index=vectors.columns, columns=["PC1", "PC2", "PC3"])
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
explained.plot.bar(ax=axes[0], title="explained variance ratio")
components.plot(ax=axes[1], marker=".", title="components as spectral shapes (audio bands, then accel bands)")
axes[1].set_xticks(range(len(vectors.columns)), vectors.columns, fontsize=6, rotation=60)
save(fig, "pca_components.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, dataset in zip(axes, ["cluster", "lmt108"], strict=True):
    in_dataset = (points.dataset == dataset).to_numpy()
    projection = PCA(n_components=2).fit_transform(vectors[in_dataset])
    subset = points[in_dataset].copy()
    subset["PC1"] = projection[:, 0]
    subset["PC2"] = projection[:, 1]
    sns.scatterplot(subset, x="PC1", y="PC2", ax=ax, **FAMILY_STYLE)
    ax.set_title(f"PCA fitted on {dataset} only")
save(fig, "pca_per_dataset.png")

default_gamma = 1 / (vectors.shape[1] * vectors.to_numpy().var())
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, factor in zip(axes, [0.3, 1, 3], strict=True):
    projection = KernelPCA(n_components=2, kernel="rbf", gamma=factor * default_gamma).fit_transform(vectors)
    points["KPC1"] = projection[:, 0]
    points["KPC2"] = projection[:, 1]
    sns.scatterplot(points, x="KPC1", y="KPC2", ax=ax, legend=ax is axes[0], **FAMILY_STYLE)
    ax.set_title(f"Kernel PCA (RBF), gamma = {factor} x default")
save(fig, "kernel_pca.png")

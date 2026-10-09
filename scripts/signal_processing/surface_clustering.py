"""Clustering surfaces without labels (week 9): k-means, Gaussian mixture, Ward.

One point per surface: median spectral-shape vector over its recordings.
Adjusted Rand index (ARI): 1 = clusters are exactly the families, 0 = random.
On both datasets pooled, ARI vs dataset tells whether the clusters find the setup.
Last check: Cluster IDs follow the categories (recording order?). Inside one family,
are surfaces with close IDs more similar? Permutation test on Spearman's rho.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from common import FAMILIES, PALETTE, load, save, shapes


def cluster_with_three_methods(vectors, k):
    return {
        "k-means": KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(vectors),
        "GMM": GaussianMixture(n_components=k, covariance_type="diag", n_init=5, random_state=0).fit_predict(vectors),
        "Ward": AgglomerativeClustering(n_clusters=k).fit_predict(vectors),
    }


def spearman_with_id_gap(ids, spectral_distances):
    id_gaps = pdist(ids[:, None])
    return stats.spearmanr(id_gaps, spectral_distances).statistic


meta, psd = load()
surface_vectors = shapes(psd).groupby([meta.dataset, meta.family, meta.surface]).median()
surfaces = surface_vectors.index.to_frame(index=False)

subsets = {
    "cluster": (surfaces.dataset == "cluster").to_numpy(),
    "lmt108": (surfaces.dataset == "lmt108").to_numpy(),
    "both": np.ones(len(surfaces), dtype=bool),
}

scores = []
for subset, selected in subsets.items():
    for k in range(2, 11):
        for method, groups in cluster_with_three_methods(surface_vectors[selected], k).items():
            scores.append(
                {
                    "subset": subset,
                    "k": k,
                    "method": method,
                    "family": adjusted_rand_score(surfaces.family[selected], groups),
                    "dataset": adjusted_rand_score(surfaces.dataset[selected], groups),
                }
            )
            if k == 4 and method == "k-means":
                surfaces.loc[selected, f"k-means group ({subset})"] = groups.astype(str)
scores = pd.DataFrame(scores)
print(scores[scores.k == 4].round(2).to_string(index=False))

long = scores.melt(id_vars=["subset", "k", "method"], var_name="ARI against", value_name="ARI")
long = long[(long.subset == "both") | (long["ARI against"] == "family")]
g = sns.relplot(
    long, x="k", y="ARI", hue="method", style="ARI against", col="subset", kind="line", markers=True, height=3.5
)
save(g.figure, "clustering_ari_vs_k.png")

fig, axes = plt.subplots(1, 3, figsize=(17, 4))
for ax, (subset, selected) in zip(axes, subsets.items(), strict=True):
    in_subset = surfaces[selected]
    if subset == "both":
        true_label = in_subset.dataset + " " + in_subset.family
    else:
        true_label = in_subset.family
    counts = pd.crosstab(true_label, in_subset[f"k-means group ({subset})"])
    sns.heatmap(counts, annot=True, cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"k-means, k = 4, on {subset}: surfaces per group")
save(fig, "clustering_tables.png")

short_dataset_name = surfaces.dataset.map({"cluster": "C", "lmt108": "L"})
leaf_names = list(short_dataset_name + " " + surfaces.surface)
fig, ax = plt.subplots(figsize=(22, 6))
tree = dendrogram(linkage(surface_vectors, "ward"), labels=leaf_names, ax=ax, leaf_font_size=6, color_threshold=0)
for tick_label, surface_index in zip(ax.get_xticklabels(), tree["leaves"], strict=True):
    tick_label.set_color(PALETTE[surfaces.family[surface_index]])
ax.set_title("Ward dendrogram of all surfaces. Label colour = family, C = Cluster, L = LMT108")
save(fig, "clustering_dendrogram.png")

fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
for ax, family in zip(axes, FAMILIES, strict=True):
    selected = ((surfaces.dataset == "cluster") & (surfaces.family == family)).to_numpy()
    ids = surfaces.surface[selected].astype(int).to_numpy()
    spectral_distances = pdist(surface_vectors[selected])

    rho = spearman_with_id_gap(ids, spectral_distances)
    shuffled_rhos = []
    for seed in range(1000):
        shuffled_ids = np.random.default_rng(seed).permutation(ids)
        shuffled_rhos.append(spearman_with_id_gap(shuffled_ids, spectral_distances))
    p_value = np.mean(np.array(shuffled_rhos) >= rho)
    print(f"{family}: rho = {rho:.2f}, permutation p = {p_value:.3f}")

    ax.scatter(pdist(ids[:, None]), spectral_distances, s=5, color=PALETTE[family])
    ax.set(title=f"Cluster {family}: rho = {rho:.2f}, p = {p_value:.3f}", xlabel="|ID difference|")
axes[0].set_ylabel("spectral distance")
save(fig, "clustering_id_drift.png")

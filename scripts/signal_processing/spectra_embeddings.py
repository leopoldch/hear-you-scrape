"""Non-linear 2D maps of the spectral shapes (week 5), same vectors as spectra_pca.py.

MDS (keeps all distances), Isomap (distances along the manifold), Laplacian eigenmap
and LLE (keep neighbours), t-SNE (keeps neighbourhoods only).
First on both datasets, then inside each dataset.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding

from common import FAMILIES, PALETTE, load, save, shapes


def make_methods():
    return {
        "MDS": MDS(n_components=2, init="classical_mds", random_state=0),
        "Isomap": Isomap(n_components=2, n_neighbors=15),
        "Laplacian eigenmap": SpectralEmbedding(n_components=2, n_neighbors=15, random_state=0),
        "LLE": LocallyLinearEmbedding(n_components=2, n_neighbors=15, random_state=0),
        "t-SNE": TSNE(n_components=2, random_state=0),
    }


meta, psd = load()
balanced = (meta.dataset == "lmt108") | ((meta.direction == 0) & (meta.force == 500))
vectors = shapes(psd)[balanced.to_numpy()]
points = meta[balanced].reset_index(drop=True)

for subset in ["both", "cluster", "lmt108"]:
    if subset == "both":
        selected = points.index.notna()
    else:
        selected = (points.dataset == subset).to_numpy()
    subset_points = points[selected].copy()

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for ax, (name, method) in zip(axes, make_methods().items(), strict=True):
        print(subset, name)
        embedding = method.fit_transform(vectors[selected])
        subset_points["x"] = embedding[:, 0]
        subset_points["y"] = embedding[:, 1]
        sns.scatterplot(
            subset_points,
            x="x",
            y="y",
            hue="family",
            style="dataset",
            palette=PALETTE,
            hue_order=FAMILIES,
            s=8,
            alpha=0.7,
            legend=ax is axes[0],
            ax=ax,
        )
        ax.set(title=name, xticks=[], yticks=[], xlabel="", ylabel="")
    fig.suptitle(f"{subset}: colour = family, shape = dataset")
    save(fig, f"embeddings_{subset}.png")

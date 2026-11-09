"""H21: is the relation found with a few labeled surfaces the same from one setup to another?

If a short calibration works on every setup, the sound -> family relation exists on each one.
To bake it into the weights once and for all, it must be shared across setups. Three checks, on
frozen BEATs acceleration embeddings (recording means, cache of few_label_classifiers.py):

1. transfer matrix: probe trained on ALL surfaces of setup A, tested on setup B (diagonal:
   leave-surfaces-out within A, 5 folds by surface).
2. family directions: per setup (standardized within the setup), the mean of each family minus
   the setup mean; cosine similarity of these directions between setups.
3. unsupervised alignment (no label of the new setup): per-setup standardization, then CORAL
   (whiten with the new setup's own covariance, recolor with the training covariance), pooled
   probe from the two other setups.

    uv run python scripts/classification/calibration_relation.py
Output: analysis/classification/calibrated_new_setup/calibration_relation.csv and .png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrated_new_setup import DEV_SETUPS, FAMILIES, OUT, group_weights, score  # noqa: E402
from few_label_classifiers import window_embeddings  # noqa: E402


def probe():
    return make_pipeline(StandardScaler(), PCA(64, random_state=0), LogisticRegression(C=0.1, max_iter=3000))


def standardize(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def coral(source, target_cov, eps=1.0):
    """Whiten `source` with its own covariance, recolor with `target_cov` (features already centered)."""

    def power(matrix, p):
        values, vectors = np.linalg.eigh(matrix)
        return (vectors * np.clip(values, 1e-6, None) ** p) @ vectors.T

    own = np.cov(source, rowvar=False) + eps * np.eye(source.shape[1])
    return source @ power(own, -0.5) @ power(target_cov + eps * np.eye(source.shape[1]), 0.5)


def main():
    selected, windows, recording = window_embeddings()
    means = pd.DataFrame(windows).groupby(recording).mean().loc[np.arange(len(selected))].to_numpy()
    families = selected.family.to_numpy()
    rows = []

    # 1. transfer matrix
    matrix = pd.DataFrame(index=DEV_SETUPS, columns=DEV_SETUPS, dtype=float)
    for a in DEV_SETUPS:
        in_a = (selected.dataset == a).to_numpy()
        predicted = np.empty(in_a.sum(), dtype=object)
        x_a, y_a, groups = means[in_a], families[in_a], selected.surface[in_a].to_numpy()
        for train, test in GroupKFold(5).split(x_a, y_a, groups):
            predicted[test] = probe().fit(x_a[train], y_a[train]).predict(x_a[test])
        matrix.loc[a, a] = score(selected[in_a], predicted)["balanced accuracy"]
        model = probe().fit(x_a, y_a, logisticregression__sample_weight=group_weights(selected[in_a]))
        for b in DEV_SETUPS:
            if b != a:
                in_b = (selected.dataset == b).to_numpy()
                matrix.loc[a, b] = score(selected[in_b], model.predict(means[in_b]))["balanced accuracy"]
    print("transfer matrix (rows: trained on, columns: tested on)\n", matrix.round(2).to_string())
    for a in DEV_SETUPS:
        for b in DEV_SETUPS:
            rows.append({"check": "transfer matrix", "trained on": a, "tested on": b, "value": matrix.loc[a, b]})

    # 2. family directions
    directions = {}
    for setup in DEV_SETUPS:
        rows_s = (selected.dataset == setup).to_numpy()
        z = standardize(means[rows_s])
        directions[setup] = np.stack([z[families[rows_s] == f].mean(0) for f in FAMILIES])
    print("\ncosine of family directions between setups")
    for i, a in enumerate(DEV_SETUPS):
        for b in DEV_SETUPS[i + 1 :]:
            for k, f in enumerate(FAMILIES):
                u, v = directions[a][k], directions[b][k]
                cosine = u @ v / np.linalg.norm(u) / np.linalg.norm(v)
                rows.append(
                    {"check": "family direction cosine", "trained on": a, "tested on": b, "family": f, "value": cosine}
                )
            np.mean([rows[-4 + k]["value"] for k in range(4)])
            print(f"  {a} vs {b}: " + ", ".join(f"{f} {rows[-4 + k]['value']:.2f}" for k, f in enumerate(FAMILIES)))
    # reference: same setup, two random halves of its surfaces
    rng = np.random.default_rng(0)
    for setup in DEV_SETUPS:
        rows_s = np.flatnonzero((selected.dataset == setup).to_numpy())
        surfaces = selected.surface.iloc[rows_s].unique()
        half = rng.choice(surfaces, len(surfaces) // 2, replace=False)
        z = standardize(means[rows_s])
        first = selected.surface.iloc[rows_s].isin(half).to_numpy()
        values = []
        for f in FAMILIES:
            fam = families[rows_s] == f
            if (fam & first).any() and (fam & ~first).any():
                u, v = z[fam & first].mean(0) - z[first].mean(0), z[fam & ~first].mean(0) - z[~first].mean(0)
                values.append(u @ v / np.linalg.norm(u) / np.linalg.norm(v))
        rows.append(
            {
                "check": "family direction cosine, same setup halves",
                "trained on": setup,
                "tested on": setup,
                "value": np.mean(values),
            }
        )
        print(f"  {setup}, two halves of its own surfaces: {np.mean(values):.2f}")

    # 3. unsupervised alignment, k = 0
    print("\nno label of the new setup: per-setup standardization + CORAL, pooled probe")
    z = np.empty_like(means)
    for setup in DEV_SETUPS:
        rows_s = (selected.dataset == setup).to_numpy()
        z[rows_s] = standardize(means[rows_s])
    reduced = PCA(64, random_state=0).fit_transform(z)
    for new_setup in DEV_SETUPS:
        target = (selected.dataset == new_setup).to_numpy()
        train_cov = np.cov(reduced[~target], rowvar=False)
        aligned = coral(reduced[target], train_cov)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
        model.fit(
            reduced[~target], families[~target], logisticregression__sample_weight=group_weights(selected[~target])
        )
        for name, x in [("standardized", reduced[target]), ("standardized + CORAL", aligned)]:
            value = score(selected[target], model.predict(x))["balanced accuracy"]
            rows.append({"check": f"k = 0, {name}", "trained on": "other two", "tested on": new_setup, "value": value})
            print(f"  {new_setup}, {name}: {value:.2f}")

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "calibration_relation.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    image = axes[0].imshow(matrix.to_numpy(float), vmin=0, vmax=1, cmap="Blues")
    for i in range(3):
        for j in range(3):
            axes[0].text(
                j,
                i,
                f"{matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix.iloc[i, j] > 0.6 else "black",
            )
    axes[0].set(
        xticks=range(3),
        yticks=range(3),
        xticklabels=DEV_SETUPS,
        yticklabels=DEV_SETUPS,
        xlabel="tested on",
        ylabel="probe trained on",
        title="probe trained on one setup (all its labels)\ndiagonal: new surfaces of the same setup; chance 0.25",
    )
    fig.colorbar(image, ax=axes[0], fraction=0.046)
    cosines = results[results.check == "family direction cosine"]
    pairs = [f"{a} vs {b}" for a, b in cosines[["trained on", "tested on"]].drop_duplicates().itertuples(index=False)]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for k, (f, color) in enumerate(zip(FAMILIES, colors)):
        values = cosines[cosines.family == f].value.to_numpy()
        axes[1].bar(np.arange(len(pairs)) + (k - 1.5) * 0.2, values, width=0.19, color=color, label=f)
    halves = results[results.check == "family direction cosine, same setup halves"].value.mean()
    axes[1].axhline(halves, color="0.3", linestyle="--", linewidth=1.5, label="same setup, two halves of its surfaces")
    axes[1].axhline(0, color="0.5", linewidth=0.8)
    axes[1].set(
        xticks=range(len(pairs)),
        xticklabels=pairs,
        ylim=(-1, 1),
        ylabel="cosine",
        title="direction of each family (family mean - setup mean)\nsame direction across setups would be near 1",
    )
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("H21: is the sound -> family relation shared between setups? (frozen BEATs, acceleration)")
    fig.tight_layout()
    fig.savefig(OUT / "calibration_relation.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()

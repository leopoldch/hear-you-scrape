"""Surface-wise train / validation / test splits for the three experiments.

All recordings of a surface stay in the same split, so the test surfaces are never seen
in training. Surfaces that are probably the same material (same name in Cluster, versions
of one plate in LMT108) form one group and also stay together.

Splits ("split" column of folds.csv):
  cluster            experiments 1 and 2: new Cluster surfaces, 5 folds x 3 seeds,
                     stratified by family. Inside each fold, 1/4 of the training groups
                     become validation surfaces (for early stopping and lambda).
                     Experiment 2 uses the same folds, but keeps train and validation
                     recordings at 20-50 mm/s and test recordings at 60 mm/s.
  cluster_to_lmt108  experiment 3: train on Cluster (validation = Cluster surfaces),
                     test on every kept LMT108 surface. One fold, 3 seeds.
Cluster + LMT108 -> SENS3 is to be added once SENS3 is in surface_families.csv.

Run this file to write analysis/splits/folds.csv and print the counts. Models read the
splits through split_recordings().
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/splits"

FAMILIES = ["wood", "metal", "textile", "polymer"]
ROLES = ["train", "val", "test"]
N_FOLDS = 5
SEEDS = [0, 1, 2]
TRAIN_VELOCITIES = [20, 30, 40, 50]
TEST_VELOCITY = 60

# LMT108 surfaces that are probably the same material as another one
LMT_SAME_MATERIAL = {
    "G5SolidRubberPlateVersion1": "SolidRubberPlate",
    "G5SolidRubberPlateVersion2": "SolidRubberPlate",
    "G4Beech_0": "Beech",
    "G4CoreBeech": "Beech",
}


def kept_surfaces():
    """One row per kept surface, with the group that must stay in one split."""
    surfaces = pd.read_csv(ROOT / "analysis/surface_families.csv", dtype={"surface": str}, keep_default_na=False)
    surfaces = surfaces[surfaces.family != ""].reset_index(drop=True)

    # Cluster: two surfaces with the same name (White ash 8 and 13, Cotton 88 and 104) are one group
    is_lmt = surfaces.dataset == "lmt108"
    material = surfaces.name.where(~is_lmt, surfaces.surface.map(LMT_SAME_MATERIAL).fillna(surfaces.surface))
    surfaces["group"] = surfaces.dataset + "/" + material
    return surfaces


def train_val_split(surfaces, seed):
    """Boolean mask of the validation surfaces: about 1/4 of the groups, stratified by family."""
    inner = StratifiedGroupKFold(N_FOLDS - 1, shuffle=True, random_state=seed)
    _, val = next(inner.split(surfaces, surfaces.family, surfaces.group))
    is_val = np.zeros(len(surfaces), dtype=bool)
    is_val[val] = True
    return is_val


def cluster_folds(surfaces, seed):
    cluster = surfaces[surfaces.dataset == "cluster"].reset_index(drop=True)
    outer = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed)
    folds = []
    for fold, (train_val, _) in enumerate(outer.split(cluster, cluster.family, cluster.group)):
        roles = pd.Series("test", index=cluster.index)
        is_val = train_val_split(cluster.iloc[train_val], seed)
        roles.iloc[train_val] = np.where(is_val, "val", "train")
        folds.append(cluster.assign(split="cluster", seed=seed, fold=fold, role=roles))
    return folds


def cross_dataset_fold(surfaces, seed):
    cluster = surfaces[surfaces.dataset == "cluster"].reset_index(drop=True)
    lmt = surfaces[surfaces.dataset == "lmt108"]
    cluster_roles = np.where(train_val_split(cluster, seed), "val", "train")
    return [
        cluster.assign(split="cluster_to_lmt108", seed=seed, fold=0, role=cluster_roles),
        lmt.assign(split="cluster_to_lmt108", seed=seed, fold=0, role="test"),
    ]


def make_folds():
    """Long table: one row per (split, seed, fold, surface) with its role."""
    surfaces = kept_surfaces()
    parts = []
    for seed in SEEDS:
        parts += cluster_folds(surfaces, seed)
        parts += cross_dataset_fold(surfaces, seed)
    columns = ["split", "seed", "fold", "dataset", "surface", "name", "family", "group", "role"]
    return pd.concat(parts, ignore_index=True)[columns]


def check(folds):
    # A group never appears in two roles of the same fold
    roles_per_group = folds.groupby(["split", "seed", "fold", "group"]).role.nunique()
    assert (roles_per_group == 1).all()

    # Each Cluster surface is a test surface exactly once per seed
    cluster = folds[folds.split == "cluster"]
    test_count = cluster[cluster.role == "test"].groupby(["seed", "surface"]).size()
    assert (test_count == 1).all() and len(test_count) == len(SEEDS) * cluster.surface.nunique()

    # The test dataset is never used for training or validation
    cross = folds[folds.split == "cluster_to_lmt108"]
    assert (cross[cross.dataset == "lmt108"].role == "test").all()


def split_recordings(recordings, folds, split, seed, fold, new_velocity=False):
    """{"train", "val", "test"} -> the recordings of that role (rows of list_recordings() or recordings.csv).

    new_velocity=True is experiment 2: train and val keep 20-50 mm/s, test keeps 60 mm/s.
    """
    chosen = folds[(folds.split == split) & (folds.seed == seed) & (folds.fold == fold)]
    recordings = recordings.astype({"surface": str})
    merged = recordings.merge(chosen[["dataset", "surface", "role"]], on=["dataset", "surface"])
    if new_velocity:
        allowed = np.where(
            merged.role == "test", merged.velocity == TEST_VELOCITY, merged.velocity.isin(TRAIN_VELOCITIES)
        )
        merged = merged[allowed]
    return {role: merged[merged.role == role].drop(columns="role") for role in ROLES}


def print_summary(folds):
    cluster = folds[folds.split == "cluster"]
    print(
        f"Cluster: {cluster.surface.nunique()} surfaces in {cluster.group.nunique()} groups, "
        f"{N_FOLDS} folds x {len(SEEDS)} seeds"
    )
    counts = cluster.groupby(["seed", "fold", "role", "family"]).size().unstack("family", fill_value=0)
    per_role = counts.groupby("role").agg(["mean", "min", "max"]).reindex(ROLES)
    print("surfaces per fold (mean, min, max over the 15 folds):")
    print(per_role.reindex(columns=FAMILIES, level=0).round(1).to_string(), "\n")

    recordings_file = ROOT / "analysis/signal_processing/recordings.csv"
    if recordings_file.exists():
        recordings = pd.read_csv(recordings_file, dtype={"surface": str})
        for experiment, new_velocity in [("experiment 1", False), ("experiment 2", True)]:
            sizes = split_recordings(recordings, folds, "cluster", seed=0, fold=0, new_velocity=new_velocity)
            print(f"{experiment}, seed 0 fold 0 recordings:", {role: len(rows) for role, rows in sizes.items()})
        print()

    cross = folds[(folds.split == "cluster_to_lmt108") & (folds.seed == 0)]
    print("Cluster -> LMT108, seed 0 surfaces:")
    print(
        cross.groupby(["dataset", "role", "family"])
        .size()
        .unstack("family", fill_value=0)
        .reindex(columns=FAMILIES)
        .to_string()
    )


def plot_folds(folds):
    """Role of every Cluster surface in each fold of seed 0, surfaces sorted by family."""
    cluster = folds[(folds.split == "cluster") & (folds.seed == 0)].copy()
    cluster["family"] = pd.Categorical(cluster.family, FAMILIES)
    table = cluster.pivot(index=["family", "group", "surface"], columns="fold", values="role").sort_index()
    codes = table.replace({role: code for code, role in enumerate(ROLES)}).astype(int).T

    fig, ax = plt.subplots(figsize=(15, 2.6))
    ax.imshow(codes, aspect="auto", cmap=ListedColormap(["#c9d6e8", "#eda100", "#d64545"]), interpolation="none")
    families = table.index.get_level_values("family")
    boundaries = np.flatnonzero(families[1:] != families[:-1]) + 0.5
    for boundary in boundaries:
        ax.axvline(boundary, color="black", linewidth=1.5)
    centers = (np.append(-0.5, boundaries) + np.append(boundaries, len(table) - 0.5)) / 2
    ax.set_xticks(centers, FAMILIES)
    ax.set_yticks(range(N_FOLDS), [f"fold {fold}" for fold in range(N_FOLDS)])
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in ["#c9d6e8", "#eda100", "#d64545"]]
    ax.legend(handles, ROLES, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_title("Cluster surfaces (columns) per fold, seed 0")
    fig.savefig(OUT / "cluster_folds.png", dpi=130, bbox_inches="tight")
    print("saved", OUT / "cluster_folds.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    folds = make_folds()
    check(folds)
    folds.to_csv(OUT / "folds.csv", index=False)
    print_summary(folds)
    plot_folds(folds)
    print("saved", OUT / "folds.csv")

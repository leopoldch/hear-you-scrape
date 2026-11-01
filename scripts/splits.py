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
ROLE_COLORS = ["#c9d6e8", "#eda100", "#d64545"]
N_FOLDS = 5
TRAIN_VELOCITIES = [20, 30, 40, 50]
TEST_VELOCITY = 60


WITHIN_DATASET = {
    "cluster": (["cluster"], 3),
    "lmt108": (["lmt108"], 5),
    "combined": (["cluster", "lmt108"], 3),
}

CROSS_DATASET = {
    "cluster_to_lmt108": ("cluster", "lmt108"),
    "lmt108_to_cluster": ("lmt108", "cluster"),
}
CROSS_DATASET_SEEDS = 3


LMT_SAME_MATERIAL = {
    "G5SolidRubberPlateVersion1": "SolidRubberPlate",
    "G5SolidRubberPlateVersion2": "SolidRubberPlate",
    "G4Beech_0": "Beech",
    "G4CoreBeech": "Beech",
}


def kept_surfaces():
    surfaces = pd.read_csv(ROOT / "analysis/surface_families.csv", dtype={"surface": str}, keep_default_na=False)
    surfaces = surfaces[surfaces.family != ""].reset_index(drop=True)

    is_lmt = surfaces.dataset == "lmt108"
    material = surfaces.name.where(~is_lmt, surfaces.surface.map(LMT_SAME_MATERIAL).fillna(surfaces.surface))
    surfaces["group"] = surfaces.dataset + "/" + material
    surfaces["stratum"] = surfaces.dataset + "/" + surfaces.family
    return surfaces


def train_val_split(surfaces, seed):
    inner = StratifiedGroupKFold(N_FOLDS - 1, shuffle=True, random_state=seed)
    _, val = next(inner.split(surfaces, surfaces.stratum, surfaces.group))
    is_val = np.zeros(len(surfaces), dtype=bool)
    is_val[val] = True
    return is_val


def within_dataset_folds(surfaces, split, datasets, seed):
    surfaces = surfaces[surfaces.dataset.isin(datasets)].reset_index(drop=True)
    outer = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed)
    folds = []
    for fold, (train_val, _) in enumerate(outer.split(surfaces, surfaces.stratum, surfaces.group)):
        roles = pd.Series("test", index=surfaces.index)
        is_val = train_val_split(surfaces.iloc[train_val], seed)
        roles.iloc[train_val] = np.where(is_val, "val", "train")
        folds.append(surfaces.assign(split=split, seed=seed, fold=fold, role=roles))
    return folds


def cross_dataset_fold(surfaces, split, train_dataset, test_dataset, seed):
    train = surfaces[surfaces.dataset == train_dataset].reset_index(drop=True)
    test = surfaces[surfaces.dataset == test_dataset]
    train_roles = np.where(train_val_split(train, seed), "val", "train")
    return [
        train.assign(split=split, seed=seed, fold=0, role=train_roles),
        test.assign(split=split, seed=seed, fold=0, role="test"),
    ]


def make_folds():
    surfaces = kept_surfaces()
    parts = []
    for split, (datasets, n_seeds) in WITHIN_DATASET.items():
        for seed in range(n_seeds):
            parts += within_dataset_folds(surfaces, split, datasets, seed)
    for split, (train_dataset, test_dataset) in CROSS_DATASET.items():
        for seed in range(CROSS_DATASET_SEEDS):
            parts += cross_dataset_fold(surfaces, split, train_dataset, test_dataset, seed)
    columns = ["split", "seed", "fold", "dataset", "surface", "name", "family", "group", "role"]
    return pd.concat(parts, ignore_index=True)[columns]


def check(folds):

    roles_per_group = folds.groupby(["split", "seed", "fold", "group"]).role.nunique()
    assert (roles_per_group == 1).all()

    for split, (datasets, n_seeds) in WITHIN_DATASET.items():
        rows = folds[folds.split == split]
        test_count = rows[rows.role == "test"].groupby(["seed", "dataset", "surface"]).size()
        n_surfaces = len(rows.drop_duplicates(["dataset", "surface"]))
        assert (test_count == 1).all() and len(test_count) == n_seeds * n_surfaces, split
        assert set(rows.dataset) == set(datasets), split

    for split, (train_dataset, test_dataset) in CROSS_DATASET.items():
        rows = folds[folds.split == split]
        assert (rows[rows.dataset == test_dataset].role == "test").all(), split
        assert (rows[rows.dataset == train_dataset].role != "test").all(), split


def split_recordings(recordings, folds, split, seed, fold, new_velocity=False):
    if new_velocity and split != "cluster":
        raise ValueError("new_velocity needs the cluster split: LMT108 has no velocity")
    chosen = folds[(folds.split == split) & (folds.seed == seed) & (folds.fold == fold)]
    recordings = recordings.astype({"surface": str})
    merged = recordings.merge(chosen[["dataset", "surface", "role"]], on=["dataset", "surface"])
    if new_velocity:
        allowed = np.where(
            merged.role == "test", merged.velocity == TEST_VELOCITY, merged.velocity.isin(TRAIN_VELOCITIES)
        )
        merged = merged[allowed]
    return {role: merged[merged.role == role].drop(columns="role") for role in ROLES}


def surface_counts(rows):
    return rows.groupby(["dataset", "role", "family"]).size().unstack("family", fill_value=0)


def print_summary(folds):
    recordings_file = ROOT / "analysis/signal_processing/recordings.csv"
    recordings = None
    if recordings_file.exists():
        recordings = pd.read_csv(recordings_file, dtype={"surface": str})

    for split, (_, n_seeds) in WITHIN_DATASET.items():
        rows = folds[folds.split == split]
        n_folds = n_seeds * N_FOLDS
        print(f"=== {split}: {rows.group.nunique()} groups, {N_FOLDS} folds x {n_seeds} seeds")
        counts = rows.groupby(["seed", "fold", "dataset", "role", "family"]).size().unstack("family", fill_value=0)
        per_role = counts.groupby(["dataset", "role"]).agg(["min", "max"])
        per_role = per_role.reindex(columns=FAMILIES, level=0).reindex(ROLES, level="role")
        print(f"surfaces per fold (min, max over the {n_folds} folds):")
        print(per_role.to_string())
        if recordings is not None:
            sizes = split_recordings(recordings, folds, split, seed=0, fold=0)
            print("recordings, seed 0 fold 0:", {role: len(part) for role, part in sizes.items()})
            if split == "cluster":
                sizes = split_recordings(recordings, folds, split, seed=0, fold=0, new_velocity=True)
                print("  with new_velocity:", {role: len(part) for role, part in sizes.items()})
        print()

    for split in CROSS_DATASET:
        rows = folds[(folds.split == split) & (folds.seed == 0)]
        print(f"=== {split}, seed 0 surfaces:")
        print(surface_counts(rows).reindex(columns=FAMILIES).to_string())
        if recordings is not None:
            sizes = split_recordings(recordings, folds, split, seed=0, fold=0)
            print("recordings:", {role: len(part) for role, part in sizes.items()})
        print()


def plot_folds(folds, split):
    rows = folds[(folds.split == split) & (folds.seed == 0)].copy()
    rows["family"] = pd.Categorical(rows.family, FAMILIES)
    table = rows.pivot(index=["dataset", "family", "group", "surface"], columns="fold", values="role").sort_index()
    codes = table.replace({role: code for code, role in enumerate(ROLES)}).astype(int).T

    fig, ax = plt.subplots(figsize=(15, 2.6))
    ax.imshow(codes, aspect="auto", cmap=ListedColormap(ROLE_COLORS), interpolation="none")

    blocks = table.index.droplevel(["group", "surface"])
    boundaries = np.flatnonzero(blocks[1:] != blocks[:-1]) + 0.5
    for boundary in boundaries:
        ax.axvline(boundary, color="black", linewidth=1.5)
    centers = (np.append(-0.5, boundaries) + np.append(boundaries, len(table) - 0.5)) / 2
    several_datasets = rows.dataset.nunique() > 1
    labels = [f"{dataset}\n{family}" if several_datasets else family for dataset, family in blocks.unique()]
    ax.set_xticks(centers, labels)
    ax.set_yticks(range(N_FOLDS), [f"fold {fold}" for fold in range(N_FOLDS)])

    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in ROLE_COLORS]
    ax.legend(handles, ROLES, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_title(f"Split {split}: role of each surface (columns) per fold, seed 0")
    fig.savefig(OUT / f"{split}_folds.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / f"{split}_folds.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    folds = make_folds()
    check(folds)
    folds.to_csv(OUT / "folds.csv", index=False)
    print_summary(folds)
    for split in WITHIN_DATASET:
        plot_folds(folds, split)
    print("saved", OUT / "folds.csv")

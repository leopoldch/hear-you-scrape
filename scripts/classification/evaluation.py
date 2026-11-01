import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "signal_processing"))
from common import (  # noqa: E402
    ACCEL_FREQ,
    ACCEL_RATE,
    AUDIO_FREQ,
    AUDIO_NPERSEG,
    AUDIO_RATE,
    FAMILIES,
    ROOT,
    list_recordings,
    list_sens3_recordings,
    load,
    read_recording,
    shapes,
    surface_folds,
)

__all__ = [
    "ACCEL_FREQ",
    "ACCEL_RATE",
    "AUDIO_FREQ",
    "AUDIO_NPERSEG",
    "AUDIO_RATE",
    "FAMILIES",
    "list_recordings",
    "list_sens3_recordings",
    "read_recording",
]

OUT = ROOT / "analysis/classification"
OUT.mkdir(parents=True, exist_ok=True)
N_FOLDS = 5
CHANCE = 1 / len(FAMILIES)
COLUMN_PREFIX_OF_INPUT = {"audio": "audio ", "acceleration": "accel ", "audio + acceleration": ""}
TEAM_SPLITS = ["cluster", "lmt108", "combined", "cluster_to_lmt108", "lmt108_to_cluster"]


def team_folds():
    return pd.read_csv(ROOT / "analysis/splits/folds.csv", dtype={"surface": str})


def team_seeds(split):
    folds = team_folds()
    return sorted(folds[folds.split == split].seed.unique())


def team_train_and_test(recordings, split, seed):
    folds = team_folds()
    folds = folds[(folds.split == split) & (folds.seed == seed)]
    for _, fold in folds.groupby("fold"):
        role = recordings[["dataset", "surface"]].merge(fold[["dataset", "surface", "role"]], how="left").role
        yield role.isin(["train", "val"]).to_numpy(), (role == "test").to_numpy()


def scores_per_dataset(recordings, predicted):
    tested = pd.notna(predicted)
    rows = []
    for dataset in recordings[tested].dataset.unique():
        on_dataset = tested & (recordings.dataset == dataset).to_numpy()
        rows.append({"tested on": dataset, **score(recordings[on_dataset], np.asarray(predicted)[on_dataset])})
    return rows


def recording_table():
    return pd.read_csv(ROOT / "analysis/signal_processing/recordings.csv", dtype={"surface": str})


def spectral_shape_features(*datasets):
    recordings, psd = load(clean=False)
    tables, features = [recordings], [shapes(psd)]
    for dataset in datasets:
        table, shape = shape_features_of(dataset)
        tables.append(table)
        features.append(shape)
    return pd.concat(tables, ignore_index=True), pd.concat(features, ignore_index=True)


def shape_features_of(dataset):
    recordings = pd.read_csv(OUT / f"{dataset}_recordings.csv", dtype={"surface": str})
    return recordings, shapes(dict(np.load(OUT / f"{dataset}_psd_cache.npz")))


def input_features(features, input_name):
    return features.loc[:, features.columns.str.startswith(COLUMN_PREFIX_OF_INPUT[input_name])]


def zscore(features):
    return (features - features.mean(axis=0)) / features.std(axis=0)


def folds(recordings):
    fold = pd.Series(0, index=recordings.index)
    for dataset in recordings.dataset.unique():
        in_dataset = recordings.dataset == dataset
        fold[in_dataset] = surface_folds(recordings[in_dataset])
    return fold.to_numpy()


def group_weights(recordings):
    weights = 1 / recordings.groupby(["dataset", "family"]).family.transform("size").to_numpy()
    return weights / weights.mean()


def fit_and_predict(model, train_features, train_families, weights, test_features):
    scaler = StandardScaler().fit(train_features)
    model.fit(scaler.transform(train_features), train_families, sample_weight=weights)
    return model.predict(scaler.transform(test_features))


def mean_recall(confusion):
    return np.mean(np.diag(confusion) / confusion.sum(axis=1))


def score(recordings, predicted):
    predicted = np.asarray(predicted)
    confusions = {}
    for surface in recordings.surface.unique():
        on_surface = (recordings.surface == surface).to_numpy()
        confusions[surface] = confusion_matrix(recordings.family[on_surface], predicted[on_surface], labels=FAMILIES)

    family_of_surface = recordings.groupby("surface").family.first()
    rng = np.random.default_rng(42)
    draws = []
    for _ in range(2000):
        resampled = np.zeros((len(FAMILIES), len(FAMILIES)))
        for family in FAMILIES:
            surfaces = family_of_surface.index[family_of_surface == family]
            for surface in rng.choice(surfaces, size=len(surfaces)):
                resampled += confusions[surface]
        draws.append(mean_recall(resampled))

    low, high = np.percentile(draws, [2.5, 97.5])
    return {"balanced accuracy": mean_recall(sum(confusions.values())), "low": low, "high": high}


def plot_scores(scores, groups, bars, panels, title, name):
    panel_names = scores[panels].unique()
    bar_names = scores[bars].unique()
    width = 0.8 / len(bar_names)

    fig, axes = plt.subplots(1, len(panel_names), figsize=(5 * len(panel_names), 4), sharey=True, squeeze=False)
    for ax, panel in zip(axes[0], panel_names, strict=True):
        in_panel = scores[scores[panels] == panel]
        group_names = in_panel[groups].unique()
        for k, bar in enumerate(bar_names):
            rows = in_panel[in_panel[bars] == bar].set_index(groups).reindex(group_names)
            x = np.arange(len(group_names)) + (k - (len(bar_names) - 1) / 2) * width
            below = rows["balanced accuracy"] - rows.low
            above = rows.high - rows["balanced accuracy"]
            ax.bar(x, rows["balanced accuracy"], width, color=f"C{k}", label=bar)
            ax.errorbar(x, rows["balanced accuracy"], yerr=[below, above], fmt="none", ecolor="black", capsize=2)
        ax.axhline(CHANCE, color="black", linestyle=":", linewidth=1)
        ax.set_xticks(range(len(group_names)), group_names, rotation=20)
        ax.set(title=panel, ylim=(0, 1))

    axes[0, 0].set_ylabel("balanced accuracy, 95% interval\n(dotted line: chance)")
    axes[0, -1].legend(title=bars, bbox_to_anchor=(1, 1), loc="upper left")
    fig.suptitle(title)
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight")
    print("saved", OUT / name)

"""H17: which classifier makes the most of k labeled surfaces of a new setup? Development only.

Same setting as the frozen H16 choice: frozen BEATs on the acceleration, probe trained on the k
labeled surfaces of the new setup only. Same draws as H16 `select` (seed [setup index, k, draw]),
k = 1, 2, 4, 10 draws, each of Cluster, LMT108, SENS3 as the new setup.

Classifiers:
    H16 choice                             recording means, PCA + logistic regression (reference)
    windows, logistic regression           train on every 0.5 s window, average probabilities per recording
    nearest family mean                    cosine distance to the mean of each family (standardized)
    shrinkage LDA                          linear discriminant with Ledoit-Wolf shrinkage
    windows + self-training                windows LR, then one round adding the unlabeled recordings of the
                                           new setup with their predicted family (most confident half)

    uv run python scripts/classification/few_label_classifiers.py
Output: analysis/classification/calibrated_new_setup/few_label_classifiers.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrated_new_setup import (
    DATA,
    DEV_SETUPS,
    FAMILIES,
    OUT,
    accel_windows_16k,
    beats_embedder,  # noqa: E402
    calibrate_and_predict,
    embed_windows,
    labeled_surfaces,
    score,
)

K_VALUES = [1, 2, 4]
N_DRAWS = 10


def window_embeddings():
    """BEATs embedding of every acceleration window of the selected recordings, cached."""
    selected = pd.read_csv(DATA / "qwen_audio/selected_recordings.csv", dtype={"surface": str})
    selected["surface"] = selected.dataset + ":" + selected.surface
    path = OUT / "window_embeddings_BEATs.npz"
    if not path.exists():
        features, owner = [], []
        for waves_file, group in selected.groupby("waves_file"):
            cache = np.load(DATA / waves_file)
            row_of_recording = pd.Series(group.index.to_numpy(), index=group.index_in_waves.to_numpy())
            keep = np.isin(cache["recording"], group.index_in_waves.to_numpy())
            features.append(embed_windows(beats_embedder, accel_windows_16k(cache["accel"][keep]), 128))
            owner.append(row_of_recording.loc[cache["recording"][keep]].to_numpy())
            print(waves_file, "windows embedded", flush=True)
        np.savez(path, features=np.concatenate(features), recording=np.concatenate(owner))
    cache = np.load(path)
    return selected, cache["features"], cache["recording"]


def logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))


def per_recording(probabilities, recording, wanted):
    table = pd.DataFrame(probabilities).groupby(recording).mean()
    return table.loc[wanted].to_numpy()


def predict(name, windows, recording, means, train, test):
    """Family of each test recording; train/test are recording indices."""
    families = np.array(FAMILIES)
    if name == "H16 choice (recording means, PCA + logistic regression)":
        labeled = np.zeros(len(LABELS), bool)
        labeled[train] = True
        table = SELECTED[SELECTED.index.isin(np.concatenate([train, test]))]
        test_mask, predicted = calibrate_and_predict(
            means[table.index],
            table.reset_index(drop=True),
            table.dataset.iloc[0],
            labeled[table.index],
            "new setup only",
        )
        return pd.Series(predicted, index=table.index[test_mask]).loc[test].to_numpy()
    if name == "nearest family mean":
        scaler = StandardScaler().fit(means[train])
        x, y = scaler.transform(means[train]), LABELS[train]
        centers = np.stack([x[y == f].mean(0) for f in FAMILIES])
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        z = scaler.transform(means[test])
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        return families[np.argmax(z @ centers.T, axis=1)]
    if name == "shrinkage LDA":
        model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
        return model.fit(means[train], LABELS[train]).predict(means[test])
    in_train = np.isin(recording, train)
    model = logistic().fit(windows[in_train], LABELS[recording[in_train]])
    probabilities = per_recording(
        model.predict_proba(windows[np.isin(recording, test)]), recording[np.isin(recording, test)], test
    )
    if name == "windows, logistic regression":
        return model.classes_[probabilities.argmax(1)]
    confident = test[probabilities.max(1) >= np.median(probabilities.max(1))]
    pseudo = pd.Series(model.classes_[probabilities.argmax(1)], index=test)
    rows = np.isin(recording, np.concatenate([train, confident]))
    labels = np.where(
        np.isin(recording[rows], train), LABELS[recording[rows]], pseudo.reindex(recording[rows]).to_numpy()
    )
    model = logistic().fit(windows[rows], labels)
    probabilities = per_recording(
        model.predict_proba(windows[np.isin(recording, test)]), recording[np.isin(recording, test)], test
    )
    return model.classes_[probabilities.argmax(1)]


CLASSIFIERS = [
    "H16 choice (recording means, PCA + logistic regression)",
    "windows, logistic regression",
    "nearest family mean",
    "shrinkage LDA",
    "windows + self-training",
]


def main():
    global LABELS, SELECTED
    selected, windows, recording = window_embeddings()
    SELECTED = selected
    LABELS = selected.family.to_numpy()
    means = pd.DataFrame(windows).groupby(recording).mean().loc[np.arange(len(selected))].to_numpy()
    rows = []
    for new_setup in DEV_SETUPS:
        target = selected[selected.dataset == new_setup]
        for k in K_VALUES:
            for draw in range(N_DRAWS):
                rng = np.random.default_rng([DEV_SETUPS.index(new_setup), k, draw])
                labeled = target.surface.isin(labeled_surfaces(target, k, rng)).to_numpy()
                train, test = target.index[labeled].to_numpy(), target.index[~labeled].to_numpy()
                for name in CLASSIFIERS:
                    predicted = predict(name, windows, recording, means, train, test)
                    rows.append(
                        {
                            "new setup": new_setup,
                            "k": k,
                            "draw": draw,
                            "classifier": name,
                            **score(selected.loc[test], predicted, n_draws=200),
                        }
                    )
            print(new_setup, k, "done", flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "few_label_classifiers.csv", index=False)
    table = scores.groupby(["classifier", "k", "new setup"])["balanced accuracy"].mean().unstack("new setup")
    table["mean"] = table.mean(axis=1)
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()

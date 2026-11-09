"""Can frozen pretrained representations recognize the family on a setup never seen in training?

Runs on the hub, after qwen_audio_features.py, on the recordings it selected. Representations:
Qwen2-Audio (audio encoder and language-model layers), BEATs and CLAP (cached by
pretrained_encoders.ipynb, averaged per recording). Inputs: audio, acceleration, both.

Tests, all scored with balanced accuracy over recordings and a 95% interval from resampling
whole surfaces (chance 0.25):

1. "new surfaces": 5 surface folds inside Cluster (does the information exist at all?).
2. "new setup": train on two setups, test on the third (Cluster, LMT108, SENS3 in turn).
   Variant: each setup standardized with its own statistics (uses unlabeled target data).
3. "new setup + k labeled surfaces": add k surfaces per family of the new setup to training,
   10 random draws, for a few representations fixed in advance.
4. Qwen answering directly: argmax of its four family logits, raw; centered on the new setup
   (no labels); recalibrated by a logistic regression on the logits of the two training setups.

Classifier: train-only standardization, PCA to 256 dimensions, logistic regression, training
weights balancing (setup, family) groups. Nothing is tuned on the test setup.
Output: analysis/classification/qwen_audio/setup_transfer_scores.csv
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

QUICK = "--quick" in sys.argv
FAMILIES = ["wood", "metal", "textile", "polymer"]
SETUPS = ["cluster", "lmt108", "sens3"]
OUT = Path("analysis/classification/qwen_audio")
SUFFIX = "_quick" if QUICK else ""
FEW_SHOT_REPRESENTATIONS = ["Qwen encoder layer 32", "Qwen LLM last token layer 16", "BEATs", "CLAP"]
FEW_SHOT_K = [1, 2, 4]
FEW_SHOT_DRAWS = 3 if QUICK else 10


def per_recording_means(window_embeddings, recording_of_window, wanted):
    sums = np.zeros((recording_of_window.max() + 1, window_embeddings.shape[1]))
    np.add.at(sums, recording_of_window, window_embeddings.astype(np.float64))
    counts = np.bincount(recording_of_window, minlength=len(sums))[:, None]
    return (sums / np.maximum(counts, 1))[wanted]


def load_beats_clap(selected):
    """Return {"BEATs" or "CLAP": {signal: array (recordings x dims)}}, averaged per recording."""
    in_main = (selected.waves_file == "waveforms.npz").to_numpy()
    recording_of_window = {
        "": np.load("analysis/classification/waveforms.npz")["recording"],
        "sens3_": np.load("analysis/classification/sens3_waveforms.npz")["recording"],
    }
    representations = {}
    for name, file in [("BEATs", "embeddings_beats.npz"), ("CLAP", "embeddings_clap.npz")]:
        cache = np.load(f"analysis/classification/pretrained/{file}")
        representations[name] = {}
        for signal in ["audio", "accel"]:
            values = np.zeros((len(selected), cache[signal].shape[1]), np.float32)
            for prefix, rows in [("", in_main), ("sens3_", ~in_main)]:
                wanted = selected.index_in_waves.to_numpy()[rows]
                values[rows] = per_recording_means(cache[prefix + signal], recording_of_window[prefix], wanted)
            representations[name][signal] = values
    return representations


def load_representations(selected):
    """Return {representation name: {signal: array (recordings x dims)}} and the Qwen logits."""
    representations = {}
    qwen = {signal: np.load(OUT / f"features_{signal}{SUFFIX}.npz") for signal in ["audio", "accel"]}
    for key in qwen["audio"].files:
        if key == "family_logits":
            continue
        kind, layer = key.rsplit("_layer", 1)
        label = {"encoder": "Qwen encoder", "llm_last": "Qwen LLM last token", "llm_audio": "Qwen LLM audio tokens"}[
            kind
        ]
        representations[f"{label} layer {layer}"] = {s: qwen[s][key].astype(np.float32) for s in qwen}

    representations.update(load_beats_clap(selected))
    return representations, qwen["audio"]["family_logits"], qwen["accel"]["family_logits"]


def features_for(representation, input_name):
    if input_name == "audio + acceleration":
        return np.concatenate([representation["audio"], representation["accel"]], axis=1)
    return representation["audio" if input_name == "audio" else "accel"]


def group_weights(table):
    weights = 1 / table.groupby(["dataset", "family"]).family.transform("size").to_numpy()
    return weights / weights.mean()


def classifier(n_train, n_dims):
    return make_pipeline(
        StandardScaler(), PCA(min(256, n_train - 1, n_dims), random_state=0), LogisticRegression(max_iter=3000)
    )


def fit_and_predict(features, selected, train, test):
    model = classifier(train.sum(), features.shape[1])
    model.fit(features[train], selected.family[train], logisticregression__sample_weight=group_weights(selected[train]))
    return model.predict(features[test])


def mean_recall(confusion):
    return np.mean(np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1))


def score(table, predicted, n_draws=1000):
    table = table.reset_index(drop=True)
    predicted = np.asarray(predicted)
    confusions = {
        surface: confusion_matrix(table.family[rows], predicted[rows], labels=FAMILIES)
        for surface, rows in table.groupby("surface").groups.items()
    }
    family_of_surface = table.groupby("surface").family.first()
    rng = np.random.default_rng(42)
    draws = []
    for _ in range(n_draws):
        resampled = np.zeros((4, 4))
        for family in FAMILIES:
            surfaces = family_of_surface.index[family_of_surface == family]
            for surface in rng.choice(surfaces, size=len(surfaces)):
                resampled += confusions[surface]
        draws.append(mean_recall(resampled))
    answers = pd.Series(predicted).value_counts(normalize=True)
    return {
        "balanced accuracy": mean_recall(sum(confusions.values())),
        "low": np.percentile(draws, 2.5),
        "high": np.percentile(draws, 97.5),
        "most frequent answer share": answers.iloc[0],
    }


def standardize_per_setup(features, selected):
    out = np.empty_like(features)
    for dataset in SETUPS:
        rows = (selected.dataset == dataset).to_numpy()
        out[rows] = (features[rows] - features[rows].mean(0)) / (features[rows].std(0) + 1e-6)
    return out


def labeled_surfaces(table, k, rng):
    surfaces = table[["surface", "family"]].drop_duplicates()
    return np.concatenate([rng.choice(surfaces.surface[surfaces.family == f], k, replace=False) for f in FAMILIES])


def main():
    selected = pd.read_csv(OUT / "selected_recordings.csv", dtype={"surface": str})
    if QUICK:
        selected = selected.groupby("dataset").head(16).reset_index(drop=True)
    selected["surface"] = selected.dataset + ":" + selected.surface
    representations, audio_logits, accel_logits = load_representations(selected)
    in_setup = {dataset: (selected.dataset == dataset).to_numpy() for dataset in SETUPS}
    rows = []

    def record(test, representation, input_name, variant, table, predicted, **extra):
        rows.append(
            {
                "test": test,
                "representation": representation,
                "input": input_name,
                "variant": variant,
                **extra,
                **score(table, predicted, n_draws=100 if QUICK else 1000),
            }
        )

    # Qwen answering directly
    for input_name, logits in [("audio", audio_logits), ("acceleration", accel_logits)]:
        for new_setup in SETUPS:
            test, train = in_setup[new_setup], ~in_setup[new_setup]
            table = selected[test]
            record(
                f"new setup: {new_setup}",
                "Qwen answer",
                input_name,
                "raw",
                table,
                np.array(FAMILIES)[logits[test].argmax(1)],
            )
            centered = logits[test] - logits[test].mean(0)
            record(
                f"new setup: {new_setup}",
                "Qwen answer",
                input_name,
                "centered on new setup",
                table,
                np.array(FAMILIES)[centered.argmax(1)],
            )
            recalibration = LogisticRegression(max_iter=2000)
            recalibration.fit(logits[train], selected.family[train], sample_weight=group_weights(selected[train]))
            record(
                f"new setup: {new_setup}",
                "Qwen answer",
                input_name,
                "recalibrated on training setups",
                table,
                recalibration.predict(logits[test]),
            )
    print("Qwen answers done", flush=True)

    folds = np.zeros(len(selected), int)
    cluster_surfaces = selected[in_setup["cluster"]][["surface", "family"]].drop_duplicates()
    splitter = StratifiedKFold(5, shuffle=True, random_state=42)
    for fold, (_, test_rows) in enumerate(splitter.split(cluster_surfaces.surface, cluster_surfaces.family)):
        folds[selected.surface.isin(cluster_surfaces.surface.iloc[test_rows]).to_numpy()] = fold

    for name, representation in representations.items():
        started = time.time()
        for input_name in ["audio", "acceleration", "audio + acceleration"]:
            features = features_for(representation, input_name)

            predicted = np.empty(len(selected), object)
            for fold in range(5):
                train = in_setup["cluster"] & (folds != fold)
                test = in_setup["cluster"] & (folds == fold)
                predicted[test] = fit_and_predict(features, selected, train, test)
            record(
                "new surfaces: cluster",
                name,
                input_name,
                "linear probe",
                selected[in_setup["cluster"]],
                predicted[in_setup["cluster"]],
            )

            for variant, values in [
                ("linear probe", features),
                ("linear probe, per-setup standardization", standardize_per_setup(features, selected)),
            ]:
                for new_setup in SETUPS:
                    test = in_setup[new_setup]
                    record(
                        f"new setup: {new_setup}",
                        name,
                        input_name,
                        variant,
                        selected[test],
                        fit_and_predict(values, selected, ~test, test),
                    )

            if name in FEW_SHOT_REPRESENTATIONS:
                rng = np.random.default_rng(0)
                for new_setup in SETUPS:
                    for k in FEW_SHOT_K:
                        for draw in range(FEW_SHOT_DRAWS):
                            labeled = (
                                in_setup[new_setup]
                                & selected.surface.isin(
                                    labeled_surfaces(selected[in_setup[new_setup]], k, rng)
                                ).to_numpy()
                            )
                            test = in_setup[new_setup] & ~labeled
                            record(
                                f"new setup: {new_setup}",
                                name,
                                input_name,
                                "linear probe + k labeled surfaces",
                                selected[test],
                                fit_and_predict(features, selected, ~in_setup[new_setup] | labeled, test),
                                k=k,
                                draw=draw,
                            )
        print(f"{name}: {time.time() - started:.0f} s", flush=True)
        pd.DataFrame(rows).to_csv(OUT / f"setup_transfer_scores{SUFFIX}.csv", index=False)

    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / f"setup_transfer_scores{SUFFIX}.csv", index=False)
    summary = scores[scores.variant != "linear probe + k labeled surfaces"]
    print(
        summary.pivot_table(index=["representation", "input", "variant"], columns="test", values="balanced accuracy")
        .round(2)
        .to_string(),
        flush=True,
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

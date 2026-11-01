import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from cnn import DEVICE, SPECTROGRAMS_OF_INPUT, classify_recordings, load_windows, load_windows_of, train, window_labels
from evaluation import (
    COLUMN_PREFIX_OF_INPUT,
    FAMILIES,
    OUT,
    fit_and_predict,
    group_weights,
    input_features,
    plot_scores,
    score,
    shape_features_of,
    spectral_shape_features,
    zscore,
)

TRAINING_SETS = {"Cluster only": ["cluster"], "LMT108 only": ["lmt108"], "both": ["cluster", "lmt108"]}
SPECTRAL_MODELS = {"logistic regression": lambda: LogisticRegression(max_iter=2000), "RBF SVM": SVC}


def plot_confusions(predictions, sens3, name):
    fig, axes = plt.subplots(1, len(predictions), figsize=(4 * len(predictions), 3.6))
    for ax, (model_name, predicted) in zip(axes, predictions.items(), strict=True):
        confusion = confusion_matrix(sens3.family, predicted, labels=FAMILIES, normalize="true")
        ax.imshow(confusion, vmin=0, vmax=1, cmap="Blues")
        for (i, j), value in np.ndenumerate(confusion):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.6 else "black")
        ax.set_xticks(range(len(FAMILIES)), FAMILIES, rotation=30)
        ax.set_yticks(range(len(FAMILIES)), FAMILIES)
        ax.set(title=model_name, xlabel="predicted family")
    axes[0].set_ylabel("true family (rows sum to 1)")
    fig.suptitle("SENS3, all speeds: models trained on Cluster and LMT108, audio + acceleration")
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight")
    print("saved", OUT / name)


def sens3():
    recordings, shapes = spectral_shape_features()
    sens3, sens3_shapes = shape_features_of("sens3")
    tests = {
        "SENS3, all speeds": np.ones(len(sens3), dtype=bool),
        "SENS3, 20-60 mm/s": (sens3.velocity <= 60).to_numpy(),
    }
    predictions = {}
    for input_name in COLUMN_PREFIX_OF_INPUT:
        sens3_features = input_features(sens3_shapes, input_name)
        for training_name, datasets in TRAINING_SETS.items():
            training = recordings[recordings.dataset.isin(datasets)]
            features = input_features(shapes[recordings.dataset.isin(datasets)], input_name)
            per_dataset_features = features.groupby(training.dataset).transform(zscore)
            weights = group_weights(training)
            for model_name, model in SPECTRAL_MODELS.items():
                predictions[input_name, training_name, model_name] = fit_and_predict(
                    model(), features, training.family, weights, sens3_features
                )
                predictions[input_name, training_name, model_name + ", per-dataset scaling"] = fit_and_predict(
                    model(), per_dataset_features, training.family, weights, zscore(sens3_features)
                )
            print(input_name, "|", training_name, "| spectral models done", flush=True)
    recordings, spectrograms, recording_of_window, family_of_window, group_of_window = load_windows()
    _, sens3_spectrograms, sens3_recording_of_window = load_windows_of("sens3")
    window_dataset = recordings.dataset.to_numpy()[recording_of_window]
    sens3_windows = np.arange(len(sens3_recording_of_window))
    for input_name in SPECTROGRAMS_OF_INPUT:
        for training_name, datasets in TRAINING_SETS.items():
            train_windows = np.flatnonzero(np.isin(window_dataset, datasets))
            model = train(input_name, spectrograms, family_of_window, group_of_window, train_windows, seed=0)
            predicted = classify_recordings(model, sens3_spectrograms, sens3_windows, sens3_recording_of_window)
            predictions[input_name, training_name, "small CNN"] = predicted.reindex(sens3.index).to_numpy()
            print(input_name, "|", training_name, "| CNN done", flush=True)
    rows = []
    for (input_name, training_name, model_name), predicted in predictions.items():
        for test_name, tested in tests.items():
            row = {"input": input_name, "trained on": training_name, "model": model_name, "test": test_name}
            rows.append(row | score(sens3[tested], predicted[tested]))
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "sens3_test.csv", index=False)
    plot_scores(
        scores[scores["trained on"] == "both"],
        groups="input",
        bars="model",
        panels="test",
        title="External test on SENS3 (bare finger): models trained on Cluster and LMT108",
        name="sens3_test.png",
    )
    plot_scores(
        scores[scores.test == "SENS3, all speeds"],
        groups="input",
        bars="model",
        panels="trained on",
        title="SENS3, all speeds: models trained on one or both datasets",
        name="sens3_test_by_training_set.png",
    )
    confusion_models = ["logistic regression", "RBF SVM", "small CNN"]
    plot_confusions(
        {name: predictions["audio + acceleration", "both", name] for name in confusion_models},
        sens3,
        name="sens3_confusions.png",
    )


SETUPS = ["cluster", "lmt108", "sens3"]


def training_sets(held_out):
    others = [dataset for dataset in SETUPS if dataset != held_out]
    return {
        "other two setups": (others, list(COLUMN_PREFIX_OF_INPUT)),
        "other two + Greatest Hits": ([*others, "greatest_hits"], ["audio"]),
        "Greatest Hits only": (["greatest_hits"], ["audio"]),
    }


def print_and_keep(rows, row):
    rows.append(row)
    print(f"{row['tested on']} | {row['trained on']} | {row['input']} | {row['model']}: {row['balanced accuracy']:.2f}")


def leave_one_out():
    main_recordings, main_shapes = spectral_shape_features()
    sens3, sens3_shapes = shape_features_of("sens3")
    hits, hits_shapes = shape_features_of("greatest_hits")
    recordings = pd.concat([main_recordings, sens3, hits], ignore_index=True)
    shapes = pd.concat([main_shapes, sens3_shapes, hits_shapes], ignore_index=True)
    rows = []
    for held_out in SETUPS:
        tested = (recordings.dataset == held_out).to_numpy()
        for training_name, (datasets, inputs) in training_sets(held_out).items():
            training = recordings.dataset.isin(datasets).to_numpy()
            weights = group_weights(recordings[training])
            for input_name in inputs:
                features = input_features(shapes, input_name)
                for model_name, model in SPECTRAL_MODELS.items():
                    predicted = fit_and_predict(
                        model(), features[training], recordings.family[training], weights, features[tested]
                    )
                    row = {"tested on": held_out, "trained on": training_name, "input": input_name, "model": model_name}
                    print_and_keep(rows, row | score(recordings[tested], predicted))
    _, main_spectrograms, main_recording_of_window, _, _ = load_windows()
    _, sens3_spectrograms, sens3_recording_of_window = load_windows_of("sens3")
    _, hits_spectrograms, hits_recording_of_window = load_windows_of("greatest_hits")
    recording_of_window = np.concatenate(
        [
            main_recording_of_window,
            sens3_recording_of_window + len(main_recordings),
            hits_recording_of_window + len(main_recordings) + len(sens3),
        ]
    )
    accel_shape = main_spectrograms["accel"].shape[1:]
    hits_accel = torch.zeros(len(hits_recording_of_window), *accel_shape, dtype=torch.float16, device=DEVICE)
    spectrograms = {
        "audio": torch.cat([main_spectrograms["audio"], sens3_spectrograms["audio"], hits_spectrograms["audio"]]),
        "accel": torch.cat([main_spectrograms["accel"], sens3_spectrograms["accel"], hits_accel]),
    }
    family_of_window, group_of_window = window_labels(recordings, recording_of_window)
    window_dataset = recordings.dataset.to_numpy()[recording_of_window]
    for held_out in SETUPS:
        tested = recordings[recordings.dataset == held_out]
        test_windows = np.flatnonzero(window_dataset == held_out)
        for training_name, (datasets, inputs) in training_sets(held_out).items():
            train_windows = np.flatnonzero(np.isin(window_dataset, datasets))
            for input_name in inputs:
                model = train(input_name, spectrograms, family_of_window, group_of_window, train_windows, seed=0)
                predicted = classify_recordings(model, spectrograms, test_windows, recording_of_window)
                row = {"tested on": held_out, "trained on": training_name, "input": input_name, "model": "small CNN"}
                print_and_keep(rows, row | score(tested, predicted.reindex(tested.index)))
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "leave_one_setup_out.csv", index=False)
    plot_scores(
        scores[scores["trained on"] == "other two setups"],
        groups="input",
        bars="model",
        panels="tested on",
        title="Trained on the two other setups, tested on the held-out one",
        name="leave_one_setup_out.png",
    )
    plot_scores(
        scores[scores.input == "audio"],
        groups="trained on",
        bars="model",
        panels="tested on",
        title="Audio: does Greatest Hits (many rooms and objects) help on a held-out setup?",
        name="leave_one_setup_out_greatest_hits.png",
    )

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from cnn import (
    BASELINE,
    SPECTROGRAMS_OF_INPUT,
    classify_recordings,
    load_windows,
    load_windows_of,
    random_spectral_curves_db,
    train,
)
from evaluation import (
    COLUMN_PREFIX_OF_INPUT,
    OUT,
    fit_and_predict,
    group_weights,
    input_features,
    plot_scores,
    score,
    shape_features_of,
    spectral_shape_features,
)

TESTS = {
    "Cluster -> LMT108 (development)": (["cluster"], "lmt108"),
    "LMT108 -> Cluster (development)": (["lmt108"], "cluster"),
    "Cluster + LMT108 -> SENS3 (external)": (["cluster", "lmt108"], "sens3"),
}


def cnn_scores(variants):
    recordings, spectrograms, recording_of_window, family_of_window, group_of_window = load_windows()
    sens3, sens3_spectrograms, sens3_recording_of_window = load_windows_of("sens3")
    window_dataset = recordings.dataset.to_numpy()[recording_of_window]
    rows = []
    for test_name, (training_datasets, test_dataset) in TESTS.items():
        train_windows = np.flatnonzero(np.isin(window_dataset, training_datasets))
        if test_dataset == "sens3":
            tested = sens3
            test_spectrograms, test_recording_of_window = (sens3_spectrograms, sens3_recording_of_window)
            test_windows = np.arange(len(sens3_recording_of_window))
        else:
            tested = recordings[recordings.dataset == test_dataset]
            test_spectrograms, test_recording_of_window = (spectrograms, recording_of_window)
            test_windows = np.flatnonzero(window_dataset == test_dataset)
        for input_name in SPECTROGRAMS_OF_INPUT:
            for variant_name, variant in variants.items():
                model = train(
                    input_name,
                    spectrograms,
                    family_of_window,
                    group_of_window,
                    train_windows,
                    seed=0,
                    variant=variant,
                    target=(test_spectrograms, test_windows, test_dataset),
                )
                predicted = classify_recordings(model, test_spectrograms, test_windows, test_recording_of_window)
                scores = score(tested, predicted.reindex(tested.index))
                rows.append({"test": test_name, "input": input_name, "variant": variant_name, **scores})
                print(f"{test_name} | {input_name} | {variant_name}: {scores['balanced accuracy']:.2f}", flush=True)
    return rows


def mark_chosen(scores):
    development = scores[scores.test.str.contains("development")]
    mean_development = development.groupby(["input", "variant"])["balanced accuracy"].mean()
    scores = scores.merge(mean_development.rename("development mean"), on=["input", "variant"])
    best = mean_development.groupby("input").idxmax().map(lambda key: key[1])
    scores["chosen"] = scores.variant == scores.input.map(best)
    print("\nchosen on the development tests:\n" + best.to_string())
    return scores


N_AUGMENTED_COPIES = 4
DOMAIN_VARIANTS = {
    "CNN, baseline": BASELINE,
    "CNN, deeper + regularized": BASELINE | {"deeper": True, "dropout": 0.5, "weight_decay": 0.01},
    "CNN, spectral augmentation": BASELINE | {"augment": True},
    "CNN, AdaBN": BASELINE | {"adapt_batchnorm": True},
    "CNN, per-window normalization": BASELINE | {"normalize_each_window": True},
    "CNN, all combined": BASELINE
    | {
        "deeper": True,
        "dropout": 0.5,
        "weight_decay": 0.01,
        "augment": True,
        "adapt_batchnorm": True,
        "normalize_each_window": True,
    },
}


def augmented_copies(features):
    torch.manual_seed(0)
    copies = []
    for _ in range(N_AUGMENTED_COPIES):
        copy = features.to_numpy().copy()
        for prefix in ["audio ", "accel "]:
            part = features.columns.str.startswith(prefix)
            if part.any():
                shifted = copy[:, part] + random_spectral_curves_db(len(copy), part.sum()).cpu().numpy()
                copy[:, part] = shifted - shifted.mean(axis=1, keepdims=True)
        copies.append(copy)
    return np.vstack(copies)


def spectral_predictions(training, train_features, test_features):
    families, weights = (training.family.to_numpy(), group_weights(training))
    augmented = np.vstack([train_features.to_numpy(), augmented_copies(train_features)])
    n_copies = N_AUGMENTED_COPIES + 1
    return {
        "logistic regression": fit_and_predict(
            LogisticRegression(max_iter=2000), train_features, families, weights, test_features
        ),
        "logistic regression, strong L2": fit_and_predict(
            LogisticRegression(C=0.01, max_iter=2000), train_features, families, weights, test_features
        ),
        "logistic regression, augmented": fit_and_predict(
            LogisticRegression(max_iter=2000),
            augmented,
            np.tile(families, n_copies),
            np.tile(weights, n_copies),
            test_features.to_numpy(),
        ),
        "RBF SVM": fit_and_predict(SVC(), train_features, families, weights, test_features),
    }


def domain_shift():
    recordings, shapes = spectral_shape_features()
    sens3, sens3_shapes = shape_features_of("sens3")
    rows = []
    for test_name, (training_datasets, test_dataset) in TESTS.items():
        in_training = recordings.dataset.isin(training_datasets)
        if test_dataset == "sens3":
            tested, test_shapes = (sens3, sens3_shapes)
        else:
            tested, test_shapes = (
                recordings[recordings.dataset == test_dataset],
                shapes[recordings.dataset == test_dataset],
            )
        for input_name in COLUMN_PREFIX_OF_INPUT:
            predictions = spectral_predictions(
                recordings[in_training],
                input_features(shapes[in_training], input_name),
                input_features(test_shapes, input_name),
            )
            for variant_name, predicted in predictions.items():
                rows.append(
                    {"test": test_name, "input": input_name, "variant": variant_name, **score(tested, predicted)}
                )
        print(test_name, "| spectral models done", flush=True)
    rows += cnn_scores(DOMAIN_VARIANTS)
    scores = mark_chosen(pd.DataFrame(rows))
    scores.to_csv(OUT / "domain_shift_variants.csv", index=False)
    plot_scores(
        scores,
        groups="variant",
        bars="input",
        panels="test",
        title="Changes against the change of setup: trained on one setup, tested on another",
        name="domain_shift_variants.png",
    )


INVARIANCE_VARIANTS = {
    "baseline": BASELINE,
    "noise": BASELINE | {"noise": True},
    "mixup": BASELINE | {"mixup": True},
    "adversarial": BASELINE | {"adversarial": True},
    "all combined": BASELINE | {"noise": True, "mixup": True, "adversarial": True, "augment": True},
}


def invariance():
    scores = mark_chosen(pd.DataFrame(cnn_scores(INVARIANCE_VARIANTS)))
    scores.to_csv(OUT / "invariance_variants.csv", index=False)
    plot_scores(
        scores,
        groups="variant",
        bars="input",
        panels="test",
        title="Training the CNN to ignore the setup: trained on one setup, tested on another",
        name="invariance_variants.png",
    )

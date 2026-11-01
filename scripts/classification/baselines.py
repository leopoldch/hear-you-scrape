import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cnn import SPECTROGRAMS_OF_INPUT, classify_recordings, load_windows, train
from evaluation import (
    COLUMN_PREFIX_OF_INPUT,
    N_FOLDS,
    OUT,
    TEAM_SPLITS,
    fit_and_predict,
    folds,
    group_weights,
    input_features,
    plot_scores,
    score,
    scores_per_dataset,
    spectral_shape_features,
    team_seeds,
    team_train_and_test,
    zscore,
)

SPECTRAL_MODELS = {
    "majority class": lambda: DummyClassifier(strategy="most_frequent"),
    "logistic regression": lambda: LogisticRegression(class_weight="balanced", max_iter=2000),
    "LDA": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.25] * 4),
    "RBF SVM": lambda: SVC(class_weight="balanced"),
}


def classifier(model_name):
    return make_pipeline(StandardScaler(), SPECTRAL_MODELS[model_name]())


def cross_validated_predictions(model_name, features, families, fold):
    predicted = np.empty(len(families), dtype=object)
    for k in range(N_FOLDS):
        test = fold == k
        model = classifier(model_name).fit(features[~test], families[~test])
        predicted[test] = model.predict(features[test])
    return predicted


def spectral():
    recordings, shapes = spectral_shape_features()
    in_cluster = (recordings.dataset == "cluster").to_numpy()
    cluster = recordings[in_cluster]
    lmt = recordings[~in_cluster]
    cluster_families = cluster.family.to_numpy()
    cluster_fold = folds(cluster)
    rows = []
    for input_name in COLUMN_PREFIX_OF_INPUT:
        features = input_features(shapes, input_name).to_numpy()
        cluster_features = features[in_cluster]
        lmt_features = features[~in_cluster]
        for model_name in SPECTRAL_MODELS:
            print(input_name, "|", model_name, flush=True)
            cluster_scaling = classifier(model_name).fit(cluster_features, cluster_families)
            per_dataset_scaling = classifier(model_name).fit(zscore(cluster_features), cluster_families)
            tests = {
                "Cluster, new surfaces": (
                    cluster,
                    cross_validated_predictions(model_name, cluster_features, cluster_families, cluster_fold),
                ),
                "LMT108, Cluster scaling": (lmt, cluster_scaling.predict(lmt_features)),
                "LMT108, per-dataset scaling": (lmt, per_dataset_scaling.predict(zscore(lmt_features))),
            }
            for test_name, (tested, predicted) in tests.items():
                rows.append({"input": input_name, "model": model_name, "test": test_name, **score(tested, predicted)})
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "spectral_shape_models.csv", index=False)
    plot_scores(
        scores,
        groups="model",
        bars="input",
        panels="test",
        title="Simple classifiers on spectral shapes, trained on Cluster",
        name="spectral_shape_models.png",
    )


def cnn_cluster():
    recordings, spectrograms, recording_of_window, family_of_window, group_of_window = load_windows()
    in_cluster = (recordings.dataset == "cluster").to_numpy()
    window_in_cluster = in_cluster[recording_of_window]
    window_fold = folds(recordings)[recording_of_window]
    rows = []
    for input_name in SPECTROGRAMS_OF_INPUT:
        predicted = pd.Series(index=recordings.index, dtype=object)
        for k in range(N_FOLDS):
            train_windows = np.flatnonzero(window_in_cluster & (window_fold != k))
            test_windows = np.flatnonzero(window_in_cluster & (window_fold == k))
            model = train(input_name, spectrograms, family_of_window, group_of_window, train_windows, seed=k)
            predicted.update(classify_recordings(model, spectrograms, test_windows, recording_of_window))
        all_cluster_windows = np.flatnonzero(window_in_cluster)
        lmt_windows = np.flatnonzero(~window_in_cluster)
        model = train(input_name, spectrograms, family_of_window, group_of_window, all_cluster_windows, seed=N_FOLDS)
        predicted.update(classify_recordings(model, spectrograms, lmt_windows, recording_of_window))
        for test_name, tested in [("Cluster, new surfaces", in_cluster), ("LMT108, transfer", ~in_cluster)]:
            scores = score(recordings[tested], predicted[tested])
            rows.append({"input": input_name, "model": "small CNN", "test": test_name, **scores})
        print(input_name, "done", flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "cnn_on_cluster.csv", index=False)
    spectral = pd.read_csv(OUT / "spectral_shape_models.csv")
    spectral = spectral[spectral.model.isin(["logistic regression", "RBF SVM"])]
    spectral = spectral[spectral.test != "LMT108, per-dataset scaling"]
    spectral["test"] = spectral.test.replace({"LMT108, Cluster scaling": "LMT108, transfer"})
    plot_scores(
        pd.concat([spectral, scores]),
        groups="input",
        bars="model",
        panels="test",
        title="Small CNN on spectrograms against simple classifiers, trained on Cluster",
        name="cnn_on_cluster.png",
    )


TRAINING_SETS = {"Cluster only": ["cluster"], "LMT108 only": ["lmt108"], "both": ["cluster", "lmt108"]}


def cnn_combined():
    recordings, spectrograms, recording_of_window, family_of_window, group_of_window = load_windows()
    window_dataset = recordings.dataset.to_numpy()[recording_of_window]
    window_fold = folds(recordings)[recording_of_window]
    rows = []
    for input_name in SPECTROGRAMS_OF_INPUT:
        for training_name, training_datasets in TRAINING_SETS.items():
            predicted = pd.Series(index=recordings.index, dtype=object)
            for k in range(N_FOLDS):
                train_windows = np.flatnonzero(np.isin(window_dataset, training_datasets) & (window_fold != k))
                test_windows = np.flatnonzero(window_fold == k)
                model = train(input_name, spectrograms, family_of_window, group_of_window, train_windows, seed=k)
                predicted.update(classify_recordings(model, spectrograms, test_windows, recording_of_window))
            for test_dataset in ["cluster", "lmt108"]:
                tested = (recordings.dataset == test_dataset).to_numpy()
                scores = score(recordings[tested], predicted[tested])
                rows.append({"input": input_name, "trained on": training_name, "tested on": test_dataset, **scores})
            print(input_name, "|", training_name, "done", flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "cnn_on_both_datasets.csv", index=False)
    plot_scores(
        scores,
        groups="trained on",
        bars="tested on",
        panels="input",
        title="Small CNN trained on one or both datasets, tested on new surfaces of each",
        name="cnn_on_both_datasets.png",
    )


TEAM_MODELS = {"logistic regression": lambda: LogisticRegression(max_iter=2000), "RBF SVM": SVC}


def spectral_team():
    recordings, shapes = spectral_shape_features()
    rows = []
    for input_name in COLUMN_PREFIX_OF_INPUT:
        features = input_features(shapes, input_name)
        for split in TEAM_SPLITS:
            for seed in team_seeds(split):
                for model_name, model in TEAM_MODELS.items():
                    predicted = pd.Series(index=recordings.index, dtype=object)
                    for train_mask, test in team_train_and_test(recordings, split, seed):
                        weights = group_weights(recordings[train_mask])
                        predicted[test] = fit_and_predict(
                            model(), features[train_mask], recordings.family[train_mask], weights, features[test]
                        )
                    for row in scores_per_dataset(recordings, predicted):
                        rows.append({"input": input_name, "split": split, "seed": seed, "model": model_name, **row})
            print(input_name, "|", split, "done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "team_splits_spectral.csv", index=False)
    print("saved", OUT / "team_splits_spectral.csv")


def cnn_team():
    recordings, spectrograms, recording_of_window, family_of_window, group_of_window = load_windows()
    rows = []
    for input_name in SPECTROGRAMS_OF_INPUT:
        for split in TEAM_SPLITS:
            predicted = pd.Series(index=recordings.index, dtype=object)
            for train_recordings, test_recordings in team_train_and_test(recordings, split, seed=0):
                train_windows = np.flatnonzero(train_recordings[recording_of_window])
                test_windows = np.flatnonzero(test_recordings[recording_of_window])
                model = train(input_name, spectrograms, family_of_window, group_of_window, train_windows, seed=0)
                predicted.update(classify_recordings(model, spectrograms, test_windows, recording_of_window))
            for row in scores_per_dataset(recordings, predicted):
                rows.append({"input": input_name, "split": split, "seed": 0, "model": "small CNN", **row})
            pd.DataFrame(rows).to_csv(OUT / "team_splits_cnn.csv", index=False)
            print(input_name, "|", split, "done", flush=True)
    print("saved", OUT / "team_splits_cnn.csv")

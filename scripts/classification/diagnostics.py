from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from evaluation import (
    ACCEL_FREQ,
    AUDIO_FREQ,
    COLUMN_PREFIX_OF_INPUT,
    FAMILIES,
    OUT,
    fit_and_predict,
    folds,
    group_weights,
    input_features,
    load,
    plot_scores,
    score,
    shape_features_of,
    spectral_shape_features,
    zscore,
)

SETUPS_OF_INPUT = {
    "audio": ["cluster", "lmt108", "sens3", "greatest_hits"],
    "acceleration": ["cluster", "lmt108", "sens3"],
}
N_SHUFFLES = 200


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def family_means(surfaces):
    return surfaces.groupby(level=0).mean().loc[FAMILIES].to_numpy()


def family_differences(a, b):
    return family_means(b) - family_means(a)


def signature_correlations(a, b):
    signatures_a, signatures_b = (m - m.mean(axis=0) for m in (family_means(a), family_means(b)))
    correlations = np.corrcoef(signatures_a, signatures_b)[: len(FAMILIES), len(FAMILIES) :]
    same = np.eye(len(FAMILIES), dtype=bool)
    return (correlations[same].mean(), correlations[~same].mean())


def shuffled(surfaces, rng):
    return surfaces.set_index(rng.permutation(surfaces.index))


def filter():
    recordings, shapes = spectral_shape_features("sens3", "greatest_hits")
    rng = np.random.default_rng(0)
    rows, curves = ([], [])
    for input_name, setups in SETUPS_OF_INPUT.items():
        features = input_features(shapes, input_name)
        surfaces = {}
        for setup in setups:
            in_setup = recordings.dataset == setup
            by_surface = features[in_setup].groupby(recordings.surface[in_setup]).mean()
            family = recordings[in_setup].groupby("surface").family.first()
            surfaces[setup] = by_surface.set_index(family.loc[by_surface.index].rename("family"))
        for a, b in combinations(setups, 2):
            differences = family_differences(surfaces[a], surfaces[b])
            filter_curve = differences.mean(axis=0)
            noise = [
                rms(d - d.mean(axis=0))
                for d in (
                    family_differences(shuffled(surfaces[a], rng), shuffled(surfaces[b], rng))
                    for _ in range(N_SHUFFLES)
                )
            ]
            same_family, different_families = signature_correlations(surfaces[a], surfaces[b])
            separation = [rms(m - m.mean(axis=0)) for m in (family_means(surfaces[a]), family_means(surfaces[b]))]
            rows.append(
                {
                    "input": input_name,
                    "from": a,
                    "to": b,
                    "filter (dB)": rms(filter_curve),
                    "family-dependent (dB)": rms(differences - filter_curve),
                    "noise floor (dB)": float(np.median(noise)),
                    "between families (dB)": float(np.mean(separation)),
                    "signature correlation, same family": same_family,
                    "signature correlation, different families": different_families,
                }
            )
            curves.append((input_name, a, b, features.columns, differences))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "setup_filter.csv", index=False)
    print(table.round(2).to_string(index=False))
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharey=True)
    for ax, (input_name, a, b, columns, differences) in zip(axes.flat, curves, strict=False):
        frequencies = [float(column.split()[1]) for column in columns]
        for k, family in enumerate(FAMILIES):
            ax.semilogx(frequencies, differences[k], color=f"C{k}", label=family)
        ax.semilogx(frequencies, differences.mean(axis=0), color="black", linewidth=2.5, label="mean: the filter")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set(title=f"{input_name}: {b} - {a}", xlabel="frequency (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel("difference of mean shapes (dB)")
    for ax in axes.flat[len(curves) :]:
        ax.axis("off")
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.suptitle("Changing setup: one curve per family. Same curves = a filter; different curves = more than a filter")
    fig.tight_layout()
    fig.savefig(OUT / "setup_filter.png", dpi=130, bbox_inches="tight")
    print("saved", OUT / "setup_filter.png")


SETUPS = ["cluster", "lmt108", "sens3"]
SPECTRAL_MODELS = {"logistic regression": lambda: LogisticRegression(max_iter=2000), "RBF SVM": SVC}


def centered(shapes, recordings):
    # Uses only the setup name, never the families: the tested setup's labels stay unseen.
    setup_means = shapes.groupby(recordings.dataset).mean()
    return shapes - setup_means.loc[recordings.dataset].to_numpy()


def training_sets(held_out):
    others = [setup for setup in SETUPS if setup != held_out]
    return {others[0]: [others[0]], others[1]: [others[1]], "other two": others}


def centering():
    recordings, shapes = spectral_shape_features("sens3")
    versions = {
        "raw": shapes,
        "centered": centered(shapes, recordings),
    }
    rows = []
    for held_out in SETUPS:
        tested = (recordings.dataset == held_out).to_numpy()
        for training_name, datasets in training_sets(held_out).items():
            training = recordings.dataset.isin(datasets).to_numpy()
            weights = group_weights(recordings[training])
            for input_name in COLUMN_PREFIX_OF_INPUT:
                for version, version_shapes in versions.items():
                    features = input_features(version_shapes, input_name)
                    for model_name, model in SPECTRAL_MODELS.items():
                        predicted = fit_and_predict(
                            model(), features[training], recordings.family[training], weights, features[tested]
                        )
                        row = {
                            "tested on": held_out,
                            "trained on": training_name,
                            "input": input_name,
                            "version": version,
                            "model": model_name,
                        }
                        rows.append(row | score(recordings[tested], predicted))
                        print(" | ".join(map(str, row.values())), f"{rows[-1]['balanced accuracy']:.2f}", flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "setup_centering.csv", index=False)
    scores["test"] = scores["trained on"] + " -> " + scores["tested on"]
    for input_name in COLUMN_PREFIX_OF_INPUT:
        plot_scores(
            scores[scores.input == input_name],
            groups="test",
            bars="version",
            panels="model",
            title=f"Centering each setup on its mean shape ({input_name})",
            name=f"setup_centering_{input_name.replace(' + ', '_')}.png",
        )


OCTAVES = {
    "audio": (AUDIO_FREQ, [100, 200, 400, 800, 1600, 3200, 6400]),
    "accel": (ACCEL_FREQ, [10, 20, 40, 80, 160, 320, 600]),
}


def flatness(psd, freq, edges):
    columns = {}
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        band = psd[:, (freq >= low) & (freq < high)] + 1e-20
        columns[f"{low}-{high} Hz"] = 10 * (np.log10(band).mean(axis=1) - np.log10(band.mean(axis=1)))
    return pd.DataFrame(columns)


def window_descriptors(windows, recording_of_window, n_recordings):
    fluctuation, modulation = ([], [])
    for start in range(0, len(windows), 4096):
        x = windows[start : start + 4096].astype(np.float32)
        x = x - x.mean(axis=2, keepdims=True)
        fluctuation.append(x.std(axis=2))
        modulation.append(np.log(np.abs(np.fft.rfft(x, axis=2))[:, :, 1:].mean(axis=1)))
    per_recording = [
        pd.DataFrame(np.concatenate(parts)).groupby(recording_of_window).mean().reindex(range(n_recordings))
        for parts in (fluctuation, modulation)
    ]
    return per_recording


def descriptors(recordings, shapes, psd, windows_cache):
    out = {}
    for signal in ["audio", "accel"]:
        name = "audio" if signal == "audio" else "acceleration"
        fluctuation, modulation = window_descriptors(windows_cache[signal], windows_cache["recording"], len(recordings))
        out[f"{name}: spectral shape"] = shapes.loc[:, shapes.columns.str.startswith(signal)]
        out[f"{name}: impulsiveness"] = np.log(recordings[[f"{signal}_kurtosis", f"{signal}_crest"]])
        out[f"{name}: flatness"] = flatness(psd[signal], *OCTAVES[signal])
        out[f"{name}: fluctuation"] = fluctuation
        out[f"{name}: modulation"] = modulation
    return {name: features.reset_index(drop=True) for name, features in out.items()}


def invariant_features():
    main_recordings, main_shapes = spectral_shape_features()
    sens3, sens3_shapes = shape_features_of("sens3")
    recordings = pd.concat([main_recordings, sens3], ignore_index=True)
    main_descriptors = descriptors(
        main_recordings, main_shapes, load(clean=False)[1], np.load(OUT / "spectrograms.npz")
    )
    sens3_descriptors = descriptors(
        sens3, sens3_shapes, np.load(OUT / "sens3_psd_cache.npz"), np.load(OUT / "sens3_spectrograms.npz")
    )
    in_cluster = (recordings.dataset == "cluster").to_numpy()
    fold = np.full(len(recordings), -1)
    fold[: len(main_recordings)] = folds(main_recordings)
    rows = []
    for name in main_descriptors:
        features = pd.concat([main_descriptors[name], sens3_descriptors[name]], ignore_index=True)
        features = features.groupby(recordings.dataset).transform(zscore).fillna(0)
        families = recordings.family
        predicted = pd.Series(index=recordings.index, dtype=object)
        for k in range(5):
            train, test = (in_cluster & (fold != k), in_cluster & (fold == k))
            model = LogisticRegression(max_iter=2000)
            predicted[test] = fit_and_predict(
                model, features[train], families[train], group_weights(recordings[train]), features[test]
            )
        rows.append(
            {"descriptor": name, "test": "cluster, new surfaces"} | score(recordings[in_cluster], predicted[in_cluster])
        )
        for source in SETUPS:
            for target in SETUPS:
                if source == target:
                    continue
                train, test = ((recordings.dataset == source).to_numpy(), (recordings.dataset == target).to_numpy())
                model = LogisticRegression(max_iter=2000)
                predicted = fit_and_predict(
                    model, features[train], families[train], group_weights(recordings[train]), features[test]
                )
                rows.append({"descriptor": name, "test": f"{source} -> {target}"} | score(recordings[test], predicted))
        print(name, " ".join(f"{row['balanced accuracy']:.2f}" for row in rows[-7:]), flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "invariant_features.csv", index=False)
    for signal in ["audio", "acceleration"]:
        in_signal = scores.descriptor.str.startswith(signal)
        plot_scores(
            scores[in_signal].assign(descriptor=scores.descriptor[in_signal].str.split(": ").str[1], signal=signal),
            groups="test",
            bars="descriptor",
            panels="signal",
            title=f"{signal}: descriptors that a filter does not change, trained on one setup",
            name=f"invariant_features_{signal}.png",
        )

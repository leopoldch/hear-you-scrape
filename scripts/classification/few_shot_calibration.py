import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from evaluation import (
    CHANCE,
    COLUMN_PREFIX_OF_INPUT,
    FAMILIES,
    OUT,
    fit_and_predict,
    group_weights,
    input_features,
    spectral_shape_features,
    zscore,
)

SETUPS = ["cluster", "lmt108", "sens3"]
SHOTS = [0, 1, 2, 4]
N_DRAWS = 20


def labeled_surfaces(recordings, k, rng):
    surfaces = recordings[["surface", "family"]].drop_duplicates()
    return np.concatenate([rng.choice(surfaces.surface[surfaces.family == f], k, replace=False) for f in FAMILIES])


def run():
    recordings, shapes = spectral_shape_features("sens3")
    shapes = shapes.groupby(recordings.dataset).transform(zscore)
    rng = np.random.default_rng(0)
    rows = []
    for new_setup in SETUPS:
        in_new = (recordings.dataset == new_setup).to_numpy()
        for k in SHOTS:
            for draw in range(N_DRAWS if k > 0 else 1):
                labeled = in_new & recordings.surface.isin(labeled_surfaces(recordings[in_new], k, rng)).to_numpy()
                test = in_new & ~labeled
                trainings = {"other setups + k surfaces": ~in_new | labeled, "k surfaces only": labeled}
                for training_name, train in trainings.items():
                    if not train.any():
                        continue
                    for input_name in COLUMN_PREFIX_OF_INPUT:
                        features = input_features(shapes, input_name)
                        predicted = fit_and_predict(
                            LogisticRegression(max_iter=2000),
                            features[train],
                            recordings.family[train],
                            group_weights(recordings[train]),
                            features[test],
                        )
                        accuracy = balanced_accuracy_score(recordings.family[test], predicted)
                        rows.append(
                            {
                                "new setup": new_setup,
                                "training": training_name,
                                "input": input_name,
                                "k": k,
                                "draw": draw,
                                "balanced accuracy": accuracy,
                            }
                        )
            print(new_setup, k, "done", flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "few_shot_calibration.csv", index=False)
    summary = scores.groupby(["input", "new setup", "training", "k"])["balanced accuracy"]
    print(summary.mean().unstack("k").round(2).to_string())
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True, sharey=True)
    for row, input_name in enumerate(COLUMN_PREFIX_OF_INPUT):
        for column, new_setup in enumerate(SETUPS):
            ax = axes[row, column]
            for color, training_name in enumerate(["other setups + k surfaces", "k surfaces only"]):
                in_curve = (scores.input == input_name) & (scores["new setup"] == new_setup)
                curve = scores[in_curve & (scores.training == training_name)].groupby("k")["balanced accuracy"]
                ax.plot(curve.mean().index, curve.mean(), "o-", color=f"C{color}", label=training_name)
                ax.fill_between(
                    curve.mean().index, curve.quantile(0.025), curve.quantile(0.975), color=f"C{color}", alpha=0.2
                )
            ax.axhline(CHANCE, color="black", linestyle=":", linewidth=1)
            ax.set(title=f"{input_name}, new setup: {new_setup}", ylim=(0, 1), xticks=SHOTS)
    for ax in axes[-1]:
        ax.set_xlabel("labeled surfaces per family of the new setup (k)")
    for ax in axes[:, 0]:
        ax.set_ylabel("balanced accuracy on its other surfaces")
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.suptitle("Calibrating a new setup with a few labeled surfaces")
    fig.tight_layout()
    fig.savefig(OUT / "few_shot_calibration.png", dpi=130, bbox_inches="tight")
    print("saved", OUT / "few_shot_calibration.png")

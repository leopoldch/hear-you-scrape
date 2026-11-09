"""H7: does fusing encoders (BEATs + CLAP + Qwen encoder) help on a new setup, and which way of
using a few labeled surfaces of the new setup works best?

Runs on the hub after qwen_audio_features.py, on the same selected recordings, reusing the helpers
of setup_transfer_probes.py. Input: audio + acceleration. Representations: BEATs, CLAP, Qwen
encoder layer 32 alone (as in H4), and their concatenations.

Ways to use k labeled surfaces per family of the new setup (k = 0, 1, 2, 4; 10 draws, the same
draws for every representation and method so comparisons are paired):

- pooled: logistic regression on the two other setups + the k surfaces (as in H4);
- pooled, new setup upweighted: same, the k surfaces weigh as much as all other setups together;
- new setup only: trained on the k surfaces alone;
- per-setup standardization: each setup standardized with its own statistics (the new setup with
  all its unlabeled recordings), then pooled.

Output: analysis/classification/qwen_audio/fusion_few_shot_scores.csv
"""

import time

import numpy as np
import pandas as pd
from setup_transfer_probes import (
    OUT,
    SETUPS,
    classifier,
    group_weights,
    labeled_surfaces,
    load_representations,
    score,
    standardize_per_setup,
)

FUSIONS = {
    "BEATs": ["BEATs"],
    "CLAP": ["CLAP"],
    "Qwen encoder 32": ["Qwen encoder layer 32"],
    "BEATs + CLAP": ["BEATs", "CLAP"],
    "BEATs + Qwen encoder 32": ["BEATs", "Qwen encoder layer 32"],
    "BEATs + CLAP + Qwen encoder 32": ["BEATs", "CLAP", "Qwen encoder layer 32"],
}
K_VALUES = [0, 1, 2, 4]
DRAWS = 10
METHODS = ["pooled", "pooled, new setup upweighted", "new setup only", "per-setup standardization"]


def fused_features(representations, names):
    blocks = []
    for name in names:
        for signal in ["audio", "accel"]:
            blocks.append(representations[name][signal])
    return np.concatenate(blocks, axis=1)


def fit_predict(features, selected, train, test, weights):
    model = classifier(train.sum(), features.shape[1])
    model.fit(features[train], selected.family[train], logisticregression__sample_weight=weights)
    return model.predict(features[test])


def main():
    selected = pd.read_csv(OUT / "selected_recordings.csv", dtype={"surface": str})
    selected["surface"] = selected.dataset + ":" + selected.surface
    representations, _, _ = load_representations(selected)
    in_setup = {d: (selected.dataset == d).to_numpy() for d in SETUPS}
    rows = []
    for fusion, names in FUSIONS.items():
        started = time.time()
        features = fused_features(representations, names)
        per_setup = standardize_per_setup(features, selected)
        for new_setup in SETUPS:
            target = in_setup[new_setup]
            for k in K_VALUES:
                for draw in range(DRAWS if k else 1):
                    rng = np.random.default_rng([SETUPS.index(new_setup), k, draw])
                    if k:
                        labeled = target & selected.surface.isin(labeled_surfaces(selected[target], k, rng)).to_numpy()
                    else:
                        labeled = np.zeros(len(selected), bool)
                    test = target & ~labeled
                    pooled = ~target | labeled
                    weights = group_weights(selected[pooled])
                    for method in METHODS:
                        if method == "new setup only" and k == 0:
                            continue
                        if method == "pooled, new setup upweighted" and k == 0:
                            continue
                        if method == "pooled":
                            predicted = fit_predict(features, selected, pooled, test, weights)
                        elif method == "pooled, new setup upweighted":
                            upweighted = weights.copy()
                            is_new = labeled[pooled]
                            upweighted[is_new] *= upweighted[~is_new].sum() / upweighted[is_new].sum()
                            predicted = fit_predict(features, selected, pooled, test, upweighted)
                        elif method == "new setup only":
                            predicted = fit_predict(features, selected, labeled, test, group_weights(selected[labeled]))
                        else:
                            predicted = fit_predict(per_setup, selected, pooled, test, weights)
                        rows.append(
                            {
                                "representation": fusion,
                                "new setup": new_setup,
                                "k": k,
                                "draw": draw,
                                "method": method,
                                **score(selected[test], predicted, n_draws=200),
                            }
                        )
        pd.DataFrame(rows).to_csv(OUT / "fusion_few_shot_scores.csv", index=False)
        print(f"{fusion}: {time.time() - started:.0f} s", flush=True)

    scores = pd.DataFrame(rows)
    summary = scores.groupby(["representation", "method", "new setup", "k"])["balanced accuracy"].mean().unstack("k")
    print(summary.round(2).to_string(), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

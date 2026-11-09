"""H29: on the same objects and the same microphone, does a hit tell the material family better than a scratch?

Greatest Hits has hits and scratches made with a drumstick on the same objects. A logistic probe on frozen BEATs
embeddings is cross-validated by video (5 folds), so a test video is never seen in training:
  hit -> hit, scratch -> scratch, hit -> scratch, scratch -> hit.
Each training set is cut to the same number of windows, so the comparison is fair.
Prediction and reading: docs/experiments.md, H29. Exploratory.

Run on the hub from the project folder:  python scripts/hub/tap_versus_scratch.py
Output: analysis/classification/tap_versus_scratch/scores.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path.home() / "ift7030-lacha"
CACHE = ROOT / "night_2026-11-05/greatest_hits"
OUT = ROOT / "analysis/classification/tap_versus_scratch"
OUT.mkdir(parents=True, exist_ok=True)

windows = pd.read_csv(CACHE / "gh_windows.csv")
features = np.load(CACHE / "gh_embeddings.npz")["BEATs"]
is_action = {action: (windows.action == action).to_numpy() for action in ["hit", "scratch"]}
n_train_windows = int(0.8 * min(is_action["hit"].sum(), is_action["scratch"].sum()))

rng = np.random.default_rng(0)
rows = []
folds = GroupKFold(n_splits=5).split(features, windows.family, groups=windows.video)
for fold, (train, _test) in enumerate(folds):
    in_train = np.zeros(len(windows), bool)
    in_train[train] = True
    for train_action in ["hit", "scratch"]:
        candidates = np.flatnonzero(in_train & is_action[train_action])
        chosen = rng.choice(candidates, min(n_train_windows, len(candidates)), replace=False)
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
        probe.fit(features[chosen], windows.family[chosen])
        for test_action in ["hit", "scratch"]:
            tested = ~in_train & is_action[test_action]
            score = balanced_accuracy_score(windows.family[tested], probe.predict(features[tested]))
            rows.append(
                {
                    "fold": fold,
                    "trained_on": train_action,
                    "tested_on": test_action,
                    "n_train": len(chosen),
                    "score": score,
                }
            )

scores = pd.DataFrame(rows)
scores.to_csv(OUT / "scores.csv", index=False)
print(scores.groupby(["trained_on", "tested_on"]).score.agg(["mean", "min", "max"]).round(3), flush=True)

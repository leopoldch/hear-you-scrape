"""H26: do the family labels mean the same thing across setups (concept shift)?

The polymer family is not the same material from one setup to the other: Cluster's polymers are
9 rigid plastics + 1 silicone, LMT108's are 5 rubbers + 3 rigid plastics, SENS3's are 4 vinyls +
2 plastics + 1 rubber. If label inconsistency is why nothing transfers, transfer with no label of
the new setup (k = 0) should rise above chance once the families are made comparable.

Conditions (family sets), each scored against its own chance level:
    4 families, as labeled
    4 families, polymer = rigid plastic only
    3 families (wood, metal, textile)
    pairs: wood vs metal, metal vs textile, wood vs textile
Representations (probe: scaler, PCA 64, logistic regression): frozen BEATs per-recording means
(audio, acceleration, both), raw or
standardized per setup (uses the new setup's unlabeled recordings, no label).
Control: the same probe within the new setup, new surfaces (4-fold split by surface).

    uv run python scripts/classification/label_consistency.py
Output: analysis/classification/label_consistency/scores.csv, label_consistency.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("analysis/classification")
OUT = DATA / "label_consistency"
OUT.mkdir(parents=True, exist_ok=True)
SETUPS = ["cluster", "lmt108", "sens3"]

CONDITIONS = {
    "4 families, as labeled": ["wood", "metal", "textile", "polymer"],
    "4 families, polymer = rigid plastic": ["wood", "metal", "textile", "polymer"],
    "3 families (no polymer)": ["wood", "metal", "textile"],
    "wood vs metal": ["wood", "metal"],
    "metal vs textile": ["metal", "textile"],
    "wood vs textile": ["wood", "textile"],
}


def load():
    selected = pd.read_csv(DATA / "qwen_audio/selected_recordings.csv", dtype={"surface": str})
    families = pd.read_csv("analysis/surface_families.csv", dtype={"surface": str})
    selected = selected.merge(families[["dataset", "surface", "polymer_type"]], on=["dataset", "surface"], how="left")
    selected["surface"] = selected.dataset + ":" + selected.surface
    accel = np.load(DATA / "calibrated_new_setup/window_embeddings_BEATs.npz")
    audio = np.load(DATA / "cross_modal/window_embeddings_audio_BEATs.npz")
    assert np.array_equal(accel["recording"], audio["recording"])
    n = len(selected)

    def per_recording(x):
        return pd.DataFrame(x).groupby(accel["recording"]).mean().loc[np.arange(n)].to_numpy()

    beats = {"audio": per_recording(audio["features"]), "acceleration": per_recording(accel["features"])}
    beats["audio + acceleration"] = np.concatenate([beats["audio"], beats["acceleration"]], axis=1)
    return selected, beats


def standardize_per_setup(x, table):
    out = np.empty_like(x)
    for s in SETUPS:
        rows = (table.dataset == s).to_numpy()
        out[rows] = (x[rows] - x[rows].mean(0)) / (x[rows].std(0) + 1e-6)
    return out


def balanced_accuracy_with_interval(table, predicted, labels, n_draws=500):
    """Mean recall over families; interval from resampling surfaces within each family."""
    table = table.reset_index(drop=True)
    hits = pd.DataFrame({"surface": table.surface, "family": table.family, "hit": predicted == table.family.to_numpy()})
    per_surface = hits.groupby(["family", "surface"]).hit.agg(["sum", "size"])
    counts = [per_surface.loc[f].to_numpy() for f in labels]  # per family: (surfaces, [hits, recordings])
    point = np.mean([c[:, 0].sum() / c[:, 1].sum() for c in counts])
    rng = np.random.default_rng(42)
    recalls = []
    for c in counts:
        picks = rng.integers(0, len(c), (n_draws, len(c)))
        recalls.append(c[picks, 0].sum(1) / c[picks, 1].sum(1))
    draws = np.mean(recalls, axis=0)
    return point, *np.percentile(draws, [2.5, 97.5])


def probe(train_table, train_x):
    weights = 1 / train_table.groupby(["dataset", "family"]).family.transform("size").to_numpy()
    model = make_pipeline(StandardScaler(), PCA(64, random_state=0), LogisticRegression(C=0.1, max_iter=1000))
    return model.fit(train_x, train_table.family, logisticregression__sample_weight=weights / weights.mean())


def main():
    selected, beats = load()
    rows = []
    for condition, labels in CONDITIONS.items():
        keep = selected.family.isin(labels).to_numpy().copy()
        if "rigid plastic" in condition:
            keep &= ~((selected.family == "polymer") & (selected.polymer_type != "rigid plastic")).to_numpy()
        table = selected[keep].reset_index(drop=True)
        for name, x_all in beats.items():
            for scaling in ["raw", "standardized per setup"]:
                x = x_all[keep]
                if scaling != "raw":
                    x = standardize_per_setup(x, table)
                for new_setup in SETUPS:
                    target = (table.dataset == new_setup).to_numpy()
                    model = probe(table[~target], x[~target])
                    ba, low, high = balanced_accuracy_with_interval(table[target], model.predict(x[target]), labels)
                    rows.append(
                        {
                            "condition": condition,
                            "chance": 1 / len(labels),
                            "representation": name,
                            "scaling": scaling,
                            "new setup": new_setup,
                            "test": "transfer, k = 0",
                            "balanced accuracy": ba,
                            "low": low,
                            "high": high,
                        }
                    )
                    # control: within the new setup, new surfaces
                    sub, xs = table[target].reset_index(drop=True), x[target]
                    predicted = np.empty(len(sub), dtype=object)
                    for train, test in GroupKFold(4).split(xs, groups=sub.surface):
                        predicted[test] = probe(sub.iloc[train], xs[train]).predict(xs[test])
                    ba, low, high = balanced_accuracy_with_interval(sub, predicted, labels)
                    rows.append(
                        {
                            "condition": condition,
                            "chance": 1 / len(labels),
                            "representation": name,
                            "scaling": scaling,
                            "new setup": new_setup,
                            "test": "within setup, new surfaces",
                            "balanced accuracy": ba,
                            "low": low,
                            "high": high,
                        }
                    )
        print(condition, "done", flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "scores.csv", index=False)
    table = scores.pivot_table(
        index=["condition", "test", "representation", "scaling"], columns="new setup", values="balanced accuracy"
    )
    table["mean"] = table.mean(axis=1)
    table["chance"] = scores.groupby(["condition", "test", "representation", "scaling"]).chance.first()
    pd.set_option("display.width", 200)
    print(table.round(2).to_string())
    plot(scores)


def plot(scores):
    best = scores[(scores.representation == "audio + acceleration") & (scores.scaling == "standardized per setup")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    colors = {"cluster": "#2a78d6", "lmt108": "#eb6834", "sens3": "#1baf7a"}
    conditions = list(CONDITIONS)
    for ax, test in zip(axes, ["transfer, k = 0", "within setup, new surfaces"]):
        part = best[best.test == test]
        for j, setup in enumerate(SETUPS):
            values = [
                part[(part.condition == c) & (part["new setup"] == setup)]["balanced accuracy"].iloc[0]
                for c in conditions
            ]
            ax.barh(
                np.arange(len(conditions)) + (j - 1) * 0.27, values, height=0.25, color=colors[setup], label=f"{setup}"
            )
        for i, c in enumerate(conditions):
            chance = 1 / len(CONDITIONS[c])
            ax.plot([chance, chance], [i - 0.45, i + 0.45], color="0.2", linestyle=":", linewidth=1.5)
        ax.set_yticks(range(len(conditions)), conditions)
        ax.set(xlim=(0, 1), xlabel="balanced accuracy", title=f"{test} (new setup = colour)")
        ax.grid(axis="x", color="0.9")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle(
        "H26: are the family labels comparable across setups? "
        "BEATs audio + acceleration, standardized per setup; dotted = chance"
    )
    fig.tight_layout()
    fig.savefig(OUT / "label_consistency.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()

"""H25: a shared audio <-> acceleration space learned by contrast, then zero-shot to a new setup.

See docs/experiments.md (H25). Frozen BEATs window embeddings (audio and
acceleration of the same 0.5 s window), two small projectors trained with a symmetric InfoNCE loss
(positive = audio and acceleration of the same window; no family label), then family probes
trained on two setups and tested on the third (k = 0), and k = 2 "new setup only".

    uv run python scripts/classification/cross_modal_contrastive.py
Output: analysis/classification/cross_modal/scores.csv, cross_modal.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrated_new_setup import (
    DATA,
    DEV_SETUPS,
    FAMILIES,
    beats_embedder,
    embed_windows,  # noqa: E402
    group_weights,
    labeled_surfaces,
    score,
)
from few_label_classifiers import window_embeddings  # noqa: E402

OUT = DATA / "cross_modal"
OUT.mkdir(parents=True, exist_ok=True)
N_STEPS, BATCH, TEMPERATURE = 3000, 512, 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def audio_window_embeddings(selected, accel_recording):
    """BEATs embedding of every audio window (16 kHz), in the same order as the acceleration windows."""
    path = OUT / "window_embeddings_audio_BEATs.npz"
    if not path.exists():
        features, owner = [], []
        for waves_file, group in selected.groupby("waves_file"):
            cache = np.load(DATA / waves_file)
            row_of_recording = pd.Series(group.index.to_numpy(), index=group.index_in_waves.to_numpy())
            keep = np.isin(cache["recording"], group.index_in_waves.to_numpy())
            windows = torch.from_numpy(cache["audio"][keep].astype(np.float32))
            features.append(embed_windows(beats_embedder, windows, 128))
            owner.append(row_of_recording.loc[cache["recording"][keep]].to_numpy())
            print(waves_file, "audio windows embedded", flush=True)
        np.savez(path, features=np.concatenate(features), recording=np.concatenate(owner))
    cache = np.load(path)
    assert np.array_equal(cache["recording"], accel_recording), "audio and acceleration windows are not aligned"
    return cache["features"]


class Projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 512), nn.GELU(), nn.Linear(512, 128))

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=-1)


def train_projectors(audio, accel, window_setup, train_setups):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    project_audio, project_accel = Projector().to(DEVICE), Projector().to(DEVICE)
    optimizer = torch.optim.AdamW(
        list(project_audio.parameters()) + list(project_accel.parameters()), lr=1e-3, weight_decay=1e-4
    )
    pools = [np.flatnonzero(window_setup == s) for s in train_setups]
    a, x = torch.as_tensor(audio, device=DEVICE), torch.as_tensor(accel, device=DEVICE)
    for _step in range(N_STEPS):
        chosen = np.concatenate([rng.choice(pool, BATCH // len(pools)) for pool in pools])
        za, zx = project_audio(a[chosen]), project_accel(x[chosen])
        logits = za @ zx.T / TEMPERATURE
        target = torch.arange(len(chosen), device=DEVICE)
        loss = (nn.functional.cross_entropy(logits, target) + nn.functional.cross_entropy(logits.T, target)) / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        retrieval = (logits.argmax(1) == target).float().mean().item()
        out = {
            "audio projected": project_audio(a).cpu().numpy(),
            "acceleration projected": project_accel(x).cpu().numpy(),
        }
    out["shared (mean of both)"] = (out["audio projected"] + out["acceleration projected"]) / 2
    return out, loss.item(), retrieval


def per_recording(window_features, recording, n):
    return pd.DataFrame(window_features).groupby(recording).mean().loc[np.arange(n)].to_numpy()


def probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))


def probabilities(features, selected, train, test):
    model = probe().fit(
        features[train], selected.family[train], logisticregression__sample_weight=group_weights(selected[train])
    )
    order = [list(model.classes_).index(f) for f in FAMILIES]
    return model.predict_proba(features[test])[:, order]


def setup_readability(features, selected):
    z = StandardScaler().fit_transform(features)
    codes = selected.surface.astype("category").cat.codes.to_numpy()
    half = codes % 2 == 0
    centers = np.stack([z[half & (selected.dataset == s).to_numpy()].mean(0) for s in DEV_SETUPS])
    predicted = np.array(DEV_SETUPS)[((z[~half][:, None] - centers[None]) ** 2).sum(-1).argmin(1)]
    return (predicted == selected.dataset.to_numpy()[~half]).mean()


def main():
    selected, accel, recording = window_embeddings()
    audio = audio_window_embeddings(selected, recording)
    window_setup = selected.dataset.to_numpy()[recording]
    n = len(selected)
    beats = {
        "BEATs audio": per_recording(audio, recording, n),
        "BEATs acceleration": per_recording(accel, recording, n),
    }
    beats["BEATs audio + acceleration"] = np.concatenate([beats["BEATs audio"], beats["BEATs acceleration"]], axis=1)

    rows, readability = [], []
    for name, features in beats.items():
        readability.append(
            {
                "representation": name,
                "protocol": "-",
                "new setup": "all",
                "setup readable": setup_readability(features, selected),
            }
        )
    for new_setup in DEV_SETUPS:
        target = (selected.dataset == new_setup).to_numpy()
        sources = [s for s in DEV_SETUPS if s != new_setup]
        representations = {(name, "-"): f for name, f in beats.items()}
        for protocol, train_setups in [("strict", sources), ("transductive", DEV_SETUPS)]:
            projected, loss, retrieval = train_projectors(audio, accel, window_setup, train_setups)
            print(
                f"{new_setup} held out, {protocol}: final loss {loss:.2f}, "
                f"audio->accel retrieval in batch {retrieval:.2f}",
                flush=True,
            )
            for name, window_features in projected.items():
                features = per_recording(window_features, recording, n)
                representations[(name, protocol)] = features
                readability.append(
                    {
                        "representation": name,
                        "protocol": protocol,
                        "new setup": new_setup,
                        "setup readable": setup_readability(features, selected),
                    }
                )
        for (name, protocol), features in representations.items():
            predicted = np.array(FAMILIES)[probabilities(features, selected, ~target, target).argmax(1)]
            rows.append(
                {
                    "representation": name,
                    "protocol": protocol,
                    "new setup": new_setup,
                    "k": 0,
                    "draw": 0,
                    **score(selected[target], predicted),
                }
            )
            for draw in range(10):
                rng = np.random.default_rng([DEV_SETUPS.index(new_setup), 2, draw])
                labeled = target & selected.surface.isin(labeled_surfaces(selected[target], 2, rng)).to_numpy()
                predicted = np.array(FAMILIES)[probabilities(features, selected, labeled, target & ~labeled).argmax(1)]
                rows.append(
                    {
                        "representation": name,
                        "protocol": protocol,
                        "new setup": new_setup,
                        "k": 2,
                        "draw": draw,
                        **score(selected[target & ~labeled], predicted, n_draws=200),
                    }
                )
        # late fusion of the separate audio and acceleration probes
        for k in [0, 2]:
            for draw in range(10) if k else [0]:
                if k:
                    rng = np.random.default_rng([DEV_SETUPS.index(new_setup), 2, draw])
                    labeled = target & selected.surface.isin(labeled_surfaces(selected[target], 2, rng)).to_numpy()
                    train, test = labeled, target & ~labeled
                else:
                    train, test = ~target, target
                p = (
                    probabilities(beats["BEATs audio"], selected, train, test)
                    + probabilities(beats["BEATs acceleration"], selected, train, test)
                ) / 2
                rows.append(
                    {
                        "representation": "late fusion (audio probe + acceleration probe)",
                        "protocol": "-",
                        "new setup": new_setup,
                        "k": k,
                        "draw": draw,
                        **score(selected[test], np.array(FAMILIES)[p.argmax(1)], n_draws=200 if k else 500),
                    }
                )
        print(new_setup, "evaluated", flush=True)

    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "scores.csv", index=False)
    pd.DataFrame(readability).to_csv(OUT / "setup_readability.csv", index=False)
    table = (
        scores.groupby(["representation", "protocol", "k", "new setup"])["balanced accuracy"]
        .mean()
        .unstack("new setup")
    )
    table["mean"] = table.mean(axis=1)
    print(table.round(3).to_string())
    low = scores[scores.k == 0].pivot_table(index=["representation", "protocol"], columns="new setup", values="low")
    print("k = 0, lower bound of the interval\n", low.round(2).to_string())
    print(
        pd.DataFrame(readability).groupby(["representation", "protocol"])["setup readable"].mean().round(2).to_string()
    )
    plot(scores)


def plot(scores):
    means = scores.groupby(["representation", "protocol", "k", "new setup"])["balanced accuracy"].mean().reset_index()
    names = means[["representation", "protocol"]].drop_duplicates().itertuples(index=False, name=None)
    names = list(names)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    colors = {"cluster": "#2a78d6", "lmt108": "#eb6834", "sens3": "#1baf7a"}
    for ax, k in zip(axes, [0, 2]):
        for j, setup in enumerate(DEV_SETUPS):
            values = [
                means[
                    (means.representation == r) & (means.protocol == p) & (means.k == k) & (means["new setup"] == setup)
                ]["balanced accuracy"].iloc[0]
                for r, p in names
            ]
            ax.barh(
                np.arange(len(names)) + (j - 1) * 0.27,
                values,
                height=0.25,
                color=colors[setup],
                label=f"{setup} held out",
            )
        ax.axvline(0.25, color="0.4", linestyle=":", linewidth=1)
        ax.set_yticks(range(len(names)), [f"{r}" + (f" [{p}]" if p != "-" else "") for r, p in names])
        ax.set(
            xlim=(0, 1),
            xlabel="balanced accuracy",
            title="k = 0: no label of the new setup" if k == 0 else "k = 2 surfaces per family, new setup only",
        )
        ax.grid(axis="x", color="0.9")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("H25: shared audio <-> acceleration space learned by contrast (no family label); dotted line = chance")
    fig.tight_layout()
    fig.savefig(OUT / "cross_modal.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()

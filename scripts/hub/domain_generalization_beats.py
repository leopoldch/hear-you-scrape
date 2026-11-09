"""H10-H14: domain-generalization training of BEATs, then weight fusion (model soup).

See docs/experiments.md (H10-H14) for the protocol fixed before running.
For each new setup, BEATs (audio + acceleration) is fine-tuned on 80% of the surfaces of the other
two setups with: ERM, DANN, cross-setup supervised contrastive, CORAL, synthetic probe responses.
The 5 models are then averaged (uniform soup). Every model is evaluated with its own head on the
new setup (k = 0) and on held-out surfaces of the known setups, and with few-shot probes on its
embeddings (k = 0, 1, 2, 4 labeled surfaces per family of the new setup, 10 draws).

Runs on the hub from the project folder:
    python scripts/hub/domain_generalization_beats.py [--quick]
Output: analysis/classification/domain_generalization/{scores,few_shot_scores}.csv
"""

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from augmented_beats_transfer import (
    BATCH,
    ENCODER_LR,
    FAMILIES,
    HEAD_LR,
    RATE,
    SEED,
    SETUPS,
    WAVE_RMS,
    control_surfaces,
    load_beats,
    load_data,
    score,
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

QUICK = "--quick" in sys.argv
N_UPDATES = 20 if QUICK else 1500
METHODS = ["erm", "dann", "cross-setup contrastive", "coral", "synthetic probe"]
SIGNALS = ["audio", "accel"]
K_VALUES = [0, 1, 2, 4]
DRAWS = 3 if QUICK else 10
OUT = Path("analysis/classification/domain_generalization")
OUT.mkdir(parents=True, exist_ok=True)
SUFFIX = "_quick" if QUICK else ""
# optional: --only SETUP (one held-out setup) and --batch N (smaller GPU); results go to a separate file
ONLY = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
if "--batch" in sys.argv:
    BATCH = int(sys.argv[sys.argv.index("--batch") + 1])
    SUFFIX += f"_batch{BATCH}"
if ONLY:
    SUFFIX += f"_{ONLY}"
DEVICE = "cuda"


class Model(nn.Module):
    def __init__(self, n_setups):
        super().__init__()
        self.encoder = load_beats()
        width = self.encoder.cfg.encoder_embed_dim * len(SIGNALS)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(width, len(FAMILIES)))
        self.setup_head = nn.Sequential(nn.Linear(width, 256), nn.ReLU(), nn.Linear(256, n_setups))
        self.projection = nn.Sequential(nn.Linear(width, 256), nn.ReLU(), nn.Linear(256, 128))

    def embed(self, batch):
        return torch.cat(
            [self.encoder.extract_features(batch[name].float() * WAVE_RMS)[0].mean(dim=1) for name in SIGNALS], dim=1
        )


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(context, x, strength):
        context.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(context, grad):
        return -context.strength * grad, None


def cross_setup_contrastive(z, families, setups, temperature=0.1):
    z = nn.functional.normalize(z, dim=1)
    similarity = z @ z.T / temperature
    not_self = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
    positives = (families[:, None] == families[None, :]) & (setups[:, None] != setups[None, :])
    log_probability = similarity - torch.logsumexp(similarity.masked_fill(~not_self, -1e9), dim=1, keepdim=True)
    has_positive = positives.any(dim=1)
    if not has_positive.any():
        return similarity.sum() * 0
    per_anchor = (log_probability * positives).sum(1)[has_positive] / positives.sum(1)[has_positive]
    return -per_anchor.mean()


def coral(features, setups):
    groups = [features[setups == s] for s in setups.unique()]
    groups = [g for g in groups if len(g) > 2]
    if len(groups) < 2:
        return features.sum() * 0
    a, b = groups[0], groups[1]
    d = features.shape[1]
    mean_gap = (a.mean(0) - b.mean(0)).pow(2).sum() / d
    covariance_gap = (torch.cov(a.T) - torch.cov(b.T)).pow(2).sum() / d**2
    return mean_gap + covariance_gap * d


def synthetic_probe(batch, generator, probability=0.8):
    """Convolve each window with a random modal impulse response (a virtual probe or finger)."""
    n = batch.shape[1]
    time_axis = torch.arange(n // 4, device=batch.device) / RATE
    out = batch.clone()
    for i in range(len(batch)):
        if float(torch.rand(1, generator=generator, device=batch.device)) > probability:
            continue
        n_modes = int(torch.randint(3, 21, (1,), generator=generator, device=batch.device))
        frequencies = torch.exp(
            torch.empty(n_modes, device=batch.device).uniform_(np.log(100), np.log(6000), generator=generator)
        )
        decays = torch.empty(n_modes, device=batch.device).uniform_(0.001, 0.03, generator=generator)
        amplitudes = torch.rand(n_modes, generator=generator, device=batch.device)
        response = (
            amplitudes[:, None]
            * torch.exp(-time_axis[None] / decays[:, None])
            * torch.sin(2 * np.pi * frequencies[:, None] * time_axis[None])
        ).sum(0)
        response[0] += 1.0
        convolved = torch.fft.irfft(torch.fft.rfft(batch[i], 2 * n) * torch.fft.rfft(response, 2 * n), 2 * n)[:n]
        out[i] = convolved / (convolved.std() + 1e-8)
    return out


def train(method, waves, recording_of_window, table, train_recordings, setup_index):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    model = Model(n_setups=len(SETUPS)).to(DEVICE)
    extra = [p for name in ["setup_head", "projection"] for p in getattr(model, name).parameters()]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": ENCODER_LR},
            {"params": list(model.head.parameters()) + extra, "lr": HEAD_LR},
        ],
        weight_decay=0.01,
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[ENCODER_LR, HEAD_LR], total_steps=N_UPDATES, pct_start=0.1
    )

    chosen_table = table.loc[train_recordings]
    weights = 1 / chosen_table.groupby(["dataset", "family"]).family.transform("size").to_numpy()
    weights = weights / weights.sum()
    first_window = np.searchsorted(recording_of_window, train_recordings)
    n_windows = np.searchsorted(recording_of_window, train_recordings, side="right") - first_window
    family_index = chosen_table.family.map({f: k for k, f in enumerate(FAMILIES)}).to_numpy()
    setup_of = chosen_table.dataset.map(setup_index).to_numpy()

    model.train()
    for step in range(N_UPDATES):
        chosen = rng.choice(len(train_recordings), BATCH, p=weights)
        windows = first_window[chosen] + rng.integers(0, n_windows[chosen])
        batch = {name: waves[name][windows].to(DEVICE).float() for name in SIGNALS}
        if method == "synthetic probe":
            batch = {name: synthetic_probe(values, generator) for name, values in batch.items()}
        families = torch.as_tensor(family_index[chosen], device=DEVICE)
        setups = torch.as_tensor(setup_of[chosen], device=DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features = model.embed(batch)
        features = features.float()
        loss = nn.functional.cross_entropy(model.head(features), families)
        if method == "dann":
            strength = 2 / (1 + np.exp(-10 * step / N_UPDATES)) - 1
            setup_logits = model.setup_head(ReverseGradient.apply(features, strength))
            loss = loss + 0.5 * nn.functional.cross_entropy(setup_logits, setups)
        elif method == "cross-setup contrastive":
            loss = loss + 0.5 * cross_setup_contrastive(model.projection(features), families, setups)
        elif method == "coral":
            loss = loss + coral(features, setups)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
    model.cpu()
    del optimizer, schedule
    gc.collect()
    torch.cuda.empty_cache()
    return model


def soup(models):
    states = [m.state_dict() for m in models]
    averaged_state = {
        key: torch.stack([s[key] for s in states]).mean(0) if value.is_floating_point() else value
        for key, value in states[0].items()
    }
    averaged = Model(n_setups=len(SETUPS))
    averaged.load_state_dict(averaged_state)
    return averaged


@torch.no_grad()
def window_outputs(model, waves, windows):
    model.to(DEVICE).eval()
    embeddings, probabilities = [], []
    for start in range(0, len(windows), 256):
        chosen = windows[start : start + 256]
        batch = {name: waves[name][chosen].to(DEVICE) for name in SIGNALS}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features = model.embed(batch).float()
            probabilities.append(model.head(features).float().softmax(1).cpu())
        embeddings.append(features.cpu())
    model.cpu()
    return torch.cat(embeddings).numpy(), torch.cat(probabilities).numpy()


def per_recording(values, recording_ids):
    return pd.DataFrame(values).groupby(recording_ids).mean()


def labeled_surfaces(table, k, rng):
    surfaces = table[["surface", "family"]].drop_duplicates()
    return np.concatenate([rng.choice(surfaces.surface[surfaces.family == f], k, replace=False) for f in FAMILIES])


def few_shot(embeddings, probe_table, new_setup):
    rows = []
    target = (probe_table.dataset == new_setup).to_numpy()
    for k in K_VALUES:
        for draw in range(DRAWS if k else 1):
            rng = np.random.default_rng([SETUPS.index(new_setup), k, draw])
            labeled = (
                target & probe_table.surface.isin(labeled_surfaces(probe_table[target], k, rng)).to_numpy()
                if k
                else np.zeros(len(target), bool)
            )
            test, train = target & ~labeled, ~target | labeled
            weights = 1 / probe_table[train].groupby(["dataset", "family"]).family.transform("size").to_numpy()
            model = make_pipeline(
                StandardScaler(), PCA(min(256, train.sum() - 1), random_state=0), LogisticRegression(max_iter=3000)
            )
            model.fit(
                embeddings[train], probe_table.family[train], logisticregression__sample_weight=weights / weights.mean()
            )
            rows.append(
                {"k": k, "draw": draw, **score(probe_table[test], model.predict(embeddings[test]), n_draws=200)}
            )
    return rows


def main():
    table, waves, recording_of_window = load_data()
    selected = pd.read_csv("analysis/classification/qwen_audio/selected_recordings.csv", dtype={"surface": str})
    offset = {"waveforms.npz": 0, "sens3_waveforms.npz": (table.dataset != "sens3").sum()}
    probe_recordings = (selected.index_in_waves + selected.waves_file.map(offset)).to_numpy()
    if QUICK:
        probe_recordings = (
            pd.Series(probe_recordings).groupby(table.surface.loc[probe_recordings].to_numpy()).head(3).to_numpy()
        )
    probe_table = table.loc[probe_recordings].reset_index(drop=True)
    probe_windows = np.flatnonzero(np.isin(recording_of_window, probe_recordings))

    done = set()
    scores_path, few_shot_path = OUT / f"scores{SUFFIX}.csv", OUT / f"few_shot_scores{SUFFIX}.csv"
    if few_shot_path.exists():
        done = set(pd.read_csv(few_shot_path)["new setup"].unique())

    for new_setup in SETUPS:
        if new_setup in done or (ONLY and new_setup != ONLY):
            continue
        training_setups = [s for s in SETUPS if s != new_setup]
        setup_index = {s: k for k, s in enumerate(training_setups)}
        held = control_surfaces(table, training_setups)
        in_training = table.dataset.isin(training_setups)
        train_recordings = table.index[in_training & ~table.surface.isin(held)].to_numpy()
        control_recordings = table.index[in_training & table.surface.isin(held)].to_numpy()
        test_recordings = table.index[table.dataset == new_setup].to_numpy()
        if QUICK:
            keep = table.groupby("surface").head(3).index
            train_recordings, control_recordings, test_recordings = (
                np.intersect1d(r, keep) for r in (train_recordings, control_recordings, test_recordings)
            )

        models = {}
        for method in METHODS:
            started = time.time()
            models[method] = train(method, waves, recording_of_window, table, train_recordings, setup_index)
            print(f"{new_setup}, {method}: trained in {time.time() - started:.0f} s", flush=True)
        models["soup of the 5"] = soup([models[m] for m in METHODS])

        score_rows, few_shot_rows = [], []
        for name, model in models.items():
            for test_name, recordings in [
                (f"new setup: {new_setup}", test_recordings),
                ("known setups, new surfaces", control_recordings),
            ]:
                windows = np.flatnonzero(np.isin(recording_of_window, recordings))
                _, probabilities = window_outputs(model, waves, windows)
                predicted = (
                    per_recording(probabilities, recording_of_window[windows])
                    .idxmax(axis=1)
                    .map(dict(enumerate(FAMILIES)))
                )
                score_rows.append(
                    {
                        "new setup": new_setup,
                        "model": name,
                        "test": test_name,
                        **score(table.loc[predicted.index], predicted.to_numpy(), n_draws=200 if QUICK else 1000),
                    }
                )
            embeddings, _ = window_outputs(model, waves, probe_windows)
            recording_embeddings = (
                per_recording(embeddings, recording_of_window[probe_windows]).loc[probe_recordings].to_numpy()
            )
            few_shot_rows += [
                {"new setup": new_setup, "model": name, **row}
                for row in few_shot(recording_embeddings, probe_table, new_setup)
            ]
            scores_of_model = [f"{r['test']} {r['balanced accuracy']:.2f}" for r in score_rows if r["model"] == name]
            k4 = [r["balanced accuracy"] for r in few_shot_rows if r["model"] == name and r["k"] == 4]
            print(f"{new_setup}, {name}: {', '.join(scores_of_model)}; few-shot k=4 {np.mean(k4):.2f}", flush=True)

        pd.DataFrame(score_rows).to_csv(scores_path, mode="a", header=not scores_path.exists(), index=False)
        pd.DataFrame(few_shot_rows).to_csv(few_shot_path, mode="a", header=not few_shot_path.exists(), index=False)
        del models
        gc.collect()
        torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

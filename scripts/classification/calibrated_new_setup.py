"""Final proposal: recognize the family on a new setup after a short calibration
(k labeled surfaces per family of the new setup), acceleration only.

Two steps, so the method is frozen before the locked test (docs/experiments.md):

    python scripts/classification/calibrated_new_setup.py select
        Development only (Cluster, LMT108, SENS3, recordings of selected_recordings.csv).
        For each candidate representation (BEATs, CLAP, BEATs + CLAP; frozen, acceleration) and
        each way of using the k surfaces, train on two setups + k surfaces of the third, test on
        the rest of the third; k = 1, 2; 10 draws. Writes the best candidate (mean over the 3 setups
        and both k) to analysis/classification/calibrated_new_setup/frozen_choice.json.

    python scripts/classification/calibrated_new_setup.py test_hatt
        Locked test, run once: refuses to run without frozen_choice.json. Trains the frozen
        candidate on all development recordings + k HaTT surfaces per family, tests on the other
        HaTT recordings; k = 0, 1, 2; 20 draws (HaTT has only 3 wood surfaces).

Embeddings are cached in analysis/classification/calibrated_new_setup/embeddings_*.npy.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "analysis/classification"
OUT = DATA / "calibrated_new_setup"
OUT.mkdir(parents=True, exist_ok=True)
FAMILIES = ["wood", "metal", "textile", "polymer"]
DEV_SETUPS = ["cluster", "lmt108", "sens3"]
CANDIDATES = {"BEATs": ["BEATs"], "CLAP": ["CLAP"], "BEATs + CLAP": ["BEATs", "CLAP"]}
METHODS = ["pooled", "pooled, new setup upweighted", "new setup only", "per-setup standardization"]
WAVE_RMS = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- embeddings of acceleration windows ----------


def accel_windows_16k(accel_8k):
    out = []
    for start in range(0, len(accel_8k), 4096):
        piece = torch.from_numpy(accel_8k[start : start + 4096].astype(np.float32))
        out.append(torchaudio.functional.resample(piece, 8000, 16000))
    return torch.cat(out)


def beats_embedder():
    sys.path.insert(0, str(ROOT / "models/beats"))
    from BEATs import BEATs, BEATsConfig

    checkpoint = torch.load(ROOT / "models/beats/BEATs_iter3_plus_AS2M.pt", map_location="cpu", weights_only=True)
    model = BEATs(BEATsConfig(checkpoint["cfg"]))
    model.load_state_dict(checkpoint["model"])
    model = model.to(DEVICE).eval()
    return lambda x: model.extract_features(x * WAVE_RMS)[0].mean(dim=1)


def clap_embedder():
    from transformers import ClapModel, ClapProcessor

    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(DEVICE).eval()
    extractor = processor.feature_extractor
    filters = torch.as_tensor(extractor.mel_filters_slaney, dtype=torch.float32, device=DEVICE)
    window = torch.hann_window(extractor.fft_window_size, device=DEVICE)

    def embed(x):
        x48 = torchaudio.functional.resample(x, 16000, 48000) * WAVE_RMS
        n = extractor.nb_max_samples
        tiled = x48.repeat(1, n // x48.shape[1])
        tiled = torch.nn.functional.pad(tiled, (0, n - tiled.shape[1]))
        power = (
            torch.stft(
                tiled,
                extractor.fft_window_size,
                extractor.hop_length,
                window=window,
                center=True,
                pad_mode="reflect",
                return_complex=True,
            ).abs()
            ** 2
        )
        mel = 10 * torch.log10(torch.clamp(filters.T @ power, min=1e-10)).transpose(1, 2)
        output = model.get_audio_features(
            input_features=mel[:, None], is_longer=torch.zeros(len(x), 1, dtype=torch.bool, device=DEVICE)
        )
        return output if torch.is_tensor(output) else output.pooler_output

    return embed


@torch.no_grad()
def embed_windows(make_embedder, windows, batch_size):
    embed = make_embedder()
    out = []
    for start in range(0, len(windows), batch_size):
        out.append(embed(windows[start : start + batch_size].to(DEVICE)).float().cpu())
    del embed
    torch.cuda.empty_cache()
    return torch.cat(out).numpy()


def recording_embeddings(name, accel_8k, recording_of_window, wanted_recordings):
    """Per-recording mean embeddings (BEATs and CLAP) of the acceleration, cached."""
    keep = np.isin(recording_of_window, wanted_recordings)
    windows, ids = None, recording_of_window[keep]
    arrays = {}
    for encoder, make, batch in [("BEATs", beats_embedder, 128), ("CLAP", clap_embedder, 32)]:
        path = OUT / f"embeddings_{name}_{encoder}.npy"
        if not path.exists():
            windows = accel_windows_16k(accel_8k[keep]) if windows is None else windows
            per_window = embed_windows(make, windows, batch)
            np.save(path, pd.DataFrame(per_window).groupby(ids).mean().loc[wanted_recordings].to_numpy(np.float32))
            print(name, encoder, "embedded", flush=True)
        arrays[encoder] = np.load(path)
    return arrays


def development_data():
    selected = pd.read_csv(DATA / "qwen_audio/selected_recordings.csv", dtype={"surface": str})
    selected["surface"] = selected.dataset + ":" + selected.surface
    blocks = {"BEATs": np.zeros((len(selected), 768), np.float32), "CLAP": np.zeros((len(selected), 512), np.float32)}
    for waves_file, group in selected.groupby("waves_file"):
        cache = np.load(DATA / waves_file)
        arrays = recording_embeddings(
            waves_file.replace(".npz", ""), cache["accel"], cache["recording"], group.index_in_waves.to_numpy()
        )
        for encoder in blocks:
            blocks[encoder][group.index] = arrays[encoder]
    return selected, blocks


def hatt_data():
    table = pd.read_csv(DATA / "hatt_recordings.csv")
    table["surface"] = "hatt:" + table.surface
    cache = np.load(DATA / "hatt_waveforms.npz")
    return table, recording_embeddings("hatt", cache["accel"], cache["recording"], np.arange(len(table)))


# ---------- calibration on a new setup ----------


def features_of(blocks, candidate):
    return np.concatenate([blocks[name] for name in CANDIDATES[candidate]], axis=1)


def group_weights(table):
    weights = 1 / table.groupby(["dataset", "family"]).family.transform("size").to_numpy()
    return weights / weights.mean()


def standardize_per_setup(features, table):
    out = np.empty_like(features)
    for dataset in table.dataset.unique():
        rows = (table.dataset == dataset).to_numpy()
        out[rows] = (features[rows] - features[rows].mean(0)) / (features[rows].std(0) + 1e-6)
    return out


def calibrate_and_predict(features, table, new_setup, labeled, method):
    target = (table.dataset == new_setup).to_numpy()
    test = target & ~labeled
    train = labeled if method == "new setup only" else ~target | labeled
    if method == "per-setup standardization":
        features = standardize_per_setup(features, table)
    weights = group_weights(table[train])
    if method == "pooled, new setup upweighted" and labeled.any():
        is_new = labeled[train]
        weights[is_new] *= weights[~is_new].sum() / weights[is_new].sum()
    model = make_pipeline(
        StandardScaler(), PCA(min(256, train.sum() - 1), random_state=0), LogisticRegression(max_iter=3000)
    )
    model.fit(features[train], table.family[train], logisticregression__sample_weight=weights)
    return test, model.predict(features[test])


def labeled_surfaces(table, k, rng):
    surfaces = table[["surface", "family"]].drop_duplicates()
    return np.concatenate([rng.choice(surfaces.surface[surfaces.family == f], k, replace=False) for f in FAMILIES])


def mean_recall(confusion):
    return np.mean(np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1))


def score(table, predicted, n_draws=500):
    table = table.reset_index(drop=True)
    predicted = np.asarray(predicted)
    confusions = {
        s: confusion_matrix(table.family[r], predicted[r], labels=FAMILIES)
        for s, r in table.groupby("surface").groups.items()
    }
    family_of_surface = table.groupby("surface").family.first()
    rng = np.random.default_rng(42)
    draws = []
    for _ in range(n_draws):
        resampled = np.zeros((4, 4))
        for family in FAMILIES:
            surfaces = family_of_surface.index[family_of_surface == family]
            for surface in rng.choice(surfaces, size=len(surfaces)):
                resampled += confusions[surface]
        draws.append(mean_recall(resampled))
    total = sum(confusions.values())
    return {
        "balanced accuracy": mean_recall(total),
        "low": np.percentile(draws, 2.5),
        "high": np.percentile(draws, 97.5),
        **{f"recall {f}": total[i, i] / max(total[i].sum(), 1) for i, f in enumerate(FAMILIES)},
    }


def run_draws(features, table, new_setup, k_values, n_draws, method, seed_of):
    rows = []
    target = (table.dataset == new_setup).to_numpy()
    for k in k_values:
        for draw in range(n_draws if k else 1):
            rng = np.random.default_rng(seed_of(k, draw))
            labeled = (
                target & table.surface.isin(labeled_surfaces(table[target], k, rng)).to_numpy()
                if k
                else np.zeros(len(table), bool)
            )
            if method == "new setup only" and not k:
                continue
            test, predicted = calibrate_and_predict(features, table, new_setup, labeled, method)
            rows.append(
                {"new setup": new_setup, "method": method, "k": k, "draw": draw, **score(table[test], predicted)}
            )
    return rows


def select():
    selected, blocks = development_data()
    rows = []
    for candidate in CANDIDATES:
        features = features_of(blocks, candidate)
        for method in METHODS:
            for new_setup in DEV_SETUPS:
                seed_of = lambda k, draw, s=new_setup: [DEV_SETUPS.index(s), k, draw]  # noqa: E731
                rows += [
                    {"candidate": candidate, **r}
                    for r in run_draws(features, selected, new_setup, [0, 1, 2], 10, method, seed_of)
                ]
            print(candidate, method, "done", flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "development_scores.csv", index=False)
    calibrated = scores[scores.k > 0]
    ranking = calibrated.groupby(["candidate", "method"])["balanced accuracy"].mean().sort_values(ascending=False)
    print(
        scores.groupby(["candidate", "method", "new setup", "k"])["balanced accuracy"]
        .mean()
        .unstack("k")
        .round(2)
        .to_string()
    )
    print(ranking.round(3).to_string())
    best_candidate, best_method = ranking.index[0]
    choice = {
        "candidate": best_candidate,
        "method": best_method,
        "input": "acceleration",
        "development mean balanced accuracy, k = 1 and 2": round(float(ranking.iloc[0]), 4),
        "chosen on": "Cluster, LMT108, SENS3 only; HaTT not used",
    }
    (OUT / "frozen_choice.json").write_text(json.dumps(choice, indent=2))
    print("frozen:", choice)


def test_hatt():
    choice_path = OUT / "frozen_choice.json"
    if not choice_path.exists():
        raise SystemExit("run `select` first: the method must be frozen before the locked test")
    result_path = OUT / "hatt_locked_test.csv"
    if result_path.exists():
        raise SystemExit(f"the locked test was already run: {result_path}")
    choice = json.loads(choice_path.read_text())
    selected, dev_blocks = development_data()
    hatt, hatt_blocks = hatt_data()
    table = pd.concat(
        [selected[["dataset", "surface", "family"]], hatt[["dataset", "surface", "family"]]], ignore_index=True
    )
    features = np.concatenate(
        [features_of(dev_blocks, choice["candidate"]), features_of(hatt_blocks, choice["candidate"])]
    )
    rows = run_draws(features, table, "hatt", [0, 1, 2], 20, choice["method"], lambda k, draw: [99, k, draw])
    if choice["method"] == "new setup only":
        # no k = 0 for this method: the zero-label reference is the same encoder pooled over the known setups
        rows += [
            {**r, "method": "pooled (k = 0 reference)"}
            for r in run_draws(features, table, "hatt", [0], 1, "pooled", lambda k, draw: [99, k, draw])
        ]
    scores = pd.DataFrame(rows)
    scores.to_csv(result_path, index=False)
    print(json.dumps(choice))
    print(
        scores.groupby(["method", "k"])[["balanced accuracy", "low", "high", *[f"recall {f}" for f in FAMILIES]]]
        .mean()
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    {"select": select, "test_hatt": test_hatt}[sys.argv[1]]()

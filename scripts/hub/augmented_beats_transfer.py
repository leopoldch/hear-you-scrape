"""H6: does a calibrated "random setup" augmentation make BEATs transfer to a new setup?

See docs/experiments.md (H6) for the protocol fixed before running.
For each new setup (Cluster, LMT108, SENS3), BEATs is fine-tuned on 80% of the surfaces of the
other two setups, with one augmentation level, then tested on every window of the new setup
(probabilities averaged per recording). The other 20% of surfaces give a "known setups, new
surfaces" control. Augmentation levels: none, medium (±6 dB), strong (±12 dB).

Runs on the hub from the project folder:
    python scripts/hub/augmented_beats_transfer.py [--quick]
Output: analysis/classification/augmentation/augmented_transfer_scores.csv (+ predictions).
Finished runs are skipped on restart.
"""

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.metrics import confusion_matrix
from torch import nn

QUICK = "--quick" in sys.argv
FAMILIES = ["wood", "metal", "textile", "polymer"]
SETUPS = ["cluster", "lmt108", "sens3"]
AUGMENTATION_DB = {"none": 0.0, "medium": 6.0, "strong": 12.0}
INPUTS = {"audio": ["audio"], "audio + acceleration": ["audio", "accel"]}
N_UPDATES = 20 if QUICK else 1500
BATCH = 64
ENCODER_LR, HEAD_LR = 2e-5, 1e-3
WAVE_RMS = 0.1
RATE = 16000
SEED = 0

OUT = Path("analysis/classification/augmentation")
OUT.mkdir(parents=True, exist_ok=True)
SCORES = OUT / f"augmented_transfer_scores{'_quick' if QUICK else ''}.csv"
PREDICTIONS = OUT / f"augmented_transfer_predictions{'_quick' if QUICK else ''}.csv"
DEVICE = "cuda"


# ---------- data: one table of recordings, one array of windows per signal ----------


def load_data():
    main = pd.read_csv("analysis/signal_processing/recordings.csv", dtype={"surface": str})
    sens3 = pd.read_csv("analysis/classification/sens3_recordings.csv", dtype={"surface": str})
    sens3["dataset"] = "sens3"
    columns = ["dataset", "surface", "family"]
    recordings = pd.concat([main[columns], sens3[columns]], ignore_index=True)
    recordings["surface"] = recordings.dataset + ":" + recordings.surface

    audio, accel, recording_of_window = [], [], []
    for file, offset in [("waveforms.npz", 0), ("sens3_waveforms.npz", len(main))]:
        cache = np.load(f"analysis/classification/{file}")
        audio.append(torch.from_numpy(cache["audio"]))
        accel_8k = cache["accel"]
        resampled = torch.empty((len(accel_8k), 2 * accel_8k.shape[1]), dtype=torch.float16)
        for start in range(0, len(accel_8k), 4096):
            piece = torch.from_numpy(accel_8k[start : start + 4096].astype(np.float32))
            resampled[start : start + 4096] = torchaudio.functional.resample(piece, 8000, RATE).half()
        accel.append(resampled)
        recording_of_window.append(cache["recording"] + offset)
    waves = {"audio": torch.cat(audio), "accel": torch.cat(accel)}
    return recordings, waves, np.concatenate(recording_of_window)


# ---------- the "random setup" augmentation ----------


def random_gain_curve_db(freqs, strength_db, generator):
    """Smooth random gain over log-frequency plus 0 to 3 resonances, in dB."""
    log_f = torch.log(freqs.clamp(min=20.0))
    knots = torch.linspace(np.log(50.0), np.log(RATE / 2), 8, device=freqs.device)
    gains = (torch.rand(8, generator=generator, device=freqs.device) * 2 - 1) * strength_db
    position = ((log_f - knots[0]) / (knots[1] - knots[0])).clamp(0, 6.999)
    left = position.floor().long()
    curve = gains[left] + (position - left) * (gains[left + 1] - gains[left])
    n_peaks = int(torch.randint(0, 4, (1,), generator=generator, device=freqs.device))
    for _ in range(n_peaks):
        center = float(
            torch.exp(torch.empty(1, device=freqs.device).uniform_(np.log(100), np.log(6000), generator=generator))
        )
        quality = float(torch.empty(1, device=freqs.device).uniform_(3, 20, generator=generator))
        height = float(torch.rand(1, generator=generator, device=freqs.device)) * strength_db
        curve = curve + height / (1 + ((freqs - center) / (center / quality)) ** 2)
    return curve


def change_speed(x, factor):
    stretched = nn.functional.interpolate(x[None, None], scale_factor=1 / factor, mode="linear", align_corners=False)[
        0, 0
    ]
    if len(stretched) >= len(x):
        return stretched[: len(x)]
    return stretched.repeat(len(x) // len(stretched) + 1)[: len(x)]


def random_setup(batch, strength_db, generator):
    """Apply an independent virtual setup to every window, then restore unit RMS."""
    if strength_db == 0:
        return batch
    n = batch.shape[1]
    freqs = torch.fft.rfftfreq(n, 1 / RATE).to(batch.device)
    out = torch.empty_like(batch)
    snr_range = (20, 50) if strength_db <= 6 else (10, 40)
    for i, x in enumerate(batch):
        factor = float(
            torch.exp(torch.empty(1, device=batch.device).uniform_(np.log(0.8), np.log(1.25), generator=generator))
        )
        x = change_speed(x, factor)
        spectrum = torch.fft.rfft(x) * 10 ** (random_gain_curve_db(freqs, strength_db, generator) / 20)
        x = torch.fft.irfft(spectrum, n=n)
        snr_db = float(torch.empty(1, device=batch.device).uniform_(*snr_range, generator=generator))
        x = x + torch.randn(n, generator=generator, device=batch.device) * x.std() * 10 ** (-snr_db / 20)
        out[i] = x / (x.std() + 1e-8)
    return out


# ---------- model ----------


def load_beats():
    sys.path.insert(0, "models/beats")
    from BEATs import BEATs, BEATsConfig

    checkpoint = torch.load("models/beats/BEATs_iter3_plus_AS2M.pt", map_location="cpu", weights_only=True)
    model = BEATs(BEATsConfig(checkpoint["cfg"]))
    model.load_state_dict(checkpoint["model"])
    preprocess = model.preprocess

    def preprocess_in_float32(source, **kwargs):
        with torch.autocast("cuda", enabled=False):
            return preprocess(source.float(), **kwargs)

    model.preprocess = preprocess_in_float32
    return model


class FineTunedBEATs(nn.Module):
    def __init__(self, signal_names):
        super().__init__()
        self.signal_names = signal_names
        self.encoder = load_beats()
        width = self.encoder.cfg.encoder_embed_dim * len(signal_names)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(width, len(FAMILIES)))

    def forward(self, batch):
        embeddings = [
            self.encoder.extract_features(batch[name].float() * WAVE_RMS)[0].mean(dim=1) for name in self.signal_names
        ]
        return self.head(torch.cat(embeddings, dim=1))


def train(recordings, waves, recording_of_window, train_recordings, signal_names, strength_db):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    model = FineTunedBEATs(signal_names).to(DEVICE)
    optimizer = torch.optim.AdamW(
        [{"params": model.encoder.parameters(), "lr": ENCODER_LR}, {"params": model.head.parameters(), "lr": HEAD_LR}],
        weight_decay=0.01,
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[ENCODER_LR, HEAD_LR], total_steps=N_UPDATES, pct_start=0.1
    )

    table = recordings.loc[train_recordings]
    weights = 1 / table.groupby(["dataset", "family"]).family.transform("size").to_numpy()
    weights = weights / weights.sum()
    first_window = np.searchsorted(recording_of_window, train_recordings)
    n_windows = np.searchsorted(recording_of_window, train_recordings, side="right") - first_window
    family_index = table.family.map({f: k for k, f in enumerate(FAMILIES)}).to_numpy()

    model.train()
    for _ in range(N_UPDATES):
        chosen = rng.choice(len(train_recordings), BATCH, p=weights)
        windows = first_window[chosen] + rng.integers(0, n_windows[chosen])
        batch = {
            name: random_setup(waves[name][windows].to(DEVICE).float(), strength_db, generator) for name in signal_names
        }
        targets = torch.as_tensor(family_index[chosen], device=DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch).float()
        loss = nn.functional.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
    return model


@torch.no_grad()
def classify(model, waves, recording_of_window, test_recordings):
    model.eval()
    windows = np.flatnonzero(np.isin(recording_of_window, test_recordings))
    probabilities = []
    for start in range(0, len(windows), 256):
        chosen = windows[start : start + 256]
        batch = {name: waves[name][chosen].to(DEVICE) for name in model.signal_names}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            probabilities.append(model(batch).float().softmax(dim=1).cpu())
    probabilities = pd.DataFrame(torch.cat(probabilities).numpy(), columns=FAMILIES)
    return probabilities.groupby(recording_of_window[windows]).mean().idxmax(axis=1)


# ---------- scoring ----------


def mean_recall(confusion):
    return np.mean(np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1))


def score(table, predicted, n_draws=1000):
    table = table.reset_index(drop=True)
    predicted = np.asarray(predicted)
    confusions = {
        surface: confusion_matrix(table.family[rows], predicted[rows], labels=FAMILIES)
        for surface, rows in table.groupby("surface").groups.items()
    }
    family_of_surface = table.groupby("surface").family.first()
    rng = np.random.default_rng(42)
    draws = []
    for _ in range(n_draws):
        resampled = np.zeros((4, 4))
        for family in FAMILIES:
            surfaces = family_of_surface.index[family_of_surface == family]
            if len(surfaces):
                for surface in rng.choice(surfaces, size=len(surfaces)):
                    resampled += confusions[surface]
        draws.append(mean_recall(resampled))
    return {
        "balanced accuracy": mean_recall(sum(confusions.values())),
        "low": np.percentile(draws, 2.5),
        "high": np.percentile(draws, 97.5),
        "most frequent answer share": pd.Series(predicted).value_counts(normalize=True).iloc[0],
    }


def control_surfaces(recordings, setups):
    """20% of the surfaces of each training setup, per family, kept out as a control (seed 0)."""
    rng = np.random.default_rng(SEED)
    held = []
    for (_dataset, _family), group in recordings[recordings.dataset.isin(setups)].groupby(["dataset", "family"]):
        surfaces = group.surface.unique()
        held += list(rng.choice(surfaces, max(1, round(0.2 * len(surfaces))), replace=False))
    return set(held)


def main():
    recordings, waves, recording_of_window = load_data()
    if QUICK:
        keep = recordings.groupby("surface").head(3).index.to_numpy()
        keep_windows = np.isin(recording_of_window, keep)
        waves = {name: values[keep_windows] for name, values in waves.items()}
        recording_of_window = recording_of_window[keep_windows]
    present = np.unique(recording_of_window)
    print(f"{len(present)} recordings, {len(recording_of_window)} windows", flush=True)

    done = set()
    if SCORES.exists():
        done = set(pd.read_csv(SCORES)[["new setup", "input", "augmentation"]].itertuples(index=False, name=None))

    for new_setup in SETUPS:
        training_setups = [s for s in SETUPS if s != new_setup]
        held = control_surfaces(recordings, training_setups)
        in_training_setups = recordings.dataset.isin(training_setups) & recordings.index.isin(present)
        train_recordings = recordings.index[in_training_setups & ~recordings.surface.isin(held)].to_numpy()
        control_recordings = recordings.index[in_training_setups & recordings.surface.isin(held)].to_numpy()
        test_recordings = recordings.index[
            (recordings.dataset == new_setup) & recordings.index.isin(present)
        ].to_numpy()
        for input_name, signal_names in INPUTS.items():
            for augmentation, strength_db in AUGMENTATION_DB.items():
                if (new_setup, input_name, augmentation) in done:
                    continue
                started = time.time()
                model = train(recordings, waves, recording_of_window, train_recordings, signal_names, strength_db)
                rows, prediction_rows = [], []
                for test_name, tested in [
                    (f"new setup: {new_setup}", test_recordings),
                    ("known setups, new surfaces", control_recordings),
                ]:
                    predicted = classify(model, waves, recording_of_window, tested)
                    rows.append(
                        {
                            "new setup": new_setup,
                            "input": input_name,
                            "augmentation": augmentation,
                            "test": test_name,
                            **score(
                                recordings.loc[predicted.index], predicted.to_numpy(), n_draws=100 if QUICK else 1000
                            ),
                        }
                    )
                    prediction_rows += [
                        {
                            "new setup": new_setup,
                            "input": input_name,
                            "augmentation": augmentation,
                            "test": test_name,
                            "recording": r,
                            "predicted": p,
                        }
                        for r, p in predicted.items()
                    ]
                pd.DataFrame(rows).to_csv(SCORES, mode="a", header=not SCORES.exists(), index=False)
                pd.DataFrame(prediction_rows).to_csv(
                    PREDICTIONS, mode="a", header=not PREDICTIONS.exists(), index=False
                )
                print(
                    f"{new_setup}, {input_name}, {augmentation}: "
                    + ", ".join(f"{r['test']} {r['balanced accuracy']:.2f}" for r in rows)
                    + f" ({time.time() - started:.0f} s)",
                    flush=True,
                )
                del model
                gc.collect()
                torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

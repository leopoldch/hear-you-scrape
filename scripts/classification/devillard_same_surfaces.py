"""H30: when the surfaces are the same, does transfer without labels work? (Devillard et al. 2025)

Ten people slide a bare finger on the same ten textures; two microphones and two accelerometers (nail, phalanx)
record at the same time. The task is to name the texture (10 classes, chance 0.10) with frozen BEATs and a
logistic probe:
  1. known people, new trial;
  2. new person, no label of that person;
  3. new sensor, no label of that sensor (nail -> phalanx, left mic -> right mic, acceleration -> audio);
  4. cosine between texture directions across people and across sensors (as in H21).
Predictions and reading: docs/experiments.md, H30.

Run:  uv run --with h5py python scripts/classification/devillard_same_surfaces.py
Output: analysis/classification/devillard_same_surfaces/{embeddings.npz,scores.csv,direction_cosines.csv}
"""

import io
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "data/devillard/datah5.zip"
OUT = ROOT / "analysis/classification/devillard_same_surfaces"
SENSORS = ["left microphone", "right microphone", "nail accelerometer", "phalanx accelerometer"]
RATE = 16000
CHUNK_SECONDS = 4
CHUNKS_PER_TRIAL = 4
WINDOW = RATE // 2
WAVE_RMS = 0.1


def load_beats():
    sys.path.insert(0, str(ROOT / "models/beats"))
    from BEATs import BEATs, BEATsConfig

    checkpoint = torch.load(ROOT / "models/beats/BEATs_iter3_plus_AS2M.pt", map_location="cpu", weights_only=True)
    model = BEATs(BEATsConfig(checkpoint["cfg"]))
    model.load_state_dict(checkpoint["model"])
    return model.eval()


@torch.no_grad()
def embed(model, chunk):
    """Same recipe as the other setups: 0.5 s windows at a fixed level, then the mean."""
    windows = chunk.reshape(-1, WINDOW)
    windows = windows / (np.sqrt((windows**2).mean(axis=1, keepdims=True)) + 1e-12) * WAVE_RMS
    features = model.extract_features(torch.tensor(windows, dtype=torch.float32))[0].mean(dim=1)
    return features.mean(dim=0).numpy()


def compute_embeddings():
    model = load_beats()
    archive = zipfile.ZipFile(ARCHIVE)
    names = sorted(name for name in archive.namelist() if name.endswith(".h5"))
    rows = []
    embeddings = []
    for number, name in enumerate(names):
        _, person, texture, file_name = name.split("/")
        with h5py.File(io.BytesIO(archive.read(name)), "r") as trial:
            signals = trial["kistler_data"][:]
        signals = resample_poly(signals, 32, 25, axis=0)
        chunk_length = CHUNK_SECONDS * RATE
        starts = np.linspace(2 * RATE, len(signals) - chunk_length - 2 * RATE, CHUNKS_PER_TRIAL).astype(int)
        if starts[0] < 0 or starts[-1] <= starts[0]:
            continue
        for start in starts:
            for sensor_number, sensor in enumerate(SENSORS):
                embeddings.append(embed(model, signals[start : start + chunk_length, sensor_number]))
                rows.append({"person": person, "texture": texture, "trial": file_name[:-3], "sensor": sensor})
        if number % 25 == 0:
            print("trial", number, "of", len(names), flush=True)
    table = pd.DataFrame(rows)
    np.savez(OUT / "embeddings.npz", BEATs=np.stack(embeddings), **{c: table[c].to_numpy() for c in table.columns})


def standardize_per_group(features, groups):
    out = np.empty_like(features)
    for group in np.unique(groups):
        rows = groups == group
        out[rows] = (features[rows] - features[rows].mean(axis=0)) / (features[rows].std(axis=0) + 1e-6)
    return out


def probe_score(features, table, train, test):
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    probe.fit(features[train], table.texture[train])
    return balanced_accuracy_score(table.texture[test], probe.predict(features[test]))


def texture_directions(features, table, rows):
    textures = sorted(table.texture.unique())
    return np.stack([features[rows & (table.texture == texture).to_numpy()].mean(axis=0) for texture in textures])


def mean_cosine(a, b):
    cosines = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return cosines.mean()


OUT.mkdir(parents=True, exist_ok=True)
if not (OUT / "embeddings.npz").exists():
    compute_embeddings()
cache = np.load(OUT / "embeddings.npz", allow_pickle=True)
table = pd.DataFrame({column: cache[column] for column in ["person", "texture", "trial", "sensor"]})
group = (table.person + " " + table.sensor).to_numpy()
features = standardize_per_group(cache["BEATs"], group)
people = sorted(table.person.unique())
is_sensor = {sensor: (table.sensor == sensor).to_numpy() for sensor in SENSORS}
is_last_trial = table.trial.str.contains("std_2").to_numpy()

scores = []
for sensor in SENSORS:
    rows = is_sensor[sensor]
    known = probe_score(features, table, rows & ~is_last_trial, rows & is_last_trial)
    scores.append({"test": "known people, new trial", "trained on": sensor, "tested on": sensor, "score": known})
    new_person = []
    for person in people:
        is_person = (table.person == person).to_numpy()
        new_person.append(probe_score(features, table, rows & ~is_person, rows & is_person))
    scores.append(
        {"test": "new person, k = 0", "trained on": sensor, "tested on": sensor, "score": np.mean(new_person)}
    )

sensor_pairs = [
    ("nail accelerometer", "phalanx accelerometer"),
    ("phalanx accelerometer", "nail accelerometer"),
    ("left microphone", "right microphone"),
    ("right microphone", "left microphone"),
    ("nail accelerometer", "left microphone"),
    ("left microphone", "nail accelerometer"),
]
for source, target in sensor_pairs:
    same_people = probe_score(features, table, is_sensor[source], is_sensor[target])
    scores.append(
        {"test": "new sensor, k = 0, same people", "trained on": source, "tested on": target, "score": same_people}
    )
    both_new = []
    for person in people:
        is_person = (table.person == person).to_numpy()
        both_new.append(probe_score(features, table, is_sensor[source] & ~is_person, is_sensor[target] & is_person))
    scores.append(
        {
            "test": "new sensor and new person, k = 0",
            "trained on": source,
            "tested on": target,
            "score": np.mean(both_new),
        }
    )
scores = pd.DataFrame(scores)
scores.to_csv(OUT / "scores.csv", index=False)
print(scores.round(3).to_string(), flush=True)

cosines = []
for sensor in SENSORS:
    directions = [
        texture_directions(features, table, is_sensor[sensor] & (table.person == p).to_numpy()) for p in people
    ]
    across = [mean_cosine(directions[i], directions[j]) for i in range(len(people)) for j in range(i + 1, len(people))]
    cosines.append({"comparison": "across people", "sensor": sensor, "cosine": np.mean(across)})
    halves = []
    for person in people:
        rows = is_sensor[sensor] & (table.person == person).to_numpy()
        halves.append(
            mean_cosine(
                texture_directions(features, table, rows & is_last_trial),
                texture_directions(features, table, rows & ~is_last_trial),
            )
        )
    cosines.append({"comparison": "same person, different trials", "sensor": sensor, "cosine": np.mean(halves)})
for source, target in sensor_pairs[::2]:
    across = mean_cosine(
        texture_directions(features, table, is_sensor[source]), texture_directions(features, table, is_sensor[target])
    )
    cosines.append({"comparison": "across sensors", "sensor": source + " vs " + target, "cosine": across})
cosines = pd.DataFrame(cosines)
cosines.to_csv(OUT / "direction_cosines.csv", index=False)
print(cosines.round(3).to_string(), flush=True)

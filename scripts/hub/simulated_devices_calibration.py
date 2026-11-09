"""H28, step 2: what do 250 simulated devices teach about a new device?

(a) Scale curve inside the simulation: probe trained on N simulated devices, tested with no label on 50 unseen ones.
(b) k = 0 on the real setups: probe trained on all simulated devices.
(c) Learned calibration: a linear projection trained by episodes on simulated devices, so that k labeled surfaces
    per family are enough (nearest prototype). Applied as is to the real setups, against the H16 logistic probe and
    prototypes without projection, on the same draws.
Embeddings are standardized inside each device (simulated or real), which needs no label.
Predictions and criteria: docs/experiments.md, H28.

Run on the hub from the project folder, after simulated_devices.py:
    python scripts/hub/simulated_devices_calibration.py
Output: analysis/classification/simulated_devices/{scale_curve,real_zero_shot,real_few_shot}.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

ROOT = Path.home() / "ift7030-lacha"
sys.path.insert(0, str(ROOT / "scripts/hub"))
import setup_transfer_probes as stp  # noqa: E402

OUT = ROOT / "analysis/classification/simulated_devices"
FAMILIES = stp.FAMILIES
N_TEST_DEVICES = 50
N_DRAWS = 10
PROJECTION_SIZE = 64


def standardize_per_group(features, groups):
    out = np.empty_like(features)
    for group in np.unique(groups):
        rows = groups == group
        out[rows] = (features[rows] - features[rows].mean(axis=0)) / (features[rows].std(axis=0) + 1e-6)
    return out


def nearest_prototype(train_features, train_families, test_features):
    prototypes = np.stack([train_features[train_families == family].mean(axis=0) for family in FAMILIES])
    distances = ((test_features[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
    return np.array(FAMILIES)[distances.argmin(axis=1)]


def train_projection(features, families, devices, surfaces):
    """Episodes: one simulated device, k labeled surfaces per family, classify its other clips by nearest prototype."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    projection = torch.nn.Linear(features.shape[1], PROJECTION_SIZE).cuda()
    optimizer = torch.optim.Adam(projection.parameters(), lr=1e-3, weight_decay=1e-4)
    features = torch.tensor(features, dtype=torch.float32).cuda()
    family_index = torch.tensor([FAMILIES.index(family) for family in families]).cuda()
    surface_table = pd.DataFrame({"device": devices, "family": families, "surface": surfaces}).drop_duplicates(
        "surface"
    )
    for episode in range(4000):
        device = rng.choice(np.unique(devices))
        k = rng.choice([1, 2, 4])
        of_device = surface_table[surface_table.device == device]
        labeled = np.concatenate(
            [rng.choice(of_device.surface[of_device.family == f], k, replace=False) for f in FAMILIES]
        )
        in_device = devices == device
        support = torch.tensor(in_device & np.isin(surfaces, labeled)).cuda()
        query = torch.tensor(in_device & ~np.isin(surfaces, labeled)).cuda()
        projected = projection(features)
        prototypes = torch.stack([projected[support & (family_index == i)].mean(dim=0) for i in range(4)])
        logits = -(torch.cdist(projected[query], prototypes) ** 2) / PROJECTION_SIZE
        loss = torch.nn.functional.cross_entropy(logits, family_index[query])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if episode % 500 == 0:
            print("episode", episode, "loss", round(loss.item(), 3), flush=True)
    weight = projection.weight.detach().cpu().numpy()
    return lambda x: x @ weight.T


def few_shot_scores(features, table, project, k_values, rng):
    """Balanced accuracy on the unlabeled surfaces of one device, for each method, averaged over the draws."""
    projected = project(features)
    rows = []
    for k in k_values:
        for draw in range(N_DRAWS):
            labeled = stp.labeled_surfaces(table, k, rng)
            train = table.surface.isin(labeled).to_numpy()
            test = ~train
            truth = table.family[test]
            predictions = {
                "logistic probe (H16)": stp.fit_and_predict(features, table, train, test),
                "prototypes": nearest_prototype(features[train], table.family[train].to_numpy(), features[test]),
                "prototypes, learned projection": nearest_prototype(
                    projected[train], table.family[train].to_numpy(), projected[test]
                ),
            }
            for method, predicted in predictions.items():
                rows.append(
                    {"k": k, "draw": draw, "method": method, "score": balanced_accuracy_score(truth, predicted)}
                )
    return pd.DataFrame(rows).groupby(["k", "method"], as_index=False).score.mean()


simulated = np.load(OUT / "embeddings.npz", allow_pickle=True)
sim_features = standardize_per_group(simulated["BEATs"], simulated["device"])
sim_family, sim_device, sim_surface = simulated["family"], simulated["device"], simulated["surface"]
device_ids = np.unique(sim_device)
test_devices = device_ids[-N_TEST_DEVICES:]
train_devices = device_ids[:-N_TEST_DEVICES]
in_test = np.isin(sim_device, test_devices)

# (a) scale curve inside the simulation
rng = np.random.default_rng(0)
curve = []
for n_devices in [1, 2, 3, 5, 10, 20, 50, 100, 200]:
    for draw in range(5):
        chosen = rng.choice(train_devices, n_devices, replace=False)
        rows = np.isin(sim_device, chosen)
        probe = LogisticRegression(C=0.1, max_iter=3000).fit(sim_features[rows], sim_family[rows])
        score = balanced_accuracy_score(sim_family[in_test], probe.predict(sim_features[in_test]))
        curve.append({"n_devices": n_devices, "draw": draw, "score": score})
curve = pd.DataFrame(curve)
curve.to_csv(OUT / "scale_curve.csv", index=False)
print(curve.groupby("n_devices").score.agg(["mean", "min", "max"]).round(3), flush=True)

# real recordings
selected = pd.read_csv(stp.OUT / "selected_recordings.csv", dtype={"surface": str})
selected["surface"] = selected.dataset + ":" + selected.surface
real = stp.load_beats_clap(selected)["BEATs"]

# (b) k = 0 on the real setups
probe = LogisticRegression(C=0.1, max_iter=3000).fit(sim_features, sim_family)
zero_shot = []
for signal in ["audio", "accel"]:
    real_features = standardize_per_group(real[signal], selected.dataset.to_numpy())
    for setup in stp.SETUPS:
        rows = (selected.dataset == setup).to_numpy()
        result = stp.score(selected[rows], probe.predict(real_features[rows]))
        zero_shot.append(dict(signal=signal, setup=setup, **result))
zero_shot = pd.DataFrame(zero_shot)
zero_shot.to_csv(OUT / "real_zero_shot.csv", index=False)
print(zero_shot.round(3).to_string(), flush=True)

# (c) learned calibration
in_train = ~in_test
project = train_projection(sim_features[in_train], sim_family[in_train], sim_device[in_train], sim_surface[in_train])

rng = np.random.default_rng(1)
few_shot = []
for device in test_devices[:20]:
    rows = sim_device == device
    table = pd.DataFrame({"surface": sim_surface[rows], "family": sim_family[rows], "dataset": "simulated"})
    scores = few_shot_scores(sim_features[rows], table, project, [1, 2, 4], rng)
    few_shot.append(scores.assign(setup="simulated (unseen devices)", signal="simulated"))
for signal in ["audio", "accel"]:
    real_features = standardize_per_group(real[signal], selected.dataset.to_numpy())
    for setup in stp.SETUPS:
        rows = (selected.dataset == setup).to_numpy()
        table = selected[rows].reset_index(drop=True)
        scores = few_shot_scores(real_features[rows], table, project, [1, 2, 4], rng)
        few_shot.append(scores.assign(setup=setup, signal=signal))
        print(signal, setup, "done", flush=True)
few_shot = pd.concat(few_shot).groupby(["signal", "setup", "k", "method"], as_index=False).score.mean()
few_shot.to_csv(OUT / "real_few_shot.csv", index=False)
print(
    few_shot.pivot_table(index=["signal", "setup", "method"], columns="k", values="score").round(3).to_string(),
    flush=True,
)

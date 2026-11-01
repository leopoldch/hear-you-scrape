import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.functional import one_hot

from evaluation import FAMILIES, OUT, recording_table

N_UPDATES = 2000
BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPECTROGRAMS_OF_INPUT = {"audio": ["audio"], "acceleration": ["accel"], "audio + acceleration": ["audio", "accel"]}
BASELINE = {
    "deeper": False,
    "dropout": 0.3,
    "weight_decay": 1e-4,
    "augment": False,
    "normalize_each_window": False,
    "noise": False,
    "mixup": False,
    "adversarial": False,
    "adapt_batchnorm": False,
}
DATASET_IDS = {"cluster": 0, "lmt108": 1, "sens3": 2, "greatest_hits": 3}
NATS_PER_DB = np.log(10) / 10


def spectrograms_on_gpu(cache):
    return {name: torch.as_tensor(cache[name], device=DEVICE) for name in ["audio", "accel"] if name in cache}


def load_windows():
    recordings = recording_table()
    cache = np.load(OUT / "spectrograms.npz")
    recording_of_window = cache["recording"]
    family_of_window, group_of_window = window_labels(recordings, recording_of_window)
    return recordings, spectrograms_on_gpu(cache), recording_of_window, family_of_window, group_of_window


def window_labels(recordings, recording_of_window):
    family = recordings.family.map({name: k for k, name in enumerate(FAMILIES)}).to_numpy()
    dataset = recordings.dataset.map(DATASET_IDS).to_numpy()
    group = dataset * len(FAMILIES) + family
    family_of_window = torch.as_tensor(family[recording_of_window], device=DEVICE)
    group_of_window = torch.as_tensor(group[recording_of_window], device=DEVICE)
    return family_of_window, group_of_window


def load_windows_of(dataset):
    recordings = pd.read_csv(OUT / f"{dataset}_recordings.csv", dtype={"surface": str})
    cache = np.load(OUT / f"{dataset}_spectrograms.npz")
    return recordings, spectrograms_on_gpu(cache), cache["recording"]


def conv_block(in_channels, out_channels, n_convolutions):
    layers = []
    for k in range(n_convolutions):
        layers += [
            nn.Conv2d(in_channels if k == 0 else out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
    return nn.Sequential(*layers, nn.MaxPool2d(2))


def branch(channels, n_convolutions):
    blocks = [conv_block(a, b, n_convolutions) for a, b in zip([1, *channels[:-1]], channels, strict=True)]
    return nn.Sequential(nn.BatchNorm2d(1), *blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten())


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(context, x, strength):
        context.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(context, gradient):
        return -context.strength * gradient, None


class SmallCNN(nn.Module):
    def __init__(self, spectrogram_names, variant):
        super().__init__()
        channels, n_convolutions = ([32, 64, 128, 128], 2) if variant["deeper"] else ([16, 32, 64, 64], 1)
        self.spectrogram_names = spectrogram_names
        self.normalize_each_window = variant["normalize_each_window"]
        self.branches = nn.ModuleDict({name: branch(channels, n_convolutions) for name in spectrogram_names})
        width = channels[-1] * len(spectrogram_names)
        self.classifier = nn.Sequential(nn.Dropout(variant["dropout"]), nn.Linear(width, len(FAMILIES)))
        if variant["adversarial"]:
            self.dataset_head = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, len(DATASET_IDS)))

    def embed(self, spectrograms):
        if self.normalize_each_window:
            spectrograms = {name: x - x.mean(dim=-1, keepdim=True) for name, x in spectrograms.items()}
        return torch.cat([self.branches[name](spectrograms[name]) for name in self.spectrogram_names], dim=1)

    def forward(self, spectrograms):
        return self.classifier(self.embed(spectrograms))


def batch(spectrograms, windows):
    return {name: values[windows].float().unsqueeze(1) for name, values in spectrograms.items()}


def random_spectral_curves_db(n, n_bins):
    position = torch.linspace(0, 1, n_bins, device=DEVICE)
    tilt = (2 * torch.rand(n, 1, device=DEVICE) - 1) * 10 * (position - 0.5)
    equalizer = 0
    for k in [1, 2, 3]:
        phase = 2 * np.pi * torch.rand(n, 1, device=DEVICE)
        equalizer = equalizer + 2 * torch.randn(n, 1, device=DEVICE) * torch.cos(np.pi * k * position + phase)
    return tilt + equalizer


def remove_mean_level(x):
    return x - x.mean(dim=(2, 3), keepdim=True)


def augment(spectrograms):
    augmented = {}
    for name, x in spectrograms.items():
        curves = random_spectral_curves_db(len(x), x.shape[2]) * NATS_PER_DB
        augmented[name] = remove_mean_level(x + curves[:, None, :, None])
    return augmented


def add_noise(spectrograms):
    noisy = {}
    for name, x in spectrograms.items():
        n, _, n_bins, n_times = x.shape
        level = -30 * torch.rand(n, 1, device=DEVICE)
        noise = ((random_spectral_curves_db(n, n_bins) + level) * NATS_PER_DB)[:, None, :, None]
        uniform = torch.rand(n, 1, n_bins, n_times, device=DEVICE).clamp(1e-7, 1 - 1e-7)
        fluctuation = torch.log(-torch.log(uniform))
        noisy[name] = remove_mean_level(torch.logaddexp(x, noise + fluctuation))
    return noisy


def mix(spectrograms):
    n = len(next(iter(spectrograms.values())))
    partner = torch.randperm(n, device=DEVICE)
    p = torch.rand(n, device=DEVICE).clamp(1e-3, 1 - 1e-3)
    log_p = torch.log(p)[:, None, None, None]
    log_q = torch.log(1 - p)[:, None, None, None]
    mixed = {name: torch.logaddexp(log_p + x, log_q + x[partner]) for name, x in spectrograms.items()}
    mixed = {name: remove_mean_level(x) for name, x in mixed.items()}
    return mixed, partner, p


def cross_entropy(logits, target_probabilities, weights):
    losses = -(target_probabilities * logits.log_softmax(dim=1)).sum(dim=1)
    return (losses * weights).sum() / weights.sum()


def train(input_name, spectrograms, family_of_window, group_of_window, windows, seed, variant=BASELINE, target=None):
    torch.manual_seed(seed)
    model = SmallCNN(SPECTROGRAMS_OF_INPUT[input_name], variant).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=variant["weight_decay"])

    windows = torch.as_tensor(windows, device=DEVICE)
    windows_per_group = torch.bincount(group_of_window[windows], minlength=len(DATASET_IDS) * len(FAMILIES))
    weight_of_window = 1 / windows_per_group[group_of_window].float()
    if target is not None:
        target_spectrograms, target_windows, target_dataset = target
        target_windows = torch.as_tensor(target_windows, device=DEVICE)
        target_dataset_ids = torch.full((BATCH_SIZE,), DATASET_IDS[target_dataset], device=DEVICE)
        target_datasets = one_hot(target_dataset_ids, len(DATASET_IDS)).float()

    model.train()
    for update in range(N_UPDATES):
        chosen = windows[torch.randint(len(windows), (BATCH_SIZE,), device=DEVICE)]
        inputs = batch(spectrograms, chosen)
        families = one_hot(family_of_window[chosen], len(FAMILIES)).float()
        datasets = one_hot(group_of_window[chosen] // len(FAMILIES), len(DATASET_IDS)).float()
        weights = weight_of_window[chosen]

        if variant["augment"]:
            inputs = augment(inputs)
        if variant["noise"]:
            inputs = add_noise(inputs)
        if variant["mixup"]:
            inputs, partner, p = mix(inputs)
            families = p[:, None] * families + (1 - p[:, None]) * families[partner]
            datasets = p[:, None] * datasets + (1 - p[:, None]) * datasets[partner]
            weights = p * weights + (1 - p) * weights[partner]

        if variant["adversarial"]:
            unlabeled_windows = target_windows[torch.randint(len(target_windows), (BATCH_SIZE,), device=DEVICE)]
            unlabeled = batch(target_spectrograms, unlabeled_windows)
            embedding = model.embed({name: torch.cat([inputs[name], unlabeled[name]]) for name in inputs})
            strength = 2 / (1 + np.exp(-10 * update / N_UPDATES)) - 1
            dataset_logits = model.dataset_head(GradientReversal.apply(embedding, strength))
            loss = (
                cross_entropy(model.classifier(embedding[:BATCH_SIZE]), families, weights)
                + 0.5 * cross_entropy(dataset_logits[:BATCH_SIZE], datasets, weights)
                + 0.5 * cross_entropy(dataset_logits[BATCH_SIZE:], target_datasets, torch.ones_like(weights))
            )
        else:
            loss = cross_entropy(model(inputs), families, weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if variant["adapt_batchnorm"]:
        adapt_batchnorm(model, target_spectrograms, target_windows)
    return model


@torch.no_grad()
def adapt_batchnorm(model, spectrograms, windows):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = None
    model.train()
    for start in range(0, len(windows), 1024):
        model(batch(spectrograms, windows[start : start + 1024]))


@torch.no_grad()
def classify_recordings(model, spectrograms, windows, recording_of_window):
    model.eval()
    probabilities = []
    for start in range(0, len(windows), 1024):
        chosen = torch.as_tensor(windows[start : start + 1024], device=DEVICE)
        probabilities.append(model(batch(spectrograms, chosen)).softmax(dim=1).cpu())
    probabilities = pd.DataFrame(torch.cat(probabilities).numpy(), columns=FAMILIES)
    return probabilities.groupby(recording_of_window[windows]).mean().idxmax(axis=1)

"""H28, step 1: simulate many virtual devices scraping virtual surfaces, and embed the clips with BEATs.

A clip is made in four steps (source-filter model, after Agarwal et al. 2021 and night experiment N26):
  1. the tip reads a rough surface profile at the sliding speed (fractal roughness + periodic texture),
     smoothed by the tip size;
  2. this force excites the surface (damped modes, damping set by the material family and by the contact)
     and the probe (its own modes, the same for every surface of the device);
  3. the sensor filters the result (band limits and a random smooth response);
  4. motor lines proportional to the speed and sensor noise are added.
The family physics is the one fixed in N26. Everything about the device is drawn at random once per device.

Run on the hub from the project folder:  python scripts/hub/simulated_devices.py
Output: analysis/classification/simulated_devices/embeddings.npz (one BEATs vector per clip).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, fftconvolve, sosfilt

ROOT = Path.home() / "ift7030-lacha"
OUT = ROOT / "analysis/classification/simulated_devices"
FAMILIES = ["wood", "metal", "textile", "polymer"]
RATE = 16000
CLIP_SECONDS = 2.0
N_DEVICES = 250
SURFACES_PER_FAMILY = 6
CLIPS_PER_SURFACE = 2
WINDOW = RATE // 2
WAVE_RMS = 0.1

FAMILY_PHYSICS = {
    "metal": {
        "loss": (1e-4, 1e-3),
        "modes_hz": (500, 8000),
        "n_modes": (40, 80),
        "roughness": (2.5, 3.0),
        "period_mm": None,
    },
    "wood": {
        "loss": (0.01, 0.03),
        "modes_hz": (200, 3000),
        "n_modes": (15, 30),
        "roughness": (2.0, 2.5),
        "period_mm": (0.5, 2.0),
    },
    "polymer": {
        "loss": (0.03, 0.1),
        "modes_hz": (300, 4000),
        "n_modes": (10, 25),
        "roughness": (2.5, 3.2),
        "period_mm": None,
    },
    "textile": {
        "loss": (0.2, 0.5),
        "modes_hz": (100, 2000),
        "n_modes": (5, 10),
        "roughness": (1.5, 2.0),
        "period_mm": (0.3, 1.5),
    },
}


def log_uniform(rng, low, high, size=None):
    return np.exp(rng.uniform(np.log(low), np.log(high), size))


def modal_response(frequencies, losses, gains, seconds=0.25):
    """Impulse response of a set of damped modes. A mode at f with loss factor eta decays as exp(-pi eta f t)."""
    time = np.arange(int(seconds * RATE)) / RATE
    decay = np.exp(-np.pi * losses[:, None] * frequencies[:, None] * time)
    return (gains[:, None] * decay * np.sin(2 * np.pi * frequencies[:, None] * time)).sum(axis=0)


def random_device(rng):
    spectrum_hz = np.fft.rfftfreq(int(CLIP_SECONDS * RATE), 1 / RATE)
    low_pass_hz = log_uniform(rng, 1000, 8000)
    high_pass_hz = log_uniform(rng, 5, 200)
    knots_hz = np.geomspace(20, 8000, 7)
    knots_db = rng.uniform(-10, 10, len(knots_hz))
    smooth_db = np.interp(np.log(spectrum_hz + 1), np.log(knots_hz), knots_db)
    band = 1 / np.sqrt(1 + (spectrum_hz / low_pass_hz) ** 4) / np.sqrt(1 + (high_pass_hz / (spectrum_hz + 1e-3)) ** 4)
    n_probe_modes = int(rng.integers(2, 9))
    probe_hz = log_uniform(rng, 100, 6000, n_probe_modes)
    probe_loss = log_uniform(rng, 0.005, 0.1, n_probe_modes)
    has_soft_contact = rng.random() < 0.4
    has_motor = rng.random() < 0.4
    return {
        "tip_radius_mm": log_uniform(rng, 0.1, 5),
        "speed_mm_s": log_uniform(rng, 10, 200),
        "force": log_uniform(rng, 0.2, 5),
        "contact_loss": log_uniform(rng, 0.01, 0.3) if has_soft_contact else 0.0,
        "probe_response": modal_response(probe_hz, probe_loss, rng.normal(size=n_probe_modes)),
        "probe_gain": log_uniform(rng, 0.1, 10),
        "sensor_response": band * 10 ** (smooth_db / 20),
        "motor_hz_per_mm_s": log_uniform(rng, 0.5, 5) if has_motor else 0.0,
        "motor_level": log_uniform(rng, 0.1, 2),
        "noise_level": log_uniform(rng, 0.01, 0.3),
    }


def random_surface(rng, family, device):
    physics = FAMILY_PHYSICS[family]
    n_modes = int(rng.integers(*physics["n_modes"]))
    mode_hz = log_uniform(rng, physics["modes_hz"][0], min(physics["modes_hz"][1], 0.45 * RATE), n_modes)
    loss = log_uniform(rng, *physics["loss"]) + device["contact_loss"]
    response = modal_response(mode_hz, np.full(n_modes, loss), rng.normal(size=n_modes))
    response[0] += rng.uniform(0.2, 1.0) * np.abs(response).max()
    period_mm = None
    if physics["period_mm"] is not None:
        period_mm = rng.uniform(*physics["period_mm"])
    return {"response": response, "roughness": rng.uniform(*physics["roughness"]), "period_mm": period_mm}


def surface_profile(rng, n_samples, step_m, roughness, period_mm):
    """Surface height every step_m metres: spectrum in k^-roughness, plus a periodic texture."""
    wavenumber = np.fft.rfftfreq(n_samples, step_m)
    amplitude = np.zeros_like(wavenumber)
    amplitude[1:] = wavenumber[1:] ** (-roughness / 2)
    height = np.fft.irfft(amplitude * np.exp(2j * np.pi * rng.random(len(wavenumber))), n_samples)
    height /= height.std() + 1e-12
    if period_mm is not None:
        position_m = np.arange(n_samples) * step_m
        wave = np.sin(2 * np.pi * position_m / (period_mm * 1e-3))
        height += rng.uniform(0.5, 1.5) * np.sign(wave) * np.sqrt(np.abs(wave))
    return height


def simulated_clip(rng, device, surface):
    n_samples = int(CLIP_SECONDS * RATE)
    time = np.arange(n_samples) / RATE
    speed_mm_s = device["speed_mm_s"] * np.exp(rng.normal(0, 0.3))
    force = device["force"] * np.exp(rng.normal(0, 0.3))

    height = surface_profile(rng, n_samples, speed_mm_s * 1e-3 / RATE, surface["roughness"], surface["period_mm"])
    excitation = np.gradient(height) * force
    tip_cutoff_hz = min(0.45 * RATE, 10 * speed_mm_s / device["tip_radius_mm"])
    excitation = sosfilt(butter(2, tip_cutoff_hz, fs=RATE, output="sos"), excitation)

    from_surface = fftconvolve(excitation, surface["response"])[:n_samples]
    from_probe = fftconvolve(excitation, device["probe_response"])[:n_samples]
    vibration = from_surface / (from_surface.std() + 1e-12) + device["probe_gain"] * from_probe / (
        from_probe.std() + 1e-12
    )
    vibration = np.fft.irfft(np.fft.rfft(vibration) * device["sensor_response"], n_samples)
    level = vibration.std() + 1e-12

    if device["motor_hz_per_mm_s"] > 0:
        line_hz = device["motor_hz_per_mm_s"] * speed_mm_s
        for harmonic in range(1, 6):
            if harmonic * line_hz < 0.45 * RATE:
                vibration += (
                    device["motor_level"]
                    * level
                    / harmonic
                    * np.sin(2 * np.pi * harmonic * line_hz * time + rng.uniform(0, 6.28))
                )
    vibration += rng.normal(scale=device["noise_level"] * level, size=n_samples)
    return (vibration / (vibration.std() + 1e-12)).astype(np.float32)


def load_beats():
    sys.path.insert(0, str(ROOT / "models/beats"))
    from BEATs import BEATs, BEATsConfig

    checkpoint = torch.load(ROOT / "models/beats/BEATs_iter3_plus_AS2M.pt", map_location="cpu", weights_only=True)
    model = BEATs(BEATsConfig(checkpoint["cfg"]))
    model.load_state_dict(checkpoint["model"])
    return model.cuda().eval()


@torch.no_grad()
def embed_clips(model, clips):
    """Same recipe as the real recordings in the night experiments: 0.5 s windows at a fixed level, then the mean."""
    embeddings = []
    for clip in clips:
        windows = clip[: len(clip) // WINDOW * WINDOW].reshape(-1, WINDOW)
        windows = windows / (np.sqrt((windows**2).mean(axis=1, keepdims=True)) + 1e-12) * WAVE_RMS
        features = model.extract_features(torch.tensor(windows).cuda())[0].mean(dim=1)
        embeddings.append(features.mean(dim=0).float().cpu().numpy())
    return np.stack(embeddings)


OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
beats = load_beats()
rows = []
all_embeddings = []
for device_number in range(N_DEVICES):
    device = random_device(rng)
    clips = []
    for family in FAMILIES:
        for surface_number in range(SURFACES_PER_FAMILY):
            surface = random_surface(rng, family, device)
            for _ in range(CLIPS_PER_SURFACE):
                clips.append(simulated_clip(rng, device, surface))
                rows.append(
                    {"device": device_number, "family": family, "surface": f"{device_number}:{family}:{surface_number}"}
                )
    all_embeddings.append(embed_clips(beats, clips))
    if device_number % 25 == 0:
        print("device", device_number, flush=True)

table = pd.DataFrame(rows)
np.savez(
    OUT / "embeddings.npz",
    BEATs=np.concatenate(all_embeddings),
    device=table.device.to_numpy(),
    family=table.family.to_numpy(),
    surface=table.surface.to_numpy(),
)
print("saved", len(table), "clips", flush=True)

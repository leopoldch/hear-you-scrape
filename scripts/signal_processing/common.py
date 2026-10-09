"""Shared code: paths, reading signals, and loading the spectra cache.

Run extract_spectra.py first; every other script reads its cache through load().
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import ndimage, signal
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from surface_families import surface_families  # noqa: E402

CLUSTER = ROOT / "data/cluster/cluster_haptic_texture_dataset/sensor_data"
LMT = ROOT / "data/lmt108/LMT_108_SurfaceMaterials_Database"
OUT = ROOT / "analysis/signal_processing"
OUT.mkdir(parents=True, exist_ok=True)

FAMILIES = ["wood", "metal", "textile", "polymer"]
PALETTE = {"wood": "#2a78d6", "metal": "#eb6834", "textile": "#1baf7a", "polymer": "#eda100"}
sns.set_theme(style="whitegrid", font_scale=0.8, rc={"axes.formatter.min_exponent": 5})

AUDIO_RATE = 44100
ACCEL_RATE = 8000  # acceleration of both datasets is resampled to 8 kHz
AUDIO_NPERSEG = 8192
ACCEL_NPERSEG = 1600
AUDIO_FREQ = np.fft.rfftfreq(AUDIO_NPERSEG, 1 / AUDIO_RATE)[:2049]
ACCEL_FREQ = np.fft.rfftfreq(ACCEL_NPERSEG, 1 / ACCEL_RATE)
COHERENCE_FREQ = np.fft.rfftfreq(1024, 1 / ACCEL_RATE)

# Bands where both datasets are usable (see bandwidth_and_noise.py)
AUDIO_BAND = (100, 8000)
ACCEL_BAND = (10, 600)


def list_recordings():
    """DataFrame with one row per recording of a kept surface."""
    rows = []

    for path in sorted((CLUSTER / "audio").rglob("*.wav")):
        surface, direction, velocity, force, repeat = path.stem.split("_")
        rows.append(
            {
                "dataset": "cluster",
                "surface": surface,
                "record": path.stem,
                "direction": int(direction),
                "velocity": int(velocity),
                "force": int(force),
                "repeat": int(repeat),
                "audio": path,
                "accel": [CLUSTER / "accel" / surface / f"{path.stem}.csv"],
            }
        )

    for path in sorted((LMT / "SoundScans/Movement").rglob("*.wav")):
        surface, trial = path.stem.split("_Sound_Movement_")
        accel_folder = LMT / "AccelScansComponents/Movement" / path.parent.name
        rows.append(
            {
                "dataset": "lmt108",
                "surface": surface,
                "record": path.stem,
                "audio": path,
                "accel": [accel_folder / f"{surface}_Movement_{axis}_{trial}.txt" for axis in "XYZ"],
            }
        )

    recordings = pd.DataFrame(rows)
    families = surface_families()[["dataset", "surface", "family"]]
    recordings = recordings.merge(families, on=["dataset", "surface"])
    return recordings[recordings.family != ""].reset_index(drop=True)


def read_audio(path):
    """Mono audio as floats."""
    rate, samples = wavfile.read(path)
    assert rate == AUDIO_RATE
    if samples.ndim > 1:
        samples = samples[:, 0]
    return samples.astype(float)


def read_accel(row):
    """3-axis acceleration (n x 3) on a regular 8 kHz grid."""
    if row["dataset"] == "lmt108":
        # LMT108: 10 kHz, no timestamps
        samples = np.column_stack([np.loadtxt(path) for path in row["accel"]])
        return signal.resample_poly(samples, 4, 5, axis=0)

    # Cluster: irregular timestamps, so interpolate on a regular time grid
    table = np.loadtxt(row["accel"][0], delimiter=",", skiprows=1)
    times = table[:, 0]
    regular_times = np.arange(times[0], times[-1], 1 / ACCEL_RATE)
    return np.column_stack([np.interp(regular_times, times, table[:, axis]) for axis in [1, 2, 3]])


def contact_segment(audio, accel, dataset):
    """(start, stop) of steady sliding, in seconds.

    Cluster: starts 0.2 s after the touchdown shock (largest acceleration in the
    first 0.45 s) and stops 0.15 s before the audio level drops.
    LMT108: the whole recording minus 0.1 s at each end.
    """
    if dataset == "lmt108":
        duration = len(audio) / AUDIO_RATE
        return 0.1, duration - 0.1

    beginning = accel[: int(0.45 * ACCEL_RATE)]
    shock = np.abs(beginning - np.median(beginning, axis=0)).sum(axis=1)
    touchdown = np.argmax(shock) / ACCEL_RATE

    frame = int(0.05 * AUDIO_RATE)
    frames = audio[: len(audio) // frame * frame].reshape(-1, frame)
    envelope = np.sqrt(np.mean(frames**2, axis=1))
    middle = envelope[len(envelope) // 4 : 3 * len(envelope) // 4]
    active_frames = np.flatnonzero(envelope > 0.5 * np.median(middle))
    end_of_contact = active_frames[-1] * 0.05

    return touchdown + 0.2, end_of_contact - 0.15


def machine_lines(meta, psd, freq, band, filter_width):
    """Narrow peaks of the median Cluster spectrum, for each velocity.

    A peak that survives the median over 72 different surfaces is a candidate setup
    line (it may still carry part of the material response). kind = "drive" if f is a
    multiple of 3.5 v. Lines are detected on all surfaces, which is fine for exploration;
    in an evaluation, detect them on the training surfaces only.
    """
    in_band = (freq >= band[0]) & (freq <= band[1])
    lines = []
    for velocity in [20, 30, 40, 50, 60]:
        median_spectrum = np.median(psd[meta.velocity == velocity], axis=0)
        level = 10 * np.log10(median_spectrum)
        prominence = level - ndimage.median_filter(level, filter_width)
        peaks, _ = signal.find_peaks(np.where(in_band, prominence, 0), height=5)

        for peak in peaks:
            drive_frequency = 3.5 * velocity
            harmonic = freq[peak] / drive_frequency
            distance_to_harmonic = abs(harmonic - round(harmonic)) * drive_frequency
            lines.append(
                {
                    "velocity": velocity,
                    "hz": freq[peak],
                    "bin": peak,
                    "prominence_db": prominence[peak],
                    "kind": "drive" if distance_to_harmonic < 8 else "fixed",
                }
            )
    return pd.DataFrame(lines)


def remove_lines(meta, psd, lines):
    """Replace +-2 bins around each machine line by interpolation (in log power)."""
    clean = psd.copy()
    for velocity, lines_at_velocity in lines.groupby("velocity"):
        masked_bins = np.zeros(psd.shape[1], dtype=bool)
        for peak in lines_at_velocity.bin:
            masked_bins[peak - 2 : peak + 3] = True

        recordings = (meta.velocity == velocity).to_numpy()
        log_power = pd.DataFrame(np.log(psd[recordings]))
        log_power.loc[:, masked_bins] = np.nan
        clean[recordings] = np.exp(log_power.interpolate(axis=1).to_numpy())
    return clean


def load(clean=True):
    """Return (meta, psd): one row of meta per recording, psd a dict of arrays.

    clean=True removes the Cluster machine lines from the audio and acceleration spectra.
    """
    meta = pd.read_csv(OUT / "recordings.csv", dtype={"surface": str})
    psd = dict(np.load(OUT / "psd_cache.npz"))
    if clean:
        audio_lines = machine_lines(meta, psd["audio"], AUDIO_FREQ, AUDIO_BAND, filter_width=41)
        accel_lines = machine_lines(meta, psd["accel"], ACCEL_FREQ, ACCEL_BAND, filter_width=15)
        psd["audio"] = remove_lines(meta, psd["audio"], audio_lines)
        psd["accel"] = remove_lines(meta, psd["accel"], accel_lines)
    return meta, psd


def third_octaves(psd, freq, band):
    """1/3-octave band levels in dB. Columns are band centres (Hz); bands need >= 2 bins."""
    centers = 1000 * 2.0 ** (np.arange(-30, 30) / 3)
    lower_edges = centers / 2 ** (1 / 6)
    upper_edges = centers * 2 ** (1 / 6)
    centers = centers[(lower_edges >= band[0]) & (upper_edges <= band[1])]
    edges = np.append(centers / 2 ** (1 / 6), centers[-1] * 2 ** (1 / 6))

    band_of_each_bin = pd.cut(freq, edges, labels=centers.round().astype(int), right=False)
    power = pd.DataFrame(psd).T.groupby(band_of_each_bin, observed=True).sum().T

    bins_per_band = pd.Series(band_of_each_bin).value_counts()
    wide_enough = [center for center in power.columns if bins_per_band[center] >= 2]
    return 10 * np.log10(power[wide_enough])


def shapes(psd):
    """One row per recording: audio then accel 1/3-octave levels, each minus its mean.

    Removing the mean level keeps only the spectral shape (sensor gains differ between datasets).
    """
    audio = third_octaves(psd["audio"], AUDIO_FREQ, AUDIO_BAND)
    accel = third_octaves(psd["accel"], ACCEL_FREQ, ACCEL_BAND)
    audio_shape = audio.sub(audio.mean(axis=1), axis=0).add_prefix("audio ")
    accel_shape = accel.sub(accel.mean(axis=1), axis=0).add_prefix("accel ")
    return pd.concat([audio_shape, accel_shape], axis=1)


def save(fig, name):
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight")
    print("saved", OUT / name)

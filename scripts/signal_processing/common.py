import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import ndimage, signal
from scipy.io import wavfile
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from surface_families import surface_families  # noqa: E402

CLUSTER = ROOT / "data/cluster/cluster_haptic_texture_dataset/sensor_data"
LMT = ROOT / "data/lmt108/LMT_108_SurfaceMaterials_Database"
SENS3 = ROOT / "data/sens3"
OUT = ROOT / "analysis/signal_processing"
OUT.mkdir(parents=True, exist_ok=True)

FAMILIES = ["wood", "metal", "textile", "polymer"]
PALETTE = {"wood": "#2a78d6", "metal": "#eb6834", "textile": "#1baf7a", "polymer": "#eda100"}
sns.set_theme(style="whitegrid", font_scale=0.8, rc={"axes.formatter.min_exponent": 5})

AUDIO_RATE = 44100
ACCEL_RATE = 8000
AUDIO_NPERSEG = 8192
ACCEL_NPERSEG = 1600
AUDIO_FREQ = np.fft.rfftfreq(AUDIO_NPERSEG, 1 / AUDIO_RATE)[:2049]
ACCEL_FREQ = np.fft.rfftfreq(ACCEL_NPERSEG, 1 / ACCEL_RATE)
COHERENCE_FREQ = np.fft.rfftfreq(1024, 1 / ACCEL_RATE)


AUDIO_BAND = (100, 8000)
ACCEL_BAND = (10, 600)


SENS3_CHUNK_SECONDS = 4.0
SENS3_SENSOR_RATE = 10000
SENS3_MIN_FORCE = 0.1
SENS3_MIN_CONTACT = 0.95
SENS3_MIN_SPEED = 20


def list_recordings():
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


def list_sens3_recordings():
    families = surface_families()
    families = families[(families.dataset == "sens3") & (families.family != "")]
    rows = []
    for surface, family in zip(families.surface, families.family, strict=True):
        for session in sorted((SENS3 / f"Material{surface}/Sliding_Friction-roughness").iterdir()):
            sensor_path = session / f"Material_sensor_{surface}.npy"
            audio_path = session / f"Material_{surface}.wav"
            normal_force = -np.load(sensor_path, mmap_mode="r")[:, 5]
            position = pd.read_csv(session / f"Material_IR_pos_{surface}.csv", header=None)
            position_times, speed = position[0].to_numpy(), position[1].to_numpy()
            audio_duration = len(wavfile.read(audio_path, mmap=True)[1]) / AUDIO_RATE
            duration = min(len(normal_force) / SENS3_SENSOR_RATE, audio_duration, position_times[-1])

            for k in range(int(duration // SENS3_CHUNK_SECONDS)):
                start, stop = k * SENS3_CHUNK_SECONDS, (k + 1) * SENS3_CHUNK_SECONDS
                force = normal_force[int(start * SENS3_SENSOR_RATE) : int(stop * SENS3_SENSOR_RATE)]
                chunk_speed = np.median(speed[(position_times >= start) & (position_times < stop)])
                if np.mean(force > SENS3_MIN_FORCE) < SENS3_MIN_CONTACT or chunk_speed < SENS3_MIN_SPEED:
                    continue
                rows.append(
                    {
                        "dataset": "sens3",
                        "surface": surface,
                        "participant": int(session.name.removeprefix("Participant")),
                        "record": f"{surface}_{session.name}_{k}",
                        "chunk_start": start,
                        "chunk_stop": stop,
                        "velocity": chunk_speed,
                        "force": np.median(force),
                        "audio": audio_path,
                        "accel": [sensor_path],
                        "family": family,
                    }
                )
    return pd.DataFrame(rows)


def read_audio(path):
    rate, samples = wavfile.read(path)
    assert rate == AUDIO_RATE
    if samples.ndim > 1:
        samples = samples[:, 0]
    return samples.astype(float)


def read_accel(row):
    if row["dataset"] == "lmt108":
        samples = np.column_stack([np.loadtxt(path) for path in row["accel"]])
        return signal.resample_poly(samples, 4, 5, axis=0)

    table = np.loadtxt(row["accel"][0], delimiter=",", skiprows=1)
    times = table[:, 0]
    regular_times = np.arange(times[0], times[-1], 1 / ACCEL_RATE)
    return np.column_stack([np.interp(regular_times, times, table[:, axis]) for axis in [1, 2, 3]])


def read_recording(row):
    if row["dataset"] != "sens3":
        audio = read_audio(row["audio"])
        accel = read_accel(row)
        start, stop = contact_segment(audio, accel, row["dataset"])
        return audio, accel, start, stop

    rate, audio = wavfile.read(row["audio"], mmap=True)
    assert rate == AUDIO_RATE
    audio = audio[int(row["chunk_start"] * AUDIO_RATE) : int(row["chunk_stop"] * AUDIO_RATE)].astype(float)
    sensor = np.load(row["accel"][0], mmap_mode="r")
    first, last = int(row["chunk_start"] * SENS3_SENSOR_RATE), int(row["chunk_stop"] * SENS3_SENSOR_RATE)
    accel = signal.resample_poly(np.asarray(sensor[first:last, :3], dtype=float), 4, 5, axis=0)
    return audio, accel, 0.0, row["chunk_stop"] - row["chunk_start"]


def contact_segment(audio, accel, dataset):
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
    meta = pd.read_csv(OUT / "recordings.csv", dtype={"surface": str})
    psd = dict(np.load(OUT / "psd_cache.npz"))
    if clean:
        audio_lines = machine_lines(meta, psd["audio"], AUDIO_FREQ, AUDIO_BAND, filter_width=41)
        accel_lines = machine_lines(meta, psd["accel"], ACCEL_FREQ, ACCEL_BAND, filter_width=15)
        psd["audio"] = remove_lines(meta, psd["audio"], audio_lines)
        psd["accel"] = remove_lines(meta, psd["accel"], accel_lines)
    return meta, psd


def third_octaves(psd, freq, band):
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
    parts = []
    for name, freq, band in [("audio", AUDIO_FREQ, AUDIO_BAND), ("accel", ACCEL_FREQ, ACCEL_BAND)]:
        if name in psd:
            levels = third_octaves(psd[name], freq, band)
            parts.append(levels.sub(levels.mean(axis=1), axis=0).add_prefix(f"{name} "))
    return pd.concat(parts, axis=1)


def surface_folds(meta):
    surfaces = meta[["surface", "family"]].drop_duplicates().reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_of_surface = {}
    for fold, (_, test) in enumerate(splitter.split(surfaces.surface, surfaces.family)):
        for surface in surfaces.surface.iloc[test]:
            fold_of_surface[surface] = fold
    return meta.surface.map(fold_of_surface).to_numpy()


def save(fig, name):
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight")
    print("saved", OUT / name)

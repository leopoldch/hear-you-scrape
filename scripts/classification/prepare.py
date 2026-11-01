import tarfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy import signal
from scipy.io import wavfile

from evaluation import (
    ACCEL_RATE,
    AUDIO_FREQ,
    AUDIO_NPERSEG,
    AUDIO_RATE,
    OUT,
    ROOT,
    list_recordings,
    list_sens3_recordings,
    read_recording,
)
from extract_spectra import process


def bundle():
    path = ROOT / "dist/hear-you-scrape-bundle.tar"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        for name in [
            "signal_processing/recordings.csv",
            "classification/sens3_recordings.csv",
            "classification/waveforms.npz",
            "classification/sens3_waveforms.npz",
        ]:
            archive.add(ROOT / "analysis" / name, arcname="analysis/" + name)
    print("saved", path)


WINDOW_SECONDS = 0.5
MAX_WINDOWS = 8
MEL_RATE = 16000
ACCEL_BINS = slice(1, 39)
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=MEL_RATE, n_fft=512, hop_length=128, n_mels=64, f_min=50, f_max=8000
)
power_spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=64, power=2)


def log_shape(power):
    log_power = torch.log(power + 1e-10)
    return log_power - log_power.mean()


def spectrogram_windows(row):
    torch.set_num_threads(1)
    audio, accel, start, stop = read_recording(row)
    audio_contact = signal.resample_poly(audio[int(start * AUDIO_RATE) : int(stop * AUDIO_RATE)], 160, 441)
    accel_contact = accel[int(start * ACCEL_RATE) : int(stop * ACCEL_RATE)]
    accel_contact = accel_contact - accel_contact.mean(axis=0)
    audio_length = int(WINDOW_SECONDS * MEL_RATE)
    accel_length = int(WINDOW_SECONDS * ACCEL_RATE)
    n_windows = min(MAX_WINDOWS, len(audio_contact) // audio_length, len(accel_contact) // accel_length)
    audio_windows, accel_windows = ([], [])
    for k in range(n_windows):
        audio_piece = torch.tensor(audio_contact[k * audio_length : (k + 1) * audio_length], dtype=torch.float32)
        accel_piece = torch.tensor(accel_contact[k * accel_length : (k + 1) * accel_length].T, dtype=torch.float32)
        audio_windows.append(log_shape(mel_spectrogram(audio_piece)))
        accel_power = power_spectrogram(accel_piece).sum(dim=0)
        accel_windows.append(log_shape(accel_power[ACCEL_BINS]))
    return (torch.stack(audio_windows).numpy(), torch.stack(accel_windows).numpy())


def spectrograms():
    recordings = list_recordings()
    with Pool() as pool:
        results = pool.map(spectrogram_windows, [row for _, row in recordings.iterrows()], chunksize=8)
    audio = np.concatenate([audio_windows for audio_windows, _ in results])
    accel = np.concatenate([accel_windows for _, accel_windows in results])
    recording_of_window = np.concatenate([np.full(len(windows), k) for k, (windows, _) in enumerate(results)])
    np.savez(
        OUT / "spectrograms.npz",
        audio=audio.astype(np.float16),
        accel=accel.astype(np.float16),
        recording=recording_of_window,
    )
    print(f"{len(recording_of_window)} windows from {len(recordings)} recordings")
    print("saved", OUT / "spectrograms.npz")


def spectra_and_windows(row):
    return (process(row), spectrogram_windows(row))


def sens3():
    recordings = list_sens3_recordings()
    with Pool() as pool:
        results = pool.map(spectra_and_windows, [row for _, row in recordings.iterrows()], chunksize=8)
    spectra, windows = zip(*results, strict=True)
    infos, audio_psd, accel_psd, _, _ = zip(*spectra, strict=True)
    pd.DataFrame(infos).to_csv(OUT / "sens3_recordings.csv", index=False)
    np.savez_compressed(
        OUT / "sens3_psd_cache.npz",
        audio=np.array(audio_psd, dtype=np.float32),
        accel=np.array(accel_psd, dtype=np.float32),
    )
    audio = np.concatenate([audio_windows for audio_windows, _ in windows])
    accel = np.concatenate([accel_windows for _, accel_windows in windows])
    recording_of_window = np.concatenate([np.full(len(w), k) for k, (w, _) in enumerate(windows)])
    np.savez(
        OUT / "sens3_spectrograms.npz",
        audio=audio.astype(np.float16),
        accel=accel.astype(np.float16),
        recording=recording_of_window,
    )
    print(f"{len(recording_of_window)} windows from {len(recordings)} SENS3 chunks")
    print("saved", OUT / "sens3_recordings.csv", OUT / "sens3_psd_cache.npz", OUT / "sens3_spectrograms.npz")


HITS_DATA = ROOT / "data/greatest_hits"
CHUNK_SECONDS = 0.5
FAMILY_OF_MATERIAL = {"wood": "wood", "metal": "metal", "cloth": "textile", "carpet": "textile", "plastic": "polymer"}


def list_scratches():
    rows = []
    for times_path in sorted(HITS_DATA.glob("*_times.txt")):
        video = times_path.name.removesuffix("_times.txt")
        audio_path = HITS_DATA / f"{video}_denoised.wav"
        rate, samples = wavfile.read(audio_path, mmap=True)
        duration = len(samples) / rate
        events = pd.read_csv(times_path, sep="\\s+", header=None, names=["onset", "material", "action", "reaction"])
        next_onsets = np.append(events.onset.to_numpy()[1:], duration)
        for onset, material, action, next_onset in zip(
            events.onset, events.material, events.action, next_onsets, strict=True
        ):
            if action == "scratch" and material in FAMILY_OF_MATERIAL and (next_onset - onset >= CHUNK_SECONDS):
                rows.append(
                    {
                        "dataset": "greatest_hits",
                        "surface": f"{video} {material}",
                        "record": f"{video} {onset}",
                        "chunk_start": onset,
                        "material": material,
                        "family": FAMILY_OF_MATERIAL[material],
                        "audio": audio_path,
                    }
                )
    return pd.DataFrame(rows)


def spectrum_and_window(row):
    torch.set_num_threads(1)
    rate, samples = wavfile.read(row["audio"], mmap=True)
    start = int(row["chunk_start"] * rate)
    chunk = samples[start : start + int(CHUNK_SECONDS * rate)].astype(float)
    audio = signal.resample_poly(chunk, AUDIO_RATE // 300, rate // 300)
    _, psd = signal.welch(audio, AUDIO_RATE, nperseg=AUDIO_NPERSEG)
    mel_input = torch.tensor(signal.resample_poly(audio, 160, 441), dtype=torch.float32)
    return (psd[: len(AUDIO_FREQ)], log_shape(mel_spectrogram(mel_input)).numpy())


def greatest_hits():
    recordings = list_scratches()
    with Pool() as pool:
        results = pool.map(spectrum_and_window, [row for _, row in recordings.iterrows()], chunksize=32)
    psd, windows = zip(*results, strict=True)
    recordings.drop(columns="audio").to_csv(OUT / "greatest_hits_recordings.csv", index=False)
    np.savez_compressed(OUT / "greatest_hits_psd_cache.npz", audio=np.array(psd, dtype=np.float32))
    np.savez(
        OUT / "greatest_hits_spectrograms.npz",
        audio=np.array(windows, dtype=np.float16),
        recording=np.arange(len(windows)),
    )
    print(f"{len(recordings)} scratches from {recordings.surface.nunique()} (video, material) pairs")
    print(recordings.groupby("family").surface.agg(["size", "nunique"]))
    print("saved", OUT / "greatest_hits_recordings.csv", OUT / "greatest_hits_spectrograms.npz")


AUDIO_OUT_RATE = 16000


def unit_rms(x):
    return x / (np.sqrt(np.mean(x**2, axis=-1, keepdims=True)) + 1e-12)


def dominant_axis(accel):
    centered = accel - accel.mean(axis=0)
    _, _, directions = np.linalg.svd(centered, full_matrices=False)
    return centered @ directions[0]


def waveform_windows(row):
    audio, accel, start, stop = read_recording(row)
    audio_contact = signal.resample_poly(audio[int(start * AUDIO_RATE) : int(stop * AUDIO_RATE)], 160, 441)
    accel_contact = accel[int(start * ACCEL_RATE) : int(stop * ACCEL_RATE)]
    audio_length = int(WINDOW_SECONDS * AUDIO_OUT_RATE)
    accel_length = int(WINDOW_SECONDS * ACCEL_RATE)
    n_windows = min(MAX_WINDOWS, len(audio_contact) // audio_length, len(accel_contact) // accel_length)
    audio_windows = np.stack([audio_contact[k * audio_length : (k + 1) * audio_length] for k in range(n_windows)])
    accel_windows = np.stack(
        [dominant_axis(accel_contact[k * accel_length : (k + 1) * accel_length]) for k in range(n_windows)]
    )
    return (unit_rms(audio_windows).astype(np.float32), unit_rms(accel_windows).astype(np.float32))


def export(recordings, name):
    with Pool() as pool:
        results = pool.map(waveform_windows, [row for _, row in recordings.iterrows()], chunksize=8)
    audio = np.concatenate([audio_windows for audio_windows, _ in results])
    accel = np.concatenate([accel_windows for _, accel_windows in results])
    recording_of_window = np.concatenate([np.full(len(windows), k) for k, (windows, _) in enumerate(results)])
    np.savez(OUT / name, audio=audio.astype(np.float16), accel=accel.astype(np.float16), recording=recording_of_window)
    print(f"{len(recording_of_window)} windows from {len(recordings)} recordings, saved", OUT / name)


def waveforms():
    export(list_recordings(), "waveforms.npz")
    export(list_sens3_recordings(), "sens3_waveforms.npz")

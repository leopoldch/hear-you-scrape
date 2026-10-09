"""Compute one spectrum per recording and cache it (~6 min on 16 cores).

For each recording, on the contact segment only:
  - Welch PSD of the audio and of the acceleration (power of the 3 axes summed)
  - Cluster only: audio PSD before touchdown (background noise)
  - RMS, kurtosis and crest factor of the band-passed signals
  - audio / acceleration coherence, after aligning the two streams by cross-correlation
Outputs: analysis/signal_processing/recordings.csv and analysis/signal_processing/psd_cache.npz
"""

from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import signal, stats

from common import (
    ACCEL_BAND,
    ACCEL_NPERSEG,
    ACCEL_RATE,
    AUDIO_BAND,
    AUDIO_FREQ,
    AUDIO_NPERSEG,
    AUDIO_RATE,
    OUT,
    contact_segment,
    list_recordings,
    read_accel,
    read_audio,
)


def bandpass(x, rate, band):
    filter_ = signal.butter(4, band, "bandpass", fs=rate, output="sos")
    return signal.sosfiltfilt(filter_, x, axis=0)


def background_noise_psd(audio, start):
    """Audio PSD before touchdown (Cluster only; NaN when there is no quiet part)."""
    if start <= 0.4:
        return np.full(len(AUDIO_FREQ), np.nan)
    quiet = audio[int(0.02 * AUDIO_RATE) : int((start - 0.2) * AUDIO_RATE)]
    freq, psd = signal.welch(quiet, AUDIO_RATE, nperseg=min(AUDIO_NPERSEG, len(quiet)))
    return np.interp(AUDIO_FREQ, freq, psd)


def time_statistics(name, x):
    rms = np.sqrt(np.mean(x**2))
    return {
        f"{name}_log_rms": np.log10(rms),
        f"{name}_kurtosis": np.mean(stats.kurtosis(x, axis=0, fisher=False)),
        f"{name}_crest": np.abs(x).max() / rms,
    }


def aligned_coherence(audio, accel):
    """Coherence between audio (at 8 kHz) and the main acceleration axis.

    The two streams are first shifted by the lag (within +-100 ms) that
    maximises their cross-correlation. Returns (coherence, lag in ms).
    """
    audio_8k = signal.resample_poly(audio, 80, 441)
    main_axis = accel[:, np.argmax(accel.var(axis=0))]
    n = min(len(audio_8k), len(main_axis))
    audio_8k, main_axis = audio_8k[:n], main_axis[:n]

    xcorr = signal.correlate(audio_8k, main_axis, method="fft")
    lags = signal.correlation_lags(n, n)
    allowed = np.abs(lags) <= 0.1 * ACCEL_RATE
    lag = lags[allowed][np.argmax(np.abs(xcorr[allowed]))]

    if lag >= 0:
        audio_8k, main_axis = audio_8k[lag:], main_axis[: n - lag]
    else:
        audio_8k, main_axis = audio_8k[: n + lag], main_axis[-lag:]

    _, coherence = signal.coherence(audio_8k, main_axis, ACCEL_RATE, nperseg=1024)
    return coherence, lag / ACCEL_RATE * 1000


def process(row):
    audio = read_audio(row.audio)
    accel = read_accel(row)
    start, stop = contact_segment(audio, accel, row.dataset)

    audio_contact = audio[int(start * AUDIO_RATE) : int(stop * AUDIO_RATE)]
    accel_contact = accel[int(start * ACCEL_RATE) : int(stop * ACCEL_RATE)]
    accel_contact = accel_contact - accel_contact.mean(axis=0)  # remove gravity

    _, audio_psd = signal.welch(audio_contact, AUDIO_RATE, nperseg=AUDIO_NPERSEG)
    _, accel_psd = signal.welch(accel_contact, ACCEL_RATE, nperseg=ACCEL_NPERSEG, axis=0)
    noise_psd = background_noise_psd(audio, start)

    audio_filtered = bandpass(audio_contact, AUDIO_RATE, AUDIO_BAND)
    accel_filtered = bandpass(accel_contact, ACCEL_RATE, ACCEL_BAND)
    coherence, lag_ms = aligned_coherence(audio_filtered, accel_filtered)

    info = row.drop(["audio", "accel"]).to_dict()
    info.update(start=start, stop=stop, lag_ms=lag_ms)
    info.update(time_statistics("audio", audio_filtered))
    info.update(time_statistics("accel", accel_filtered))
    return info, audio_psd[: len(AUDIO_FREQ)], accel_psd.sum(axis=1), noise_psd, coherence


if __name__ == "__main__":
    recordings = list_recordings()
    with Pool() as pool:
        results = pool.map(process, [row for _, row in recordings.iterrows()], chunksize=8)

    infos, audio_psd, accel_psd, noise_psd, coherence = zip(*results, strict=True)
    pd.DataFrame(infos).to_csv(OUT / "recordings.csv", index=False)
    np.savez_compressed(
        OUT / "psd_cache.npz",
        audio=np.array(audio_psd, dtype=np.float32),
        accel=np.array(accel_psd, dtype=np.float32),
        noise=np.array(noise_psd, dtype=np.float32),
        coherence=np.array(coherence, dtype=np.float32),
    )
    print("saved", OUT / "psd_cache.npz")

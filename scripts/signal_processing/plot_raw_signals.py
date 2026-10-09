"""What do the raw signals look like? One surface per family: waveform, spectrogram, acceleration.

Cluster at 20 and 60 mm/s (direction 0, 500 mN), LMT108 first trial. Reads raw files.
"""

import matplotlib.pyplot as plt
import numpy as np

from common import ACCEL_RATE, AUDIO_RATE, contact_segment, list_recordings, read_accel, read_audio, save

recordings = list_recordings()

cluster = recordings[
    (recordings.dataset == "cluster")
    & (recordings.direction == 0)
    & (recordings.force == 500)
    & (recordings.repeat == 0)
    & recordings.velocity.isin([20, 60])
]
lmt = recordings[recordings.dataset == "lmt108"]
examples = {
    "cluster": cluster.groupby(["family", "velocity"]).first().reset_index(),
    "lmt108": lmt.groupby("family").first().reset_index(),
}

for dataset, chosen in examples.items():
    fig, axes = plt.subplots(len(chosen), 3, figsize=(15, 2.2 * len(chosen)))
    for row_axes, (_, row) in zip(axes, chosen.iterrows(), strict=True):
        waveform_ax, spectrogram_ax, accel_ax = row_axes
        audio = read_audio(row.audio)
        accel = read_accel(row)
        start, stop = contact_segment(audio, accel, dataset)
        title = f"{row.family}, surface {row.surface}"
        if dataset == "cluster":
            title += f", {row.velocity:.0f} mm/s"

        audio_time = np.arange(len(audio)) / AUDIO_RATE
        waveform_ax.plot(audio_time, audio, lw=0.3, color="0.3")
        waveform_ax.axvspan(start, stop, alpha=0.15)
        waveform_ax.set_title(f"{title} - audio (shaded: analysed segment)")

        spectrogram_ax.specgram(audio, NFFT=2048, Fs=AUDIO_RATE, noverlap=1536)
        spectrogram_ax.set_yscale("log")
        spectrogram_ax.set_ylim(50, AUDIO_RATE / 2)
        spectrogram_ax.set_title("audio spectrogram")

        accel_time = np.arange(len(accel)) / ACCEL_RATE
        accel_ax.plot(accel_time, accel - accel.mean(axis=0), lw=0.3)
        accel_ax.set_title("acceleration X, Y, Z")
    fig.tight_layout()
    save(fig, f"raw_signals_{dataset}.png")

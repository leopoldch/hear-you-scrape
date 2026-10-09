"""Separate Cluster machine lines from contact noise with NMF (weeks 4 and 12).

Magnitude spectrogram V (freq x time) ~ W H, with K spectral templates W and their
activations H. Fitted on 8 surfaces (2 per family) at 40 mm/s, placed one after another.
A template is "machine" if it is narrow (half its energy within 3% of the band) and
active on every surface (its weakest per-surface mean activation is > 20% of its strongest).
We rebuild the spectrogram with and without those templates.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.decomposition import NMF

from common import ACCEL_RATE, AUDIO_RATE, FAMILIES, contact_segment, list_recordings, read_accel, read_audio, save

N_TEMPLATES = 12


def contact_spectrogram(row, modality, rate, nperseg, max_freq):
    """Magnitude spectrogram (10 Hz - max_freq) of the contact segment of one recording."""
    audio = read_audio(row.audio)
    accel = read_accel(row)
    start, stop = contact_segment(audio, accel, "cluster")
    if modality == "audio":
        x = audio
    else:
        x = accel[:, np.argmax(accel.var(axis=0))]
    freq, _, stft = signal.stft(x[int(start * rate) : int(stop * rate)], rate, nperseg=nperseg)
    shown = (freq >= 10) & (freq <= max_freq)
    return freq[shown], np.abs(stft[shown])


def half_energy_width(templates, freq):
    """Hz needed to hold half the energy of each template (largest bins first)."""
    largest_first = np.sort(templates, axis=0)[::-1]
    bins_needed = (largest_first.cumsum(axis=0) < 0.5).sum(axis=0)
    return bins_needed * (freq[1] - freq[0])


def presence_on_every_surface(activations, block_ends):
    """For each template: weakest / strongest mean activation over the surfaces."""
    block_starts = np.r_[0, block_ends[:-1]]
    block_lengths = block_ends - block_starts
    mean_per_surface = np.add.reduceat(activations, block_starts, axis=1) / block_lengths
    return mean_per_surface.min(axis=1) / mean_per_surface.max(axis=1)


recordings = list_recordings()
recordings = recordings[
    (recordings.dataset == "cluster")
    & (recordings.velocity == 40)
    & (recordings.direction == 0)
    & (recordings.force == 500)
    & (recordings.repeat == 0)
]
chosen = pd.concat([recordings[recordings.family == family].head(2) for family in FAMILIES])
surface_names = [f"{row.family} {row.surface}" for _, row in chosen.iterrows()]

for modality, rate, nperseg, max_freq in [("accel", ACCEL_RATE, 512, 1000), ("audio", AUDIO_RATE, 4096, 3000)]:
    # One long spectrogram: the 8 surfaces one after the other
    blocks = []
    for _, row in chosen.iterrows():
        freq, magnitude = contact_spectrogram(row, modality, rate, nperseg, max_freq)
        blocks.append(magnitude)
    spectrogram = np.hstack(blocks)
    block_ends = np.cumsum([block.shape[1] for block in blocks])

    nmf = NMF(N_TEMPLATES, beta_loss="kullback-leibler", solver="mu", max_iter=2000, init="nndsvda", random_state=0)
    templates = nmf.fit_transform(spectrogram)  # frequency x template
    activations = nmf.components_  # template x time
    scale = templates.sum(axis=0)
    templates = templates / scale
    activations = activations * scale[:, None]  # templates @ activations is unchanged

    width = half_energy_width(templates, freq)
    presence = presence_on_every_surface(activations, block_ends)
    machine = (width < 0.03 * max_freq) & (presence > 0.2)
    print(f"{modality}: half-energy width (Hz) {width.round().astype(int)}")
    print(f"  weakest / strongest activation over surfaces {presence.round(2)}")
    print(f"  machine templates {np.flatnonzero(machine)}")

    # Figure 1: templates and activations, stacked with an offset
    colors = np.where(machine, "red", "C0")
    stacked_templates = pd.DataFrame(templates + np.arange(N_TEMPLATES) * templates.max(), index=freq)
    stacked_activations = pd.DataFrame(
        (activations / activations.max(axis=1, keepdims=True)).T + np.arange(N_TEMPLATES)
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 3]})
    stacked_templates.plot(ax=axes[0], legend=False, color=colors, yticks=[], title="templates (red = machine)")
    stacked_activations.plot(ax=axes[1], legend=False, color=colors, yticks=[], lw=0.6)
    axes[1].vlines(block_ends[:-1], 0, N_TEMPLATES, color="black")
    axes[1].set_title(" | ".join(surface_names) + "   (activations)")
    save(fig, f"nmf_templates_{modality}.png")

    # Figure 2: original spectrogram, machine part, contact part
    machine_part = templates[:, machine] @ activations[machine]
    contact_part = templates[:, ~machine] @ activations[~machine]
    color_range = {"vmin": np.log(np.percentile(spectrogram, 30)), "vmax": np.log(np.percentile(spectrogram, 99.5))}
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    for ax, image, title in [
        (axes[0], spectrogram, "original"),
        (axes[1], machine_part, "machine templates only"),
        (axes[2], contact_part, "without machine templates"),
    ]:
        extent = [0, spectrogram.shape[1], freq[0], freq[-1]]
        ax.imshow(np.log(image + 1e-6), aspect="auto", origin="lower", extent=extent, **color_range)
        ax.vlines(block_ends[:-1], freq[0], freq[-1], color="white")
        ax.set(title=title, ylabel="Hz")
    save(fig, f"nmf_separation_{modality}.png")

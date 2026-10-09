"""Which frequencies are usable? How far is the signal above the noise?

1. Median spectrum of each dataset, and Cluster background noise measured before touchdown.
2. Signal-to-noise ratio of the Cluster audio, per frequency band.
3. Where the acceleration spectrum becomes flat (sensor noise floor).
4. Is that floor just quantization noise? A quantizer with step q adds white noise
   with PSD (q^2 / 12) / (fs / 2) per axis, times 3 axes here.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ACCEL_FREQ, AUDIO_FREQ, list_recordings, load, save


def db(power):
    return 10 * np.log10(power)


meta, psd = load(clean=False)
cluster = (meta.dataset == "cluster").to_numpy()
lmt = ~cluster

audio = pd.DataFrame(index=AUDIO_FREQ)
audio["cluster, contact"] = db(np.median(psd["audio"][cluster], axis=0))
audio["lmt108, contact"] = db(np.median(psd["audio"][lmt], axis=0))
audio["cluster, before contact"] = db(np.nanmedian(psd["noise"][cluster], axis=0))

accel = pd.DataFrame(index=ACCEL_FREQ)
accel["cluster"] = db(np.median(psd["accel"][cluster], axis=0))
accel["lmt108"] = db(np.median(psd["accel"][lmt], axis=0))

# 1. Figure (each curve relative to the maximum of the figure, 0 Hz bin dropped)
audio_relative = audio.iloc[1:] - audio.iloc[1:].max().max()
accel_relative = accel.iloc[1:] - accel.iloc[1:].max()
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
audio_relative.plot(ax=axes[0], logx=True, title="audio: median PSD (dB)")
accel_relative.plot(ax=axes[1], logx=True, title="acceleration: median PSD (dB)")
for ax in axes:
    ax.set_xlabel("frequency (Hz)")
save(fig, "bandwidth_noise.png")

# 2. Signal-to-noise ratio
snr = audio["cluster, contact"] - audio["cluster, before contact"]
bands = pd.cut(AUDIO_FREQ, [20, 100, 500, 2000, 8000, 11025])
print("Cluster audio SNR, contact vs before contact (dB):")
print(snr.groupby(bands, observed=True).median().round(1).to_string())

# 3. Acceleration noise floor
for dataset in ["cluster", "lmt108"]:
    floor = accel[dataset][(ACCEL_FREQ > 2000) & (ACCEL_FREQ < 3500)].median()
    above_floor = (accel[dataset] > floor + 1) & (ACCEL_FREQ < 3500)
    print(f"{dataset} accel: flat floor at {floor:.1f} dB, starting at {ACCEL_FREQ[above_floor].max():.0f} Hz")


# 4. Quantization
def read_cluster_axis(path):
    return np.loadtxt(path, delimiter=",", skiprows=1)[:, 3]


recordings = list_recordings()
for dataset, rate, read in [("cluster", 8600, read_cluster_axis), ("lmt108", 10000, np.loadtxt)]:
    some_files = recordings[recordings.dataset == dataset].accel.iloc[::150].str[0]
    steps = []
    for path in some_files:
        distinct_values = np.unique(read(path))
        steps.append(np.median(np.diff(distinct_values)))  # most common gap = quantization step
    step = max(steps)
    quantization_floor = db(3 * step**2 / 12 / (rate / 2))
    print(f"{dataset} accel: step = {step:.1e} g -> quantization floor {quantization_floor:.1f} dB")

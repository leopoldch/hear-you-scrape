"""Turn the Penn Haptic Texture Toolkit (HaTT) recordings into the project's window format.

HaTT (Culbertson, Lopez Delgado, Kuchenbecker, IEEE Haptics Symposium 2014): a handheld tool with
a 3.175 mm stainless-steel hemispherical tip, explored freely by one person on 100 textures; 10 s
per texture at 10 kHz, acceleration (x, y, z), speed, normal force. No audio.
Licence: non-profit research only, no redistribution (data/hatt/license.pdf). Data stay in data/.

Download from https://repository.upenn.edu/meam_papers/299: 0-RecordedData_Models.zip extracted to
data/hatt/raw/ and 1-RecordedData_Friction.zip extracted to data/hatt/raw_friction/ (two separate
10 s recordings per texture).

Like the other datasets: acceleration resampled to 8 kHz, each 0.5 s window projected on its
dominant axis and set to unit RMS; recordings of 4 s (8 windows). The first 0.5 s is skipped, so
each 10 s file gives 2 recordings, 4 per texture. Only textures with a family (scripts/surface_families.py).
Output: analysis/classification/hatt_recordings.csv and hatt_waveforms.npz (accel, recording).
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from surface_families import hatt_family  # noqa: E402

SOURCES = {"models": ROOT / "data/hatt/raw", "friction": ROOT / "data/hatt/raw_friction"}
OUT = ROOT / "analysis/classification"
HATT_RATE = 10000
ACCEL_RATE = 8000
WINDOW_SECONDS = 0.5
WINDOWS_PER_RECORDING = 8
SKIP_SECONDS = 0.5


def read_blocks(path):
    text = path.read_text()
    blocks = {}
    for name, body in re.findall(r"<([A-Za-z_]+)>\s*(<value>.*?)</\1>", text, flags=re.S):
        blocks[name] = np.array(re.findall(r"<value>([^<]+)</value>", body), dtype=float)
    return blocks


def unit_rms(x):
    return x / (np.sqrt(np.mean(x**2, axis=-1, keepdims=True)) + 1e-12)


def dominant_axis(accel):
    centered = accel - accel.mean(axis=0)
    _, _, directions = np.linalg.svd(centered, full_matrices=False)
    return centered @ directions[0]


def main():
    rows, windows, recording_of_window = [], [], []
    texture_dirs = [
        (source, path) for source, folder in SOURCES.items() if folder.exists() for path in sorted(folder.iterdir())
    ]
    for source, texture_dir in texture_dirs:
        name = texture_dir.name
        family, polymer_type = hatt_family(name)
        if not family:
            continue
        accel = read_blocks(texture_dir / f"Accel_{name}.xml")
        speed = read_blocks(texture_dir / f"Position_{name}.xml")["Speed"]
        force = read_blocks(texture_dir / f"Force_{name}.xml")["ForceNormal"]
        xyz = np.stack([accel["Accel_x"], accel["Accel_y"], accel["Accel_z"]], axis=1)
        xyz = resample_poly(xyz, 4, 5, axis=0)

        window = int(WINDOW_SECONDS * ACCEL_RATE)
        recording_length = window * WINDOWS_PER_RECORDING
        start = int(SKIP_SECONDS * ACCEL_RATE)
        record = 0
        while start + recording_length <= len(xyz):
            chunk = xyz[start : start + recording_length]
            seconds = slice(
                int(start / ACCEL_RATE * HATT_RATE), int((start + recording_length) / ACCEL_RATE * HATT_RATE)
            )
            for k in range(WINDOWS_PER_RECORDING):
                windows.append(unit_rms(dominant_axis(chunk[k * window : (k + 1) * window])))
                recording_of_window.append(len(rows))
            rows.append(
                {
                    "dataset": "hatt",
                    "surface": name,
                    "family": family,
                    "polymer_type": polymer_type,
                    "source": source,
                    "record": record,
                    "start": start / ACCEL_RATE,
                    "stop": (start + recording_length) / ACCEL_RATE,
                    "velocity": np.mean(speed[seconds]),
                    "force": np.mean(np.abs(force[seconds])),
                }
            )
            record += 1
            start += recording_length
        print(f"{source}, {name}: {family}, {record} recordings", flush=True)

    recordings = pd.DataFrame(rows)
    recordings.to_csv(OUT / "hatt_recordings.csv", index=False)
    np.savez(
        OUT / "hatt_waveforms.npz", accel=np.array(windows, dtype=np.float16), recording=np.array(recording_of_window)
    )
    print(
        recordings.groupby("family").agg(
            surfaces=("surface", "nunique"),
            recordings=("surface", "size"),
            speed_mm_s=("velocity", "median"),
            force_n=("force", "median"),
        )
    )
    print("saved", OUT / "hatt_recordings.csv", OUT / "hatt_waveforms.npz")


if __name__ == "__main__":
    main()

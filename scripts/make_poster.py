"""README banner: a parody of the "Alien" (1979) poster.

The egg is the spectrogram of a real Cluster scrape (velvet, 60 mm/s), cracked open over the
grid of the original poster, next to the tagline "In space no one can hear you scream",
with "scream" crossed out. Writes docs/poster.png.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse, Rectangle
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/signal_processing"))
from common import AUDIO_RATE, CLUSTER, read_audio  # noqa: E402

GREEN = "#b8ffcf"
DARK_TO_GREEN = LinearSegmentedColormap.from_list("egg", ["black", "#0b3d1f", "#3fae6a", GREEN])
EGG_X, EGG_Y, EGG_WIDTH, EGG_HEIGHT = 0.22, 0.55, 0.17, 0.62
HORIZON = EGG_Y - EGG_HEIGHT / 2 + 0.01  # the egg stands on the grid


def scrape_spectrogram():
    audio = read_audio(CLUSTER / "audio/102/102_0_60_500_0.wav")
    contact = audio[int(0.4 * AUDIO_RATE) : int(1.6 * AUDIO_RATE)]
    freq, _, power = signal.spectrogram(contact, AUDIO_RATE, nperseg=1024, noverlap=896)
    return 10 * np.log10(power[(freq > 50) & (freq < 12000)] + 1e-12)


def draw_floor_grid(ax):
    """Perspective grid under the egg, as on the original poster, kept on the left half."""
    floor = Rectangle((0, 0), 0.46, HORIZON, transform=ax.transData)
    lines = []
    for x_bottom in np.linspace(-0.6, 1.04, 25):
        lines += ax.plot([EGG_X, x_bottom], [HORIZON, 0], color="#1f5c35", lw=0.6)
    for depth in np.geomspace(0.006, HORIZON, 9):
        lines += ax.plot([0, 0.46], [HORIZON - depth, HORIZON - depth], color="#1f5c35", lw=0.6)
    for line in lines:
        line.set_clip_path(floor)


def draw_egg(ax, level):
    egg = Ellipse((EGG_X, EGG_Y), EGG_WIDTH, EGG_HEIGHT, facecolor="none", edgecolor="none")
    ax.add_patch(egg)
    left, right = EGG_X - EGG_WIDTH / 2, EGG_X + EGG_WIDTH / 2
    bottom, top = EGG_Y - EGG_HEIGHT / 2, EGG_Y + EGG_HEIGHT / 2
    image = ax.imshow(
        level,
        extent=(left, right, bottom, top),
        origin="lower",
        aspect="auto",
        cmap=DARK_TO_GREEN,
        vmin=np.percentile(level, 30),
        vmax=np.percentile(level, 99.5),
    )
    image.set_clip_path(egg)

    crack_x = np.linspace(left + 0.012, right - 0.012, 10)
    crack_y = bottom + 0.16 + np.random.default_rng(3).uniform(-0.02, 0.02, len(crack_x))
    for width, alpha in [(22, 0.05), (10, 0.15), (2.5, 1)]:
        ax.plot(crack_x, crack_y, color=GREEN, lw=width, alpha=alpha, solid_capstyle="round")


def draw_text(ax):
    crossed_out_scream = "".join(letter + "̶" for letter in "scream")
    ax.text(0.69, 0.66, "H E A R - Y O U - S C R A P E", color="white", ha="center", fontsize=30, weight="light")
    ax.text(0.69, 0.47, f"In space no one can hear you {crossed_out_scream}", color="0.6", ha="center", fontsize=17)
    ax.text(0.69, 0.37, "scrape.", color=GREEN, ha="center", fontsize=22, style="italic")
    ax.text(0.69, 0.12, "Cluster  ·  LMT108  ·  SENS3", color="0.45", ha="center", fontsize=10)


fig = plt.figure(figsize=(12, 5), facecolor="black")
ax = fig.add_axes((0, 0, 1, 1))
ax.set(xlim=(0, 1), ylim=(0, 1), facecolor="black")
ax.axis("off")

draw_floor_grid(ax)
draw_egg(ax, scrape_spectrogram())
draw_text(ax)

output = ROOT / "docs/poster.png"
output.parent.mkdir(exist_ok=True)
fig.savefig(output, dpi=110, facecolor="black")
print("saved", output)

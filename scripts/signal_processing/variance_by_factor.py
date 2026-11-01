import matplotlib.pyplot as plt
import pandas as pd

from common import ACCEL_BAND, ACCEL_FREQ, AUDIO_BAND, AUDIO_FREQ, load, save, third_octaves

meta, psd = load()
cluster = (meta.dataset == "cluster").to_numpy()
cluster_meta = meta[cluster]


def explained_share(levels, factors):
    groups = [cluster_meta[factor] for factor in factors]
    group_means = levels.groupby(groups).transform("mean")
    return group_means.var() / levels.var()


fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
for ax, modality, freq, band in [
    (axes[0], "audio", AUDIO_FREQ, AUDIO_BAND),
    (axes[1], "accel", ACCEL_FREQ, ACCEL_BAND),
]:
    levels = third_octaves(psd[modality][cluster], freq, band).set_index(cluster_meta.index)

    surface = explained_share(levels, ["surface"])
    velocity = explained_share(levels, ["velocity"])
    surface_and_velocity = explained_share(levels, ["surface", "velocity"])

    table = pd.DataFrame(
        {
            "surface": surface,
            "velocity": velocity,
            "surface x velocity": surface_and_velocity - surface - velocity,
            "direction": explained_share(levels, ["direction"]),
            "force": explained_share(levels, ["force"]),
        }
    )
    table["residual"] = 1 - table.sum(axis=1)
    family = explained_share(levels, ["family"])

    table.plot.bar(stacked=True, ax=ax, width=0.85, legend=False, title=f"Cluster {modality}")
    family.plot(ax=ax, color="black", marker=".", label="family (part of surface)", use_index=False)
    ax.set_xlabel("1/3-octave band centre (Hz)")
    ax.set_ylabel("share of variance")

    print(f"{modality}, mean share over bands:")
    print(table.mean().round(2).to_string())
    print(f"family {family.mean():.2f}\n")

axes[1].legend(bbox_to_anchor=(1, 1))
save(fig, "variance_decomposition.png")

"""H28 and H29: transfer the geometry between families, not the coordinates.

H21 showed that the family directions do not line up across setups (cosine near 0). This script
asks whether the *relative* geometry is shared instead: if metal and textile are far apart in
every setup, even along different axes, one could transfer the structure.

H28, relational transfer (k = 0, transductive: uses the new setup's unlabeled surfaces)
    1. Diagnostic (RSA): per setup, the 4 x 4 distance matrix between family centroids; rank
       correlation of the 6 distances between setups, with a permutation test over the
       24 relabelings of the families.
    2. Gromov-Wasserstein: entropic GW between the surfaces of a labeled setup and the
       surfaces of the new setup, using only the distances inside each setup; target surface
       family = the source family that receives most of its mass. Both source setups vote.
       The lowest GW loss over 20 random starts is kept (an unsupervised criterion).
       Control: GW between two halves of the same setup, where the geometry is shared.
    3. Family-level matching: k-means (4) on the new setup's surfaces, then the permutation
       of clusters whose 4 x 4 distance matrix best matches the source families'.
H29, rank-normalized physical descriptors
    Per setup, each descriptor (spectral centroid, slope, HF/LF, flatness, peakiness, level,
    kurtosis; audio and acceleration) is replaced by its rank. Is the order of the families
    the same in each setup (Kendall W)? Then a probe trained on ranks from two setups is tested
    on the third (k = 0), against z-scores. The same rank transform is applied to BEATs.

Representations: BEATs per-recording means (audio, acceleration), and the descriptors,
standardized per setup. Scores: balanced accuracy with an interval over surfaces.

    uv run python scripts/classification/relational_transfer.py
Output: analysis/classification/relational_transfer/ (scores.csv, rsa.csv, family_order.csv, figures)
"""

import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "signal_processing"))
from label_consistency import balanced_accuracy_with_interval  # noqa: E402
from label_consistency import load as load_beats  # noqa: E402

from common import ACCEL_BAND, ACCEL_FREQ, AUDIO_BAND, AUDIO_FREQ, third_octaves  # noqa: E402
from common import load as load_psd  # noqa: E402

DATA = Path("analysis/classification")
OUT = DATA / "relational_transfer"
OUT.mkdir(parents=True, exist_ok=True)
SETUPS = ["cluster", "lmt108", "sens3"]
FAMILIES = ["wood", "metal", "textile", "polymer"]
PAIRS = list(itertools.combinations(range(4), 2))


# ---------- data


def spectral_descriptors(psd, freq, band, split):
    in_band = (freq >= band[0]) & (freq <= band[1])
    f, p = freq[in_band], psd[:, in_band]
    level = 10 * np.log10(p)
    bands = third_octaves(psd, freq, band)
    return {
        "centroid": np.log(np.exp((p * np.log(f)).sum(1) / p.sum(1))),
        "slope": np.polyfit(np.log2(bands.columns.astype(float)), bands.T, 1)[0],
        "hf_lf": 10 * np.log10(p[:, f >= split].sum(1) / p[:, f < split].sum(1)),
        "flatness": stats.gmean(p, axis=1) / p.mean(1),
        "peakiness": np.percentile(level - ndimage.median_filter(level, (1, 31)), 99, axis=1),
    }


def load_descriptors():
    meta, psd = load_psd()
    sens3 = pd.read_csv(DATA / "sens3_recordings.csv", dtype={"surface": str})
    sens3_psd = dict(np.load(DATA / "sens3_psd_cache.npz"))
    tables = []
    for table, spectra in [(meta, psd), (sens3, sens3_psd)]:
        out = table[["dataset", "surface", "family"]].copy()
        for modality, freq, band, split in [
            ("audio", AUDIO_FREQ, AUDIO_BAND, 2000),
            ("acceleration", ACCEL_FREQ, ACCEL_BAND, 200),
        ]:
            short = "audio" if modality == "audio" else "accel"
            for name, values in spectral_descriptors(spectra[short], freq, band, split).items():
                out[f"{modality} {name}"] = values
            out[f"{modality} level"] = table[f"{short}_log_rms"]
            out[f"{modality} kurtosis"] = np.log(table[f"{short}_kurtosis"])
        tables.append(out)
    table = pd.concat(tables, ignore_index=True)
    table = table[table.family.isin(FAMILIES)].reset_index(drop=True)
    table["surface"] = table.dataset + ":" + table.surface
    names = [c for c in table.columns if c.startswith(("audio ", "acceleration "))]
    x = table[names].to_numpy(float)
    ok = np.isfinite(x).all(1)
    return table[ok].reset_index(drop=True), x[ok], names


def per_setup(x, table, how):
    out = np.empty_like(x, dtype=float)
    for s in SETUPS:
        rows = (table.dataset == s).to_numpy()
        if how == "z-score":
            out[rows] = (x[rows] - x[rows].mean(0)) / (x[rows].std(0) + 1e-9)
        else:  # rank, mapped to (0, 1)
            out[rows] = (stats.rankdata(x[rows], axis=0) - 0.5) / rows.sum()
    return out


def representations():
    selected, beats = load_beats()
    keep = selected.family.isin(FAMILIES).to_numpy()
    selected = selected[keep].reset_index(drop=True)
    table, desc, names = load_descriptors()
    audio_cols = [i for i, n in enumerate(names) if n.startswith("audio")]
    accel_cols = [i for i, n in enumerate(names) if n.startswith("acceleration")]
    return {
        "BEATs audio": (selected, beats["audio"][keep]),
        "BEATs acceleration": (selected, beats["acceleration"][keep]),
        "descriptors audio": (table, desc[:, audio_cols]),
        "descriptors acceleration": (table, desc[:, accel_cols]),
        "descriptors both": (table, desc),
    }, names


def surface_means(table, x):
    frame = pd.DataFrame(x).groupby(table.surface.to_numpy()).mean()
    info = table.drop_duplicates("surface").set_index("surface").loc[frame.index]
    info.index.name = "surface"
    return info.reset_index(), frame.to_numpy()


# ---------- H28 part 1: RSA


def family_distances(table, x, setup):
    rows = (table.dataset == setup).to_numpy()
    centroids = np.stack([x[rows & (table.family == f).to_numpy()].mean(0) for f in FAMILIES])
    d = np.array([np.linalg.norm(centroids[i] - centroids[j]) for i, j in PAIRS])
    return d / d.mean()


def relabeled(d, order):
    full = np.zeros((4, 4))
    for (i, j), v in zip(PAIRS, d):
        full[i, j] = full[j, i] = v
    return np.array([full[order[i], order[j]] for i, j in PAIRS])


def rsa(reps):
    rows, matrices = [], {}
    for name, (table, x) in reps.items():
        z = per_setup(x, table, "z-score")
        d = {s: family_distances(table, z, s) for s in SETUPS}
        matrices[name] = d
        for a, b in itertools.combinations(SETUPS, 2):
            rho = stats.spearmanr(d[a], d[b]).statistic
            null = [
                stats.spearmanr(d[a], relabeled(d[b], order)).statistic for order in itertools.permutations(range(4))
            ]
            rows.append(
                {
                    "representation": name,
                    "setups": f"{a} vs {b}",
                    "spearman": rho,
                    "p (24 relabelings)": np.mean(np.array(null) >= rho - 1e-12),
                    "same closest pair": PAIRS[np.argmin(d[a])] == PAIRS[np.argmin(d[b])],
                }
            )
    return pd.DataFrame(rows), matrices


# ---------- H28 part 2: Gromov-Wasserstein


def sinkhorn_log(p, q, cost, eps, n_iter=200):
    log_k = -cost / eps
    u, v = np.zeros(len(p)), np.zeros(len(q))
    for _ in range(n_iter):
        u = np.log(p) - logsumexp(log_k + v[None, :], axis=1)
        v = np.log(q) - logsumexp(log_k + u[:, None], axis=0)
    return np.exp(log_k + u[:, None] + v[None, :])


def entropic_gw(c1, c2, p, q, eps=0.01, n_iter=100, rng=None):
    const = (c1**2 @ p)[:, None] + (c2**2 @ q)[None, :]
    t = p[:, None] * q[None, :]
    if rng is not None:
        t = sinkhorn_log(p, q, -np.log(rng.random(t.shape) + 1e-3), 1.0)
    for _ in range(n_iter):
        tens = const - 2 * c1 @ t @ c2.T
        t = sinkhorn_log(p, q, tens, eps)
    return t, float(((const - 2 * c1 @ t @ c2.T) * t).sum())


def distance_matrix(x):
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    return d / d.max()


def gw_votes(src_info, src_x, tgt_x, n_starts=20, seed=0):
    """Mass each target surface receives from each source family (rows: target, cols: FAMILIES)."""
    fam = src_info.family.to_numpy()
    counts = pd.Series(fam).value_counts()
    p = np.array([1 / counts[f] for f in fam])
    p /= p.sum()  # every source family carries a quarter of the mass
    q = np.full(len(tgt_x), 1 / len(tgt_x))
    c1, c2 = distance_matrix(src_x), distance_matrix(tgt_x)
    rng = np.random.default_rng(seed)
    best = min((entropic_gw(c1, c2, p, q, rng=rng if k else None) for k in range(n_starts)), key=lambda r: r[1])[0]
    return np.stack([best[fam == f].sum(0) for f in FAMILIES], axis=1) / q[:, None]


def score_surfaces(table, surface_prediction):
    predicted = table.surface.map(surface_prediction).to_numpy()
    return balanced_accuracy_with_interval(table, predicted, FAMILIES)


def gromov_wasserstein(reps):
    rows = []
    for name, (table, x) in reps.items():
        z = per_setup(x, table, "z-score")
        info, means = surface_means(table, z)
        for new in SETUPS:
            target = (info.dataset == new).to_numpy()
            votes = sum(
                gw_votes(
                    info[info.dataset == s].reset_index(drop=True), means[(info.dataset == s).to_numpy()], means[target]
                )
                for s in SETUPS
                if s != new
            )
            prediction = dict(zip(info.surface[target], np.array(FAMILIES)[votes.argmax(1)]))
            sub = table[table.dataset == new].reset_index(drop=True)
            ba, low, high = score_surfaces(sub, prediction)
            rows.append(
                {
                    "method": "Gromov-Wasserstein, k = 0",
                    "representation": name,
                    "new setup": new,
                    "balanced accuracy": ba,
                    "low": low,
                    "high": high,
                }
            )
            # family-level matching of 4 clusters
            src_d = np.mean([family_distances(table, z, s) for s in SETUPS if s != new], axis=0)
            clusters = KMeans(4, n_init=20, random_state=0).fit_predict(means[target])
            cl_cent = np.stack([means[target][clusters == c].mean(0) for c in range(4)])
            tgt_d = np.array([np.linalg.norm(cl_cent[i] - cl_cent[j]) for i, j in PAIRS])
            tgt_d /= tgt_d.mean()
            order = min(itertools.permutations(range(4)), key=lambda o: np.abs(relabeled(tgt_d, o) - src_d).sum())
            # order[f] = cluster that plays family f
            cluster_family = {order[f]: FAMILIES[f] for f in range(4)}
            prediction = dict(zip(info.surface[target], [cluster_family[c] for c in clusters]))
            ba, low, high = score_surfaces(sub, prediction)
            rows.append(
                {
                    "method": "cluster matching, k = 0",
                    "representation": name,
                    "new setup": new,
                    "balanced accuracy": ba,
                    "low": low,
                    "high": high,
                }
            )
            # control: GW between two halves of the new setup itself
            scores = []
            for split in range(10):
                rng = np.random.default_rng(split)
                half = np.zeros(target.sum(), bool)
                for f in FAMILIES:
                    idx = np.flatnonzero(info.family[target].to_numpy() == f)
                    half[rng.choice(idx, len(idx) // 2, replace=False)] = True
                tinfo, tmeans = info[target].reset_index(drop=True), means[target]
                votes = gw_votes(
                    tinfo[half].reset_index(drop=True), tmeans[half], tmeans[~half], n_starts=10, seed=split
                )
                prediction = dict(zip(tinfo.surface[~half], np.array(FAMILIES)[votes.argmax(1)]))
                held = sub[sub.surface.isin(tinfo.surface[~half])].reset_index(drop=True)
                scores.append(score_surfaces(held, prediction)[0])
            rows.append(
                {
                    "method": "control: GW within the setup",
                    "representation": name,
                    "new setup": new,
                    "balanced accuracy": np.mean(scores),
                    "low": np.percentile(scores, 10),
                    "high": np.percentile(scores, 90),
                }
            )
            print(rows[-3:], flush=True)
    return rows


# ---------- H29: ranks


def kendall_w(ranks):
    m, n = ranks.shape
    return 12 * ((ranks.sum(0) - ranks.sum(0).mean()) ** 2).sum() / (m**2 * (n**3 - n))


def family_order(table, x, names):
    rng = np.random.default_rng(0)
    null = np.array([kendall_w(np.array([rng.permutation(4) + 1 for _ in SETUPS])) for _ in range(20000)])
    r = per_setup(x, table, "rank")
    rows = []
    for k, name in enumerate(names):
        med = {
            s: [np.median(r[(table.dataset == s).to_numpy() & (table.family == f).to_numpy(), k]) for f in FAMILIES]
            for s in SETUPS
        }
        ranks = np.array([stats.rankdata(med[s]) for s in SETUPS])  # 3 setups x 4 families
        w = kendall_w(ranks)
        rows.append(
            {
                "descriptor": name,
                "kendall W": w,
                "p (random orders)": np.mean(null >= w - 1e-12),
                **{f"order {s}": " < ".join(np.array(FAMILIES)[np.argsort(med[s])]) for s in SETUPS},
            }
        )
    return pd.DataFrame(rows).sort_values("kendall W", ascending=False)


def rank_probes(reps):
    rows = []
    for name, (table, x) in reps.items():
        for how in ["z-score", "rank"]:
            xs = per_setup(x, table, how)
            for new in SETUPS:
                target = (table.dataset == new).to_numpy()
                train = table[~target]
                weights = 1 / train.groupby(["dataset", "family"]).family.transform("size").to_numpy()
                model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
                model.fit(xs[~target], train.family, logisticregression__sample_weight=weights / weights.mean())
                ba, low, high = balanced_accuracy_with_interval(table[target], model.predict(xs[target]), FAMILIES)
                rows.append(
                    {
                        "method": f"probe on {how} per setup, k = 0",
                        "representation": name,
                        "new setup": new,
                        "balanced accuracy": ba,
                        "low": low,
                        "high": high,
                    }
                )
    return rows


# ---------- figures


def plot_rsa(matrices):
    fig, axes = plt.subplots(len(matrices), 3, figsize=(8, 2.4 * len(matrices)))
    for r, (name, d) in enumerate(matrices.items()):
        for c, s in enumerate(SETUPS):
            full = np.zeros((4, 4))
            for (i, j), v in zip(PAIRS, d[s]):
                full[i, j] = full[j, i] = v
            ax = axes[r, c]
            ax.imshow(full, cmap="Blues", vmin=0, vmax=2)
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, f"{full[i, j]:.2f}", ha="center", va="center", fontsize=7)
            ax.set_xticks(range(4), FAMILIES, fontsize=7, rotation=45)
            ax.set_yticks(range(4), FAMILIES, fontsize=7)
            ax.set_title(f"{name}\n{s}", fontsize=8)
    fig.suptitle("Distances between family centroids, per setup (divided by their mean)")
    fig.tight_layout()
    fig.savefig(OUT / "family_distances.png", dpi=150)


def plot_scores(scores):
    methods = scores.method.unique()
    fig, axes = plt.subplots(1, len(methods), figsize=(3.2 * len(methods), 3.6), sharey=True)
    for ax, method in zip(axes, methods):
        part = scores[scores.method == method]
        for k, setup in enumerate(SETUPS):
            s = part[part["new setup"] == setup]
            y = np.arange(len(s)) + 0.25 * (k - 1)
            ax.errorbar(
                s["balanced accuracy"],
                y,
                xerr=[s["balanced accuracy"] - s.low, s.high - s["balanced accuracy"]],
                fmt="o",
                ms=4,
                label=setup,
            )
        ax.set_yticks(range(len(s)), s.representation, fontsize=7)
        ax.axvline(0.25, color="gray", ls="--", lw=1)
        ax.set_title(method, fontsize=8)
        ax.set_xlim(0, 1)
    axes[0].legend(fontsize=7)
    fig.supxlabel("balanced accuracy (chance 0.25)")
    fig.tight_layout()
    fig.savefig(OUT / "scores.png", dpi=150)


def main():
    reps, names = representations()
    rsa_table, matrices = rsa(reps)
    rsa_table.to_csv(OUT / "rsa.csv", index=False)
    print(rsa_table.to_string(), flush=True)
    plot_rsa(matrices)

    table, x = reps["descriptors both"]
    order = family_order(table, x, names)
    order.to_csv(OUT / "family_order.csv", index=False)
    print(order.to_string(), flush=True)

    scores = pd.DataFrame(rank_probes(reps) + gromov_wasserstein(reps))
    scores.to_csv(OUT / "scores.csv", index=False)
    print(scores.round(2).to_string())
    plot_scores(scores)


if __name__ == "__main__":
    main()

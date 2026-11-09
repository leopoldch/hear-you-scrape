"""H27: ask CLAP directly, with text, which material is being scraped (no training on our setups).

CLAP (laion/clap-htsat-unfused) was trained on web audio paired with text. Each recording is
embedded (mean of its 0.5 s windows, unit RMS as everywhere else) and compared by cosine to
text prompts such as "the sound of scraping wood". Prompt ensembling: several templates x several
words per family, averaged per family. No label of any setup is used.

Two readouts:
    raw            argmax of the cosine
    prior-balanced the cosine minus its mean over the setup's recordings, per family (uses the
                   setup's unlabeled recordings only: corrects a constant preference for one word)
Data: our 3 setups (audio, and acceleration fed as audio) and Greatest Hits scratches (airborne
microphone, video-like recordings: the "web-like" control).

    uv run python scripts/classification/clap_text_zero_shot.py
Output: analysis/classification/clap_text_zero_shot/scores.csv, clap_text_zero_shot.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrated_new_setup import DATA, DEVICE, FAMILIES, clap_embedder, embed_windows, score  # noqa: E402

OUT = DATA / "clap_text_zero_shot"
OUT.mkdir(parents=True, exist_ok=True)
TEMPLATES = [
    "the sound of scraping {}",
    "a stick scratching a {} surface",
    "rubbing a finger on {}",
    "friction noise of sliding on {}",
    "scratching {}",
]
WORDS = {
    "wood": ["wood", "a wooden board", "timber"],
    "metal": ["metal", "steel", "aluminium"],
    "textile": ["fabric", "cloth", "textile"],
    "polymer": ["plastic", "rubber", "vinyl"],
}


def our_setups():
    """Per-recording CLAP embeddings of audio and acceleration (cached)."""
    selected = pd.read_csv(DATA / "qwen_audio/selected_recordings.csv", dtype={"surface": str})
    selected["surface"] = selected.dataset + ":" + selected.surface
    path = OUT / "clap_our_setups.npz"
    if not path.exists():
        blocks = {
            "audio": np.zeros((len(selected), 512), np.float32),
            "acceleration": np.zeros((len(selected), 512), np.float32),
        }
        for waves_file, group in selected.groupby("waves_file"):
            cache = np.load(DATA / waves_file)
            keep = np.isin(cache["recording"], group.index_in_waves.to_numpy())
            owner = (
                pd.Series(group.index.to_numpy(), index=group.index_in_waves.to_numpy())
                .loc[cache["recording"][keep]]
                .to_numpy()
            )
            audio = torch.from_numpy(cache["audio"][keep].astype(np.float32))
            accel = torchaudio.functional.resample(
                torch.from_numpy(cache["accel"][keep].astype(np.float32)), 8000, 16000
            )
            for name, windows in [("audio", audio), ("acceleration", accel)]:
                per_window = embed_windows(clap_embedder, windows, 32)
                blocks[name][group.index] = pd.DataFrame(per_window).groupby(owner).mean().loc[group.index].to_numpy()
            print(waves_file, "embedded", flush=True)
        np.savez(path, **blocks)
    blocks = np.load(path)
    return selected, {name: blocks[name] for name in blocks.files}


def greatest_hits(max_per_family=300):
    table = pd.read_csv(DATA / "greatest_hits_recordings.csv")
    table = pd.concat([g.sample(min(len(g), max_per_family), random_state=0) for _, g in table.groupby("family")])
    table = table.reset_index(drop=True)
    path = OUT / "clap_greatest_hits.npy"
    if not path.exists():
        windows = []
        for _, row in table.iterrows():
            video = row.surface.rsplit(" ", 1)[0]
            rate, samples = wavfile.read(Path("data/greatest_hits") / f"{video}_denoised.wav", mmap=True)
            start = int(row.chunk_start * rate)
            chunk = torch.tensor(samples[start : start + rate // 2].astype(np.float32))
            if chunk.ndim > 1:
                chunk = chunk.mean(1)
            chunk = torchaudio.functional.resample(chunk, rate, 16000)[:8000]
            chunk = torch.nn.functional.pad(chunk, (0, 8000 - len(chunk)))
            windows.append(chunk / (chunk.pow(2).mean().sqrt() + 1e-8))
        np.save(path, embed_windows(clap_embedder, torch.stack(windows), 32))
    table["dataset"] = "greatest_hits"
    return table, np.load(path)


@torch.no_grad()
def family_text_embeddings():
    from transformers import ClapModel, ClapProcessor

    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(DEVICE).eval()
    out = []
    for family in FAMILIES:
        texts = [t.format(w) for t in TEMPLATES for w in WORDS[family]]
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(DEVICE)
        e = model.get_text_features(**inputs)
        e = e if torch.is_tensor(e) else e.pooler_output
        e = torch.nn.functional.normalize(e.float(), dim=-1).mean(0)
        out.append(torch.nn.functional.normalize(e, dim=0).cpu().numpy())
    return np.stack(out)


def predict(audio_embeddings, text, balanced):
    a = audio_embeddings / np.linalg.norm(audio_embeddings, axis=1, keepdims=True)
    cosine = a @ text.T
    if balanced:
        cosine = cosine - cosine.mean(0)
    return np.array(FAMILIES)[cosine.argmax(1)]


def main():
    text = family_text_embeddings()
    print("cosine between family prompts:\n", np.round(text @ text.T, 2))
    selected, ours = our_setups()
    hits, hits_embeddings = greatest_hits()
    rows = []
    cases = [
        (s, signal, ours[signal][(selected.dataset == s).to_numpy()], selected[selected.dataset == s])
        for s in ["cluster", "lmt108", "sens3"]
        for signal in ["audio", "acceleration"]
    ]
    cases.append(("greatest_hits", "audio", hits_embeddings, hits))
    for setup, signal, embeddings, table in cases:
        for balanced in [False, True]:
            predicted = predict(embeddings, text, balanced)
            result = score(table, predicted)
            shares = pd.Series(predicted).value_counts(normalize=True)
            rows.append(
                {
                    "setup": setup,
                    "signal": signal,
                    "readout": "prior-balanced" if balanced else "raw",
                    **result,
                    "most frequent answer": shares.index[0],
                    "its share": shares.iloc[0],
                }
            )
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "scores.csv", index=False)
    pd.set_option("display.width", 200)
    print(
        scores[["setup", "signal", "readout", "balanced accuracy", "low", "high", "most frequent answer", "its share"]]
        .round(2)
        .to_string()
    )
    plot(scores)


def plot(scores):
    labels = [f"{r.setup}, {r.signal}" for r in scores[scores.readout == "raw"].itertuples()]
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(labels))
    for offset, (readout, color) in zip([-0.2, 0.2], [("raw", "#2a78d6"), ("prior-balanced", "#eb6834")]):
        part = scores[scores.readout == readout]
        ax.barh(y + offset, part["balanced accuracy"], height=0.38, color=color, label=readout)
        ax.errorbar(
            part["balanced accuracy"],
            y + offset,
            xerr=[part["balanced accuracy"] - part.low, part.high - part["balanced accuracy"]],
            fmt="none",
            ecolor="0.3",
            linewidth=1,
        )
    ax.axvline(0.25, color="0.4", linestyle=":", linewidth=1)
    ax.set_yticks(y, labels)
    ax.set(
        xlim=(0, 1),
        xlabel="balanced accuracy (4 families, chance 0.25)",
        title="H27: CLAP asked with text, no training on any setup",
    )
    ax.grid(axis="x", color="0.9")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "clap_text_zero_shot.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()

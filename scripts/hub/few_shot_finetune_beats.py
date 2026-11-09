"""H15: fine-tune BEATs on the other two setups + k labeled surfaces of the new setup.

See docs/experiments.md (H15). Same labeled-surface draws as H7
(fusion_few_shot_probes.py), drawn from the recordings of selected_recordings.csv, so the scores
compare with the frozen probes draw by draw. Test: the other selected recordings of the new setup.

Runs from the project folder (locally or on the hub):
    python scripts/hub/few_shot_finetune_beats.py [--quick]
Output: analysis/classification/few_shot_finetune/scores.csv (finished runs are skipped).
"""

import gc
import sys
import time
from pathlib import Path

import augmented_beats_transfer
import numpy as np
import pandas as pd
import torch
from augmented_beats_transfer import FAMILIES, SETUPS, classify, load_data, score
from augmented_beats_transfer import train as train_beats

QUICK = "--quick" in sys.argv
augmented_beats_transfer.N_UPDATES = 20 if QUICK else 1500
augmented_beats_transfer.BATCH = 32  # fits the 8 GB RTX 2070; 1500 x 32 = same examples as 750 x 64
K_VALUES = [2, 4]
DRAWS = [0, 1]
OUT = Path("analysis/classification/few_shot_finetune")
OUT.mkdir(parents=True, exist_ok=True)
SCORES = OUT / f"scores{'_quick' if QUICK else ''}.csv"


def labeled_surfaces(table, k, rng):
    surfaces = table[["surface", "family"]].drop_duplicates()
    return np.concatenate([rng.choice(surfaces.surface[surfaces.family == f], k, replace=False) for f in FAMILIES])


def main():
    table, waves, recording_of_window = load_data()
    selected = pd.read_csv("analysis/classification/qwen_audio/selected_recordings.csv", dtype={"surface": str})
    selected["surface"] = selected.dataset + ":" + selected.surface
    offset = {"waveforms.npz": 0, "sens3_waveforms.npz": (table.dataset != "sens3").sum()}
    selected["recording"] = (selected.index_in_waves + selected.waves_file.map(offset)).to_numpy()

    done = set()
    if SCORES.exists():
        done = set(pd.read_csv(SCORES)[["new setup", "k", "draw"]].itertuples(index=False, name=None))

    for new_setup in SETUPS:
        target = selected[selected.dataset == new_setup]
        for k in K_VALUES:
            for draw in DRAWS:
                if (new_setup, k, draw) in done:
                    continue
                started = time.time()
                rng = np.random.default_rng([SETUPS.index(new_setup), k, draw])
                labeled = labeled_surfaces(target, k, rng)
                labeled_recordings = table.index[table.surface.isin(labeled)].to_numpy()
                other_setups = table.index[table.dataset != new_setup].to_numpy()
                train_recordings = np.sort(np.concatenate([other_setups, labeled_recordings]))
                test_recordings = target.recording[~target.surface.isin(labeled)].to_numpy()
                if QUICK:
                    keep = table.groupby("surface").head(3).index
                    train_recordings = np.intersect1d(train_recordings, keep)
                    test_recordings = np.intersect1d(test_recordings, keep)
                model = train_beats(table, waves, recording_of_window, train_recordings, ["audio", "accel"], 0.0)
                predicted = classify(model, waves, recording_of_window, test_recordings)
                row = {
                    "new setup": new_setup,
                    "k": k,
                    "draw": draw,
                    "labeled surfaces": " ".join(labeled),
                    **score(table.loc[predicted.index], predicted.to_numpy(), n_draws=200),
                }
                pd.DataFrame([row]).to_csv(SCORES, mode="a", header=not SCORES.exists(), index=False)
                print(
                    f"{new_setup}, k={k}, draw {draw}: {row['balanced accuracy']:.2f} ({time.time() - started:.0f} s)",
                    flush=True,
                )
                del model
                gc.collect()
                torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

EXPERIMENTS = {
    "spectral_shape_models": ("baselines", "spectral"),
    "extract_spectrograms": ("prepare", "spectrograms"),
    "cnn_on_cluster": ("baselines", "cnn_cluster"),
    "cnn_on_both_datasets": ("baselines", "cnn_combined"),
    "spectral_models_on_team_splits": ("baselines", "spectral_team"),
    "cnn_on_team_splits": ("baselines", "cnn_team"),
    "extract_sens3": ("prepare", "sens3"),
    "test_on_sens3": ("transfer", "sens3"),
    "domain_shift_variants": ("variants", "domain_shift"),
    "invariance_variants": ("variants", "invariance"),
    "extract_greatest_hits": ("prepare", "greatest_hits"),
    "leave_one_setup_out": ("transfer", "leave_one_out"),
    "setup_filter": ("diagnostics", "filter"),
    "setup_centering": ("diagnostics", "centering"),
    "invariant_features": ("diagnostics", "invariant_features"),
    "few_shot_calibration": ("few_shot_calibration", "run"),
    "export_waveforms": ("prepare", "waveforms"),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", nargs="?", default="all", choices=["all", "bundle", *EXPERIMENTS])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.experiment == "all" or (
        args.experiment in {"spectral_models_on_team_splits", "cnn_on_team_splits"}
        and not (root / "analysis/splits/folds.csv").exists()
    ):
        subprocess.run([sys.executable, str(root / "scripts/splits.py")], check=True)
    if args.experiment == "all":
        for experiment in EXPERIMENTS:
            print(experiment, flush=True)
            subprocess.run([sys.executable, str(Path(__file__).resolve()), experiment], check=True)
    elif args.experiment == "bundle":
        importlib.import_module("prepare").bundle()
    else:
        module, function = EXPERIMENTS[args.experiment]
        getattr(importlib.import_module(module), function)()

#!/usr/bin/env bash
# Reproduce the whole signal-processing analysis; figures and tables go to analysis/signal_processing/.
# extract_spectra.py takes ~6 minutes (16 cores) and writes the cache; the other scripts take seconds,
# except spectra_embeddings.py (~5 minutes).
set -euo pipefail
cd "$(dirname "$0")/../.."
uv run python scripts/surface_families.py
for name in extract_spectra plot_raw_signals bandwidth_and_noise machine_lines variance_by_factor \
            velocity_effect family_spectra family_descriptors audio_accel_coherence \
            spectra_pca spectra_embeddings surface_clustering machine_noise_nmf; do
    echo "=== $name"
    uv run python "scripts/signal_processing/$name.py"
done

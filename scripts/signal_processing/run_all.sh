#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
uv run python scripts/surface_families.py
for name in extract_spectra plot_raw_signals bandwidth_and_noise machine_lines variance_by_factor \
            velocity_effect family_spectra family_descriptors audio_accel_coherence \
            spectra_pca spectra_embeddings surface_clustering machine_noise_nmf; do
    echo "=== $name"
    uv run python "scripts/signal_processing/$name.py"
done

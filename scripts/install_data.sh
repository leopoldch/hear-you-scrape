#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/install_data.sh [all|cluster|lmt108|sens3]
cd "$(dirname "$0")/.."
dataset="${1:-all}"
mkdir -p data

case "$dataset" in
    all|cluster|lmt108|sens3) ;;
    *) echo "Unknown dataset: $dataset" >&2; exit 1 ;;
esac

download() {
    local url="$1" archive="$2"
    local status
    if [[ ! -f "$archive" ]]; then
        echo "Downloading $archive..."
        status=$(curl -L --fail --retry 2 --retry-max-time 60 --connect-timeout 30 \
            -C - -o "$archive.part" -w '%{http_code}' "$url") || return 1
        if [[ "$status" != 200 && "$status" != 206 ]]; then
            echo "Download failed (HTTP $status): $url" >&2
            return 1
        fi
        mv "$archive.part" "$archive"
    fi
}

install_dataset() {
    local name="$1" url="$2"
    shift 2
    if [[ ! -f "data/$name/.installed" ]]; then
        download "$url" "data/$name.zip"
        mkdir -p "data/$name"
        echo "Extracting $name..."
        unzip -q -o "data/$name.zip" "$@" -d "data/$name"
        rm "data/$name.zip"
        touch "data/$name/.installed"
    fi
}

# Source: https://doi.org/10.6084/m9.figshare.29438288
if [[ "$dataset" == all || "$dataset" == cluster ]]; then
    install_dataset cluster "https://ndownloader.figshare.com/files/61118716" \
        '*/sensor_data/audio/*' '*/sensor_data/accel/*' '*/texture_list.xlsx' '*/README.md'
fi

# Source: https://zeus.lkn.ei.tum.de/downloads/texture/
if [[ "$dataset" == all || "$dataset" == lmt108 ]]; then
    install_dataset lmt108 "https://zeus.lkn.ei.tum.de/downloads/texture/download/LMT_108_SurfaceMaterials_Database_V1.1.zip" \
        '*/AccelScansComponents/Movement/*' '*/SoundScans/Movement/*'
fi

# Source: https://www.sens3.net/surfaces (files hosted on 4TU.ResearchData)
# The list of the 50 MaterialN.zip archives comes from the 4TU API.
# Keep sliding files only; delete each ZIP after extraction.
SENS3_FILES_API="https://data.4tu.nl/v2/articles/e7f8f6dd-b61c-42fc-bea9-5ff103cbe396/files"
if [[ "$dataset" == all || "$dataset" == sens3 ]]; then
    mkdir -p data/sens3
    file_list=$(curl -s --fail "$SENS3_FILES_API" | python3 -c '
import json, sys
for archive in json.load(sys.stdin):
    if archive["name"].startswith("Material"):
        print(archive["name"], archive["download_url"])
')
    while read -r name url; do
        [[ -f "data/sens3/$name.installed" ]] && continue
        download "$url" "data/sens3/$name"
        echo "Extracting $name..."
        unzip -q -o "data/sens3/$name" '*Material_sensor_*.csv' '*Material_*.wav' -d data/sens3
        rm "data/sens3/$name"
        touch "data/sens3/$name.installed"
    done <<< "$file_list"
fi

#!/usr/bin/env bash
# Download LoveDA dataset (Train + Val splits) from Zenodo.
# Test split omitted: labels are held out for the LoveDA benchmark server,
# so it's useless for our training/eval loop.
#
# Source: https://zenodo.org/records/5706578
# Layout after unzip:
#   data/loveda/
#     Train/
#       Urban/
#         images_png/   (1156 PNGs, 1024x1024 RGB)
#         masks_png/    (1156 PNGs, 1024x1024 single-channel, values 0-7)
#       Rural/
#         images_png/   (1366 PNGs)
#         masks_png/    (1366 PNGs)
#     Val/
#       Urban/
#         images_png/   (677 PNGs)
#         masks_png/    (677 PNGs)
#       Rural/
#         images_png/   (992 PNGs)
#         masks_png/    (992 PNGs)
#
# Class values in masks:
#   0 = no-data  (ignore in loss, ignore in stats)
#   1 = background  (real class, trained but excluded from emissions)
#   2 = building
#   3 = road
#   4 = water
#   5 = barren
#   6 = forest
#   7 = agriculture

set -euo pipefail

# Resolve data dir relative to repo root (script lives in scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." &> /dev/null && pwd )"
DATA_DIR="${REPO_ROOT}/data/loveda"

mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

# MD5s from torchgeo's LoveDA loader (which in turn got them from Zenodo).
declare -A MD5S=(
    ["Train.zip"]="de2b196043ed9b4af1690b3f9a7d558f"
    ["Val.zip"]="84cae2577468ff0b5386758bb386d31d"
)

# URLs on Zenodo
declare -A URLS=(
    ["Train.zip"]="https://zenodo.org/records/5706578/files/Train.zip?download=1"
    ["Val.zip"]="https://zenodo.org/records/5706578/files/Val.zip?download=1"
)

download_and_verify() {
    local filename="$1"
    local expected_md5="${MD5S[$filename]}"
    local url="${URLS[$filename]}"

    if [[ -f "${filename}" ]]; then
        echo "[${filename}] exists, checking MD5..."
        local actual_md5
        actual_md5="$(md5sum "${filename}" | awk '{print $1}')"
        if [[ "${actual_md5}" == "${expected_md5}" ]]; then
            echo "[${filename}] MD5 OK, skipping download"
            return 0
        else
            echo "[${filename}] MD5 mismatch (got ${actual_md5}, want ${expected_md5})"
            echo "[${filename}] removing and re-downloading"
            rm -f "${filename}"
        fi
    fi

    echo "[${filename}] downloading from Zenodo..."
    # -c: resume partial downloads. --tries=3 in case Zenodo hiccups.
    wget -c --tries=3 -O "${filename}" "${url}"

    echo "[${filename}] verifying MD5..."
    local actual_md5
    actual_md5="$(md5sum "${filename}" | awk '{print $1}')"
    if [[ "${actual_md5}" != "${expected_md5}" ]]; then
        echo "ERROR: [${filename}] MD5 mismatch after download"
        echo "  got:  ${actual_md5}"
        echo "  want: ${expected_md5}"
        exit 1
    fi
    echo "[${filename}] MD5 OK"
}

for f in Train.zip Val.zip; do
    download_and_verify "$f"
done

# Unzip. The Zenodo zips contain a top-level folder matching the zip name
# (Train/ and Val/), each with Urban/ and Rural/ subdirs containing
# images_png/ and masks_png/. So we just unzip in place.
for f in Train.zip Val.zip; do
    dirname="${f%.zip}"
    if [[ -d "${dirname}" ]]; then
        echo "[${f}] ${dirname}/ already unzipped, skipping"
    else
        echo "[${f}] unzipping..."
        unzip -q "${f}"
    fi
done

# Sanity check: count files. These counts are from the LoveDA paper /
# torchgeo. If they don't match, the unzip is broken.
expected_counts() {
    python3 - <<'PY'
import os, sys
from pathlib import Path

base = Path(".").resolve()
expected = {
    "Train/Urban/images_png": 1156,
    "Train/Urban/masks_png":  1156,
    "Train/Rural/images_png": 1366,
    "Train/Rural/masks_png":  1366,
    "Val/Urban/images_png":    677,
    "Val/Urban/masks_png":     677,
    "Val/Rural/images_png":    992,
    "Val/Rural/masks_png":     992,
}
ok = True
for rel, want in expected.items():
    p = base / rel
    if not p.is_dir():
        print(f"  MISSING: {rel}")
        ok = False
        continue
    got = sum(1 for f in p.iterdir() if f.suffix.lower() == ".png")
    flag = "OK" if got == want else "MISMATCH"
    print(f"  {flag:8s} {rel}: {got} (expected {want})")
    if got != want:
        ok = False
sys.exit(0 if ok else 1)
PY
}

echo
echo "=== File count check ==="
if expected_counts; then
    echo "All counts match. LoveDA ready at: ${DATA_DIR}"
else
    echo "ERROR: file counts don't match expected LoveDA layout."
    exit 1
fi

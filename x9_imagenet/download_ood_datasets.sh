#!/usr/bin/env bash
# Download the 8-set OOD suite for the ImageNet-scale experiments
# (documentation/imagenet_scale_plan.md, Phase 0).
#
# Sources:
#   - OpenOOD processed copies (Google Drive via gdown) for the five sets of
#     the OpenOOD v1.5 ImageNet benchmark (ssb_hard, ninco, inaturalist,
#     texture, openimage_o) plus imagenet_o and their benchmark imglists,
#     so the G1 gate can evaluate on OpenOOD's exact image lists.
#   - MOS subsets (direct HTTP, pages.cs.wisc.edu) for SUN and Places:
#     the canonical 10k ImageNet-OOD subsets; OpenOOD's places365 copy is
#     their CIFAR-benchmark version, not this.
#
# Fallback URLs if gdown hits Google Drive quota ("Too many users have
# viewed or downloaded this file"): retry later, or fetch the originals and
# accept a different directory layout (then the imglist paths need mapping):
#   NINCO      https://zenodo.org/record/8013288/files/NINCO_all.tar.gz
#   Textures   https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
#   ImageNet-O https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar
#   (ssb_hard and openimage_o have no clean direct source; retry gdown.)
#
# Usage:
#   DATASET_ROOT_DIR=/work/... bash x9_imagenet/download_ood_datasets.sh
set -euo pipefail

ROOT="${DATASET_ROOT_DIR:?set DATASET_ROOT_DIR}/ood_suite"
mkdir -p "$ROOT"
cd "$ROOT"
command -v gdown >/dev/null 2>&1 || python -m pip install --user gdown

# name  gdrive_id  (OpenOOD scripts/download/download.py, fetched 2026-08-05)
OPENOOD_SETS="
ssb_hard 1PzkA-WGG8Z18h0ooL_pDdz9cO-DCIouE
ninco 1Z82cmvIB0eghTehxOGP5VTdLt7OD3nk6
inaturalist 1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj
texture 1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam
openimage_o 1VUFXnB_z70uHfdgJG2E_pjYOcEgqM7tE
imagenet_o 1S9cFV7fGvJCcka220-pIO9JPZL1p1V8w
benchmark_imglist 1lI1j0_fDDvjIt9JlWAw09X8ks-yrR_H1
"

echo "$OPENOOD_SETS" | while read -r name gid; do
    [ -z "${name:-}" ] && continue
    # skip only if the directory exists AND contains files (an empty dir is a
    # leftover from a failed attempt and must be retried)
    if [ -d "$name" ] && [ -n "$(find "$name" -type f -print -quit 2>/dev/null)" ]; then
        echo "[skip] $name already present"
        continue
    fi
    rm -rf "$name" "${name}.zip"
    echo "[gdown] $name"
    gdown "$gid" -O "${name}.zip"
    # a Drive quota/confirm page saved as .zip fails this integrity test
    if ! unzip -tq "${name}.zip" >/dev/null 2>&1; then
        echo "ERROR: ${name}.zip is not a valid zip (Google Drive quota page?)."
        echo "       Retry later or use the fallback URLs in the header."
        rm -f "${name}.zip"
        exit 1
    fi
    mkdir -p "$name"
    unzip -q "${name}.zip" -d "$name"
    rm -f "${name}.zip"
done

# MOS 10k subsets (direct HTTP)
for pair in "sun SUN" "places Places"; do
    set -- $pair
    dst="$1"; tarname="$2"
    if [ -d "$dst" ]; then
        echo "[skip] $dst already present"
        continue
    fi
    echo "[wget] $dst"
    wget -q --show-progress \
        "https://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/${tarname}.tar.gz"
    mkdir -p "$dst"
    tar -xzf "${tarname}.tar.gz" -C "$dst" --strip-components=1
    rm -f "${tarname}.tar.gz"
done

echo
echo "=== image counts per set (expected in parentheses) ==="
expected="ssb_hard:49000 ninco:5879 inaturalist:10000 texture:5640 openimage_o:17632 imagenet_o:2000 sun:10000 places:10000"
for kv in $expected; do
    name="${kv%%:*}"; exp="${kv##*:}"
    if [ -d "$name" ]; then
        n=$(find "$name" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l | tr -d ' ')
        flag=""
        [ "$n" != "$exp" ] && flag="  <-- differs from expected; check whether the zip carries extras beyond the imglist"
        echo "$name: $n ($exp)$flag"
    else
        echo "$name: MISSING"
    fi
done
echo
echo "imglists (for the OpenOOD-suite sets, evaluation uses these, not raw folder contents):"
# the zip carries its own top-level benchmark_imglist/ folder; OOD test lists
# live in the imagenet/ subdir without 'imagenet' in their filenames
if [ -d benchmark_imglist/benchmark_imglist/imagenet ]; then
    wc -l benchmark_imglist/benchmark_imglist/imagenet/test_*.txt
fi
echo "done."

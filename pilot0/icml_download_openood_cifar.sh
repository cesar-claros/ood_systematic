#!/usr/bin/env bash
# ICML roster-A step-0 downloads: the OpenOOD CIFAR-side data that the
# stage-3 script never needed (retarget protocol section 8.2).
#
# Fetches into the layout the frozen extractors and clip_severity_v2.py
# expect:
#     $DATASET_ROOT_DIR/openood/data/benchmark_imglist/...   (if absent)
#     $DATASET_ROOT_DIR/openood/data/images_classic/{cifar10,cifar100,
#         tin,mnist,svhn,places365,texture}/...
#     $DATASET_ROOT_DIR/openood/results/cifar10_res18_v1.5 unpack
#     $DATASET_ROOT_DIR/openood/results/cifar100_res18_v1.5 unpack
# Google Drive IDs verified 2026-08-31 against OpenOOD's own
# scripts/download/download.py (texture and benchmark_imglist IDs match
# the committed stage-3 script).
#
# Idempotent: items already on disk are skipped; rerun after partial
# failures (Google Drive throttling of large gdown pulls is common).
# A verification report prints at the end, including the unpacked
# checkpoint dir names to pass to icml_roster_a_enumerate.py.
#
# Usage:  nohup bash pilot0/icml_download_openood_cifar.sh > icml_download.log 2>&1 &

set -u

: "${DATASET_ROOT_DIR:?DATASET_ROOT_DIR is not set}"
command -v gdown >/dev/null || { echo "gdown not found: pip install gdown"; exit 1; }
command -v unzip >/dev/null || { echo "unzip not found"; exit 1; }

OODROOT="$DATASET_ROOT_DIR/openood"
DATA="$OODROOT/data"
TMP="$OODROOT/tmp"
mkdir -p "$DATA/images_classic" "$OODROOT/results" "$TMP"

FAILURES=0

# fetch <gdrive_id> <name> <destroot>  (same placement logic as the
# stage-3 script: single top dir moved/renamed; bare or structural
# content wrapped under <name>/)
fetch () {
  local id="$1" name="$2" destroot="$3"
  if [ -e "$destroot/$name" ] && [ -n "$(ls -A "$destroot/$name" 2>/dev/null)" ]; then
    echo "[skip] $name already present at $destroot/$name"
    return 0
  fi
  echo "[fetch] $name"
  local zip="$TMP/$name.zip" unz="$TMP/unz_$name"
  rm -rf "$unz"; mkdir -p "$unz"
  if ! gdown "$id" -O "$zip"; then
    echo "[FAIL] gdown $name (Google Drive throttling? rerun later)"
    FAILURES=$((FAILURES+1)); return 1
  fi
  if ! unzip -q "$zip" -d "$unz"; then
    echo "[FAIL] unzip $name"; rm -f "$zip"; FAILURES=$((FAILURES+1)); return 1
  fi
  local entries; entries=$(ls -A "$unz")
  local count; count=$(echo "$entries" | wc -l)
  if [ "$count" -eq 1 ] && [ -d "$unz/$entries" ] && [ "$entries" = "$name" ]; then
    mv "$unz/$entries" "$destroot/$name"
  elif [ "$count" -eq 1 ] && [ -d "$unz/$entries" ] \
       && ! echo "$entries" | grep -qxE 'images|train|val|test|data'; then
    echo "[note] $name zip unpacks as '$entries'; placing it as '$name'"
    mv "$unz/$entries" "$destroot/$name"
  else
    mkdir -p "$destroot/$name"
    mv "$unz"/* "$destroot/$name/"
  fi
  rm -rf "$zip" "$unz"
  echo "[done] $name -> $destroot/$name"
}

# --- imglists (usually already present from stage 3) ---
if [ -f "$DATA/benchmark_imglist/cifar10/train_cifar10.txt" ]; then
  echo "[skip] benchmark_imglist already present"
else
  echo "[fetch] benchmark_imglist"
  if gdown 1lI1j0_fDDvjIt9JlWAw09X8ks-yrR_H1 -O "$TMP/imglist.zip" \
     && unzip -q "$TMP/imglist.zip" -d "$DATA/"; then
    rm -f "$TMP/imglist.zip"; echo "[done] benchmark_imglist"
  else
    echo "[FAIL] benchmark_imglist"; FAILURES=$((FAILURES+1))
  fi
fi

# --- images_classic datasets for the CIFAR benchmarks ---
fetch 1Co32RiiWe16lTaiOU6JMMnyUYS41IlO1 cifar10   "$DATA/images_classic"
fetch 1PGKheHUsf29leJPPGuXqzLBMwl8qMF8_ cifar100  "$DATA/images_classic"
fetch 1PZ-ixyx52U989IKsMA2OT-24fToTrelC tin       "$DATA/images_classic"
fetch 1CCHAGWqA1KJTFFswuF9cbhmB-j98Y1Sb mnist     "$DATA/images_classic"
fetch 1DQfc11HOtB1nEwqS4pWUFp8vtQ3DczvI svhn      "$DATA/images_classic"
fetch 1Ec-LRSTf6u5vEctKX9vRp9OA6tqnJ0Ay places365 "$DATA/images_classic"
fetch 1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam texture   "$DATA/images_classic"

# --- released ResNet-18 v1.5 checkpoint bundles ---
fetch_ckpts () {
  local id="$1" name="$2" pattern="$3"
  if find "$OODROOT/results" -maxdepth 1 -type d -name "$pattern" | grep -q .; then
    echo "[skip] $name already unpacked under results/ ($pattern)"
    return 0
  fi
  echo "[fetch] $name checkpoints"
  local zip="$TMP/$name.zip"
  if gdown "$id" -O "$zip" && unzip -q "$zip" -d "$OODROOT/results/"; then
    rm -f "$zip"; echo "[done] $name"
  else
    echo "[FAIL] $name"; FAILURES=$((FAILURES+1))
  fi
}
fetch_ckpts 1byGeYxM_PlLjT72wZsMQvP6popJeWBgt cifar10_res18_v1.5  'cifar10_*'
fetch_ckpts 1s-1oNrRtmA0pGefxXJOUVRYpaoAML0C- cifar100_res18_v1.5 'cifar100_*'

# --- verification report ---
echo
echo "=== ICML roster-A download verification ==="
check_imglist_and_image () {
  local rel="$1"
  if [ ! -f "$DATA/$rel" ]; then
    echo "MISSING imglist $rel"; FAILURES=$((FAILURES+1)); return
  fi
  local img
  img=$(head -1 "$DATA/$rel" | awk '{print $1}')
  if [ -f "$DATA/images_classic/$img" ]; then
    echo "OK  $rel (first image present)"
  else
    echo "MISSING images for $rel (first ref: images_classic/$img)"
    FAILURES=$((FAILURES+1))
  fi
}
for src in cifar10 cifar100; do
  check_imglist_and_image "benchmark_imglist/$src/train_$src.txt"
  check_imglist_and_image "benchmark_imglist/$src/test_$src.txt"
  for ood in cifar100 cifar10 tin mnist svhn texture places365; do
    [ "$ood" = "$src" ] && continue
    check_imglist_and_image "benchmark_imglist/$src/test_$ood.txt"
  done
done
echo
echo "checkpoint dirs for icml_roster_a_enumerate.py:"
find "$OODROOT/results" -maxdepth 1 -type d \( -name 'cifar10_*' -o -name 'cifar100_*' \) -print
NC10=$(find "$OODROOT/results"/cifar10_* -name 'best*.ckpt' 2>/dev/null | wc -l)
NC100=$(find "$OODROOT/results"/cifar100_* -name 'best*.ckpt' 2>/dev/null | wc -l)
echo "best*.ckpt found: cifar10=$NC10 cifar100=$NC100"
{ [ "$NC10" -ge 1 ] && [ "$NC100" -ge 1 ]; } || FAILURES=$((FAILURES+1))
echo "failures: $FAILURES"
[ "$FAILURES" -eq 0 ] && echo "ROSTER-A STEP 0 COMPLETE" || echo "ROSTER-A STEP 0 INCOMPLETE: rerun this script (idempotent) after resolving the failures above"
exit "$FAILURES"

#!/usr/bin/env bash
# Stage-3 OpenOOD ImageNet-200 pilot: step-0 downloads (protocol stage 3).
#
# Downloads the OpenOOD benchmark imglists, the three released ImageNet-200
# ResNet-18 v1.5 checkpoints, the ID val/test images (imagenet_1k bundle),
# and the five OOD sets (ssb_hard, ninco, inaturalist, openimage_o,
# texture) into the layout the frozen extractor expects:
#     $DATASET_ROOT_DIR/openood/data/benchmark_imglist/...
#     $DATASET_ROOT_DIR/openood/data/images_largescale/<set>/...
#     $DATASET_ROOT_DIR/openood/data/images_classic/texture/...
#     $DATASET_ROOT_DIR/openood/results/...
# Google Drive IDs are from OpenOOD's own scripts/download/download.py.
#
# Idempotent: items already on disk are skipped, so rerun after partial
# failures (Google Drive throttling of large gdown pulls is common; just
# rerun later). A verification report prints at the end.
#
# Usage:  nohup bash pilot0/stage3_download_openood.sh > stage3_download.log 2>&1 &

set -u

: "${DATASET_ROOT_DIR:?DATASET_ROOT_DIR is not set}"
command -v gdown >/dev/null || { echo "gdown not found: pip install gdown"; exit 1; }
command -v unzip >/dev/null || { echo "unzip not found"; exit 1; }

OODROOT="$DATASET_ROOT_DIR/openood"
DATA="$OODROOT/data"
TMP="$OODROOT/tmp"
mkdir -p "$DATA/images_largescale" "$DATA/images_classic" "$OODROOT/results" "$TMP"

FAILURES=0

# fetch <gdrive_id> <name> <destroot>
# Unzips into a scratch dir, then places the content at <destroot>/<name>:
# a single top-level dir is moved (renamed to <name> if it differs); bare
# files are wrapped into <name>/.
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
  if [ "$count" -eq 1 ] && [ -d "$unz/$entries" ]; then
    if [ "$entries" != "$name" ]; then
      echo "[note] $name zip unpacks as '$entries'; placing it as '$name'"
    fi
    mv "$unz/$entries" "$destroot/$name"
  else
    mkdir -p "$destroot/$name"
    mv "$unz"/* "$destroot/$name/"
  fi
  rm -rf "$zip" "$unz"
  echo "[done] $name -> $destroot/$name"
}

# --- imglists (unzips directly into data/: the zip carries benchmark_imglist/) ---
if [ -f "$DATA/benchmark_imglist/imagenet200/train_imagenet200.txt" ]; then
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

# --- the three ResNet-18 v1.5 checkpoint runs ---
if ls "$OODROOT/results"/**/best*.ckpt >/dev/null 2>&1 || ls "$OODROOT/results"/*/*/best*.ckpt >/dev/null 2>&1; then
  echo "[skip] checkpoints already present under results/"
else
  echo "[fetch] imagenet200_res18_v1.5 checkpoints"
  if gdown 1ddVmwc8zmzSjdLUO84EuV4Gz1c7vhIAs -O "$TMP/ckpts.zip" \
     && unzip -q "$TMP/ckpts.zip" -d "$OODROOT/results/"; then
    rm -f "$TMP/ckpts.zip"; echo "[done] checkpoints"
  else
    echo "[FAIL] checkpoints"; FAILURES=$((FAILURES+1))
  fi
fi

# --- ID val images (special-cased: the train materializer also writes into
# imagenet_1k/, so presence of the DIRECTORY must not skip the val bundle;
# the marker is imagenet_1k/val, and the unpack MERGES into the dir) ---
if [ -d "$DATA/images_largescale/imagenet_1k/val" ] && [ -n "$(ls -A "$DATA/images_largescale/imagenet_1k/val" 2>/dev/null)" ]; then
  echo "[skip] imagenet_1k/val already present"
else
  echo "[fetch] imagenet_1k (val images)"
  if gdown 1i1ipLDFARR-JZ9argXd2-0a6DXwVhXEj -O "$TMP/imagenet_1k.zip" \
     && rm -rf "$TMP/unz_imagenet_1k" && mkdir -p "$TMP/unz_imagenet_1k" \
     && unzip -q "$TMP/imagenet_1k.zip" -d "$TMP/unz_imagenet_1k"; then
    VALDIR=$(find "$TMP/unz_imagenet_1k" -maxdepth 3 -type d -name val | head -1)
    if [ -n "$VALDIR" ]; then
      mkdir -p "$DATA/images_largescale/imagenet_1k"
      mv "$VALDIR" "$DATA/images_largescale/imagenet_1k/val"
      echo "[done] imagenet_1k/val"
    else
      echo "[FAIL] imagenet_1k zip contains no val/ dir; inspect $TMP/unz_imagenet_1k"
      FAILURES=$((FAILURES+1))
    fi
    rm -f "$TMP/imagenet_1k.zip"
    find "$TMP/unz_imagenet_1k" -type d -empty -delete 2>/dev/null
  else
    echo "[FAIL] imagenet_1k"; FAILURES=$((FAILURES+1))
  fi
fi

# --- OOD sets (dir names must match the imglist path prefixes) ---
fetch 1PzkA-WGG8Z18h0ooL_pDdz9cO-DCIouE ssb_hard    "$DATA/images_largescale"
fetch 1Z82cmvIB0eghTehxOGP5VTdLt7OD3nk6 ninco       "$DATA/images_largescale"
fetch 1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj inaturalist "$DATA/images_largescale"
fetch 1VUFXnB_z70uHfdgJG2E_pjYOcEgqM7tE openimage_o "$DATA/images_largescale"
fetch 1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam texture     "$DATA/images_classic"

# --- verification report ---
echo
echo "=== stage-3 download verification ==="
for f in benchmark_imglist/imagenet200/train_imagenet200.txt \
         benchmark_imglist/imagenet200/test_imagenet200.txt \
         benchmark_imglist/imagenet/test_ssb_hard.txt \
         benchmark_imglist/imagenet/test_ninco.txt \
         benchmark_imglist/imagenet/test_textures.txt \
         benchmark_imglist/imagenet/test_inaturalist.txt \
         benchmark_imglist/imagenet/test_openimage_o.txt; do
  if [ -f "$DATA/$f" ]; then echo "OK  imglist $f"; else echo "MISSING imglist $f"; FAILURES=$((FAILURES+1)); fi
done
NCKPT=$(find "$OODROOT/results" -name 'best*.ckpt' 2>/dev/null | sed 's|/[^/]*$||' | sort -u | wc -l)
echo "checkpoint run dirs with best*.ckpt: $NCKPT (expect 3)"
[ "$NCKPT" -ge 3 ] || FAILURES=$((FAILURES+1))
for d in images_largescale/imagenet_1k/val images_largescale/ssb_hard \
         images_largescale/ninco images_largescale/inaturalist \
         images_largescale/openimage_o images_classic/texture; do
  if [ -d "$DATA/$d" ] && [ -n "$(ls -A "$DATA/$d")" ]; then
    echo "OK  dataset $d ($(du -sh "$DATA/$d" 2>/dev/null | cut -f1))"
  else
    echo "MISSING dataset $d"; FAILURES=$((FAILURES+1))
  fi
done
echo "failures: $FAILURES"
[ "$FAILURES" -eq 0 ] && echo "STEP 0 COMPLETE" || echo "STEP 0 INCOMPLETE: rerun this script (idempotent) after resolving the failures above"
exit "$FAILURES"

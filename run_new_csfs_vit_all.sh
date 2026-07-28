#!/bin/bash
# Sequential ViT new-CSF sweep over all four sources on ONE GPU.
#
# Wraps run_new_csfs_pilot.sh once per source with the paper's Table-6 OOD
# suite (derived from clip_scores/clip_distances_<source>.csv) and the
# matching pilot_<source>_vit.txt experiment list. Arms default to `newcsf`
# only (MahalanobisPP + NCI): ASH and ReAct are not run on ViT because their
# activation-shaping assumptions (post-ReLU non-negative sparse features) are
# CNN-specific; export ARMS explicitly if that decision changes.
#
#   CSF_NUM_WORKERS=12 CSF_BATCH_SIZE=128 \
#     nohup bash run_new_csfs_vit_all.sh > new_csfs_vit_all.log 2>&1 &
#
# CSF_BATCH_SIZE / CSF_NUM_WORKERS / TORCH_HOME handling are inherited from
# run_new_csfs_pilot.sh (which redirects TORCH_HOME to a writable cache for
# the timm ViT backbone download when the container cache is read-only).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ARMS="${ARMS:-newcsf}"
export ARMS

declare -A MODES=(
  [cifar10]="iid_test ood_sncs_c100 ood_nsncs_ti ood_nsncs_isun ood_nsncs_lsun_resize ood_nsncs_lsun_cropped ood_nsncs_svhn ood_nsncs_places365 ood_nsncs_textures"
  [cifar100]="iid_test ood_sncs_c10 ood_nsncs_ti ood_nsncs_isun ood_nsncs_lsun_resize ood_nsncs_lsun_cropped ood_nsncs_svhn ood_nsncs_places365 ood_nsncs_textures"
  [supercifar]="iid_test ood_sncs_c10 ood_nsncs_ti ood_nsncs_isun ood_nsncs_lsun_resize ood_nsncs_lsun_cropped ood_nsncs_svhn ood_nsncs_places365 ood_nsncs_textures"
  [tinyimagenet]="iid_test ood_sncs_c10 ood_sncs_c100 ood_nsncs_isun ood_nsncs_lsun_resize ood_nsncs_lsun_cropped ood_nsncs_svhn ood_nsncs_places365 ood_nsncs_textures"
)

resolve_list() {
    # accept pilot_<key>_vit.txt with either the short or the full source name
    local key="$1"
    for cand in "pilot_${key}_vit.txt" "pilot_${key}100_vit.txt"; do
        if [ -f "$cand" ]; then echo "$cand"; return 0; fi
    done
    return 1
}

overall_rc=0
for src in cifar10 cifar100 supercifar tinyimagenet; do
    if ! list_file="$(resolve_list "$src")"; then
        echo "=== [$src] experiment list not found (pilot_${src}_vit.txt); SKIPPING ==="
        overall_rc=1
        continue
    fi
    n_exp=$(grep -cv '^\s*\(#\|$\)' "$list_file" || true)
    echo ""
    echo "================================================================"
    echo "=== [$src] $n_exp experiments from $list_file  ($(date '+%F %T'))"
    echo "=== modes: ${MODES[$src]}"
    echo "================================================================"
    if ! EXPERIMENTS_FILE="$list_file" TEST_MODES="${MODES[$src]}" \
         bash run_new_csfs_pilot.sh; then
        echo "=== [$src] runner exited nonzero; continuing with next source ==="
        overall_rc=1
    fi
done

echo ""
echo "=== all sources done ($(date '+%F %T')), exit $overall_rc ==="
exit $overall_rc

# X6-GradPCA Pilot 1: deep-gradient stage

Campaign code for the activation-gradient added-value pilot. Protocol of record: `documentation/X6_gradpca_pilot1_protocol.md` (repo root, Obsidian side). Theory: `documentation/X6_gradpca_theorems.md` (E-series).

## Contents

- `per_sample_grads.py` — exact per-sample gradients for one Conv2d/Linear layer from a single batched forward + backward (hook-based Goodfellow trick; no functorch, torch 1.13-safe). Unit tests: `tests/test_per_sample_grads.py` (local, 7/7).
- `deep_gradpca.py` — the per-checkpoint stage: fits and scores `GradPCA_lastlayer_{sum,max}` (deep subset) plus the matched head baselines `GradPCA_head_{sum,max}` / `ActPCA_cmeans` from the same forwards, replicates the pipeline's metric conventions, and writes `outputs/<slug>.json` + `<slug>_scores.npz`. Includes a runtime hook-vs-autograd self-check that aborts on disagreement.
- `run_pilot1_deep.sh` — roster loop (see header for env vars).

## Pilot 1 runbook (HPC, code/ repo root, paper container)

1. **Head arms via the standard pipeline** (also feeds the X6 absolute-performance package):

   ```bash
   ARMS=gradpca CSF_BATCH_SIZE=256 CSF_NUM_WORKERS=12 \
     EXPERIMENTS_FILE=pilot1_gradpca_experiments.txt \
     TEST_MODES="iid_test ood_sncs_c10 ood_nsncs_svhn ood_nsncs_ti ood_nsncs_lsun_cropped ood_nsncs_lsun_resize ood_nsncs_isun ood_nsncs_textures ood_nsncs_places365" \
     nohup bash run_new_csfs_pilot.sh > gradpca_pilot1_head.log 2>&1 &
   ```

   Then `python tests/check_gradpca_e1_confids.py --model_path <exp> --modes iid_test,iid_val,...` per checkpoint.

2. **Deep-gradient smoke test** (one checkpoint, capped samples; ~minutes):

   ```bash
   SMOKE=2048 EXPERIMENTS="cifar100_paper_sweep/confidnet_bbvgg13_do0_run1_rew2.2" \
     bash x6_gradpca/run_pilot1_deep.sh
   ```

   Confirm in the log: self-check passed, k* values printed, JSON written. Then inspect `--list_params` output once per architecture and record it in the protocol doc.

3. **Full deep stage** (all six checkpoints):

   ```bash
   EXPERIMENTS_FILE=pilot1_gradpca_experiments.txt \
     nohup bash x6_gradpca/run_pilot1_deep.sh > gradpca_deep_pilot1.log 2>&1 &
   ```

4. Aggregate: the per-checkpoint JSONs already contain the matched deep-minus-head deltas per mode (`deltas` block); the cross-seed gate evaluation happens against the protocol doc's criteria.

`outputs/` stays on the HPC (gitignored), matching the x6_spectral convention.

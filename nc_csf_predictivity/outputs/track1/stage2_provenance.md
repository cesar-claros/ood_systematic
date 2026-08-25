# Stage-2 provenance and reproducibility closure (audit #6, E5)

Date: 2026-08-25. Repository: `cesar-claros/ood_systematic` (this checkout, `code/`).

## Outcome-blind specification evidence

The design note (`documentation/heldout_theory_validation_design.md`) received its execution header after results existed, so the strongest evidence of outcome-blind specification is the code history, as audit #6 section 9.5 prescribes:

- **Pre-outcome validation-code commit:** `3971000` ("implement schema 2 support in extract_pool_coords.py and heldout_theory_validation.py"). This commit contains the complete evaluation pipeline (theory arm with no fitted parameters, all five baselines, folds, materiality rule |gap| >= 10 AUGRC-milli units, gates) and the schema-2 extractor, committed before the pool sweep produced any coordinate file and therefore before any held-out outcome could be inspected. The runner's self-test (theory-as-generator, sign accuracy 0.982) is part of that commit.
- **Frozen input manifest:** `pilot0/pool_manifest.json` (280 checkpoints; sha256 `4f39558050eb05ac493cedf4746827cf5e7583542da4281a15013f8047ad3deb`), generated from the harmonized outcome table's cell enumeration before extraction.
- The claim contract and its dated amendments (`documentation/companion_phase_diagram_claim_contract.md`) fix the gates; the Stage-3 mode decision followed them mechanically.

## Data artifacts

- **Coordinate files:** `pilot0/pool_coords/` (280 schema-2 JSONs; gitignored as data; content hashes frozen in `pilot0/pool_coords_manifest.sha256`, 280 lines, committed). Extraction: `pilot0/extract_pool_coords.py` on the HPC (`EXPERIMENT_ROOT_DIR=/work/cniel/sw/FD_Shifts/project/experiments` checkpoint store), campaign container (digest recorded in `pilot1/MANIFEST.md`), nohup per-source dispatch; per-set loader provenance and `suite_complete` recorded inside every JSON. GPU forwards are not bit-stable across hardware/reruns; the frozen H-estimator definitions (pilot0, 2026-08-15) are deterministic given features.
- **Outcome table:** `nc_csf_predictivity/outputs/track1/dataset/long_harmonized.parquet` (280 checkpoints x 8 shifts x 20 scores; unit facts regression-checked by `data_unit_check.py` -> `data_unit_report.md`).
- **Held-out reports:** `heldout_theory_report.md` (sha256 `bd838657ae891948f3518dceee48dcdf79523ce55adc9e7eb307a6f3b2c2dcb8`), `heldout_theory_report.json` (sha256 `012a4bf5c8a95e6e5aad34839e3915d286492577b15131f252273bbba2449eff`), plus the E1-E4 closure outputs `stage2_closure_report.md`/`.json` and the per-cell theory cache `theory_cell_predictions.parquet` (committed alongside this file; hashes in the commit).

## Commands and seeds

```
# HPC (campaign container, .env with EXPERIMENT_ROOT_DIR/DATASET_ROOT_DIR)
python pilot0/extract_pool_coords.py --sweep --list          # MISSING 0 verified before dispatch
python pilot0/extract_pool_coords.py --sweep --source {cifar10|cifar100|supercifar100|tinyimagenet}

# local
python heldout_theory_validation.py       # folds seed 2027; bootstrap seeds 11 (ckpt5), 12 (loso); B=2000
python stage2_closure.py                  # E1-E4; frozen rules in docstring; seeds 21 (E2 bands), 31/32 (E3 CIs)
```

Fold construction: grouped 5-fold by checkpoint, `numpy.random.default_rng(2027)` permutation; leave-one-source-out and leave-one-OOD-set-out folds are deterministic partitions. All bootstraps resample checkpoints (clusters), never checkpoint-shift rows.

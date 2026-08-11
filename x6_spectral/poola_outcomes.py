"""Pool A outcome generation (X6 pristine tier; run ONLY after the lock).

Computes this pool's projection-variant AUGRC tables for the first time,
from the same cached features the predictions used. Each cell's probe and
projectors are reconstructed via poola_measure.setup_cell, whose
determinism against the measurement release is verified (records identical
modulo wall-clock). Deployed evaluation convention for the pool, frozen
here before any outcome is inspected:

- Mahalanobis statistics fit on probe-train (all samples, matching
  mahalanobis.py, which takes no correct-only flag), per arm: raw features,
  global back-projections, and class-pred back-projections routed by the
  probe's raw-logit argmax; pseudo-inverse at rcond 1e-6.
- AUGRC per (score, OOD set): mixture = the full ID test set plus the full
  OOD set; failures = ID samples the probe misclassifies plus every OOD
  sample (the paper's new-class convention, identical to the r8 trial's
  failure labels); values reported x1000 as in the paper tables.
- One row per (cell, trial key, OOD set) with delta_augrc =
  AUGRC(base score) - AUGRC(variant score); positive = variant helps.
  Class-avg variants are out of trial scope and not generated.

Pass-5.1 protocol (pool "l14", the confirmatory rerun): the lock is a
TRACKED, TAGGED manifest of prediction, feature, and script hashes
(poola_lock.py), verified here before any outcome is computed; a boolean
flag is no longer accepted for this pool. Evaluation is SAMPLE-DISJOINT
from adaptation: the first BATCH_PER_DRAW x N_DRAWS = 640 rows of every
OOD feature file (consumed by the batch trials) are excluded from the
outcome mixture (re-review 5.2). The historical "main" pool keeps its
original semantics (--locked attestation, full OOD files) and is labeled
exploratory per the pass-5 re-review.

Usage (HPC, from code/):
    python x6_spectral/poola_outcomes.py --pool l14 \
        --features-dir $EXPERIMENT_ROOT_DIR/pool_a/features
    python x6_spectral/poola_outcomes.py --pool main --locked \
        --features-dir $EXPERIMENT_ROOT_DIR/pool_a/features   # historical
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "x8_pool_a"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import projection_filtering_analysis as pfa
from poola_lock import DEFAULT_TAG, pred_dir_for, verify_lock
from poola_measure import (BATCH_PER_DRAW, ID_SOURCES, N_DRAWS, N_PER_CLASS,
                           NPZ_NAME, POOLS, SEEDS, load_npz, refit_maha,
                           setup_cell)
from spectra_campaign_harness import (DEPLOYED_TRIAL_KEYS, batch_augrc,
                                      deployed_scores, make_backprojector,
                                      trial_arm_key)

ADAPT_ROWS = BATCH_PER_DRAW * N_DRAWS

#: Trial key -> (family, variant); mirror of score_tier_b.TRIAL_FAMILY7
#: (duplicated here so outcome generation stays import-light; score_tier_b
#: remains the master copy for the checkpoint pools).
FAMILY_VARIANT = {
    "mls": ("MLS", "global"), "energy": ("Energy", "global"),
    "msr": ("MSR", "global"), "maha": ("Maha", "global"),
    "gradnorm": ("GradNorm", "global"),
    "maha_cp": ("Maha", "class pred"),
    "gradnorm_cp": ("GradNorm", "class pred"),
    "recerr_cp": ("PCA RecError", "class pred"),
    "recerr_class": ("PCA RecError", "class"),
}


def outcome_cell(features_dir: Path, encoder: str, source: str, n_pc: int,
                 seed: int, skip_rows: int = 0) -> list[dict] | None:
    """All outcome rows for one cell, or None when features are missing.

    skip_rows > 0 excludes the leading adaptation rows of every OOD feature
    file from the evaluation mixture (disjoint protocol)."""
    cell = setup_cell(features_dir, encoder, source, n_pc, seed)
    if cell is None:
        return None
    id_test = load_npz(features_dir, encoder, source, "test")
    if id_test is None:
        return None
    h_test, y_test = id_test
    w, b = cell["w_eff"], cell["b_eff"]
    n_classes = cell["n_classes"]
    bp_global, class_bp = cell["bp_global"], cell["class_bp"]
    h_sub, y_sub, g_mean = cell["h_sub"], cell["y_sub"], cell["g_mean"]

    preds_sub = (h_sub @ w.T + b).argmax(1)
    z_cp_sub = np.empty_like(h_sub)
    for c, (mean_c, comps_c, n_c) in enumerate(class_bp):
        mask = preds_sub == c
        if mask.any():
            z_cp_sub[mask] = make_backprojector(mean_c, comps_c,
                                                n_c)(h_sub[mask])
    maha_sets = {"raw": refit_maha(h_sub, y_sub, n_classes, g_mean),
                 "global": refit_maha(bp_global(h_sub), y_sub, n_classes,
                                      g_mean),
                 "cp": refit_maha(z_cp_sub, y_sub, n_classes, g_mean)}

    scores_id = deployed_scores(h_test, w, b, bp_global, class_bp,
                                maha_sets)
    fail_id = ((h_test @ w.T + b).argmax(1) != y_test).astype(float)

    rows = []
    for ood in pfa.OOD_DATASETS[source]:
        loaded = load_npz(features_dir, encoder, NPZ_NAME[ood], "test")
        if loaded is None:
            continue
        ood_feats = loaded[0][skip_rows:]
        if len(ood_feats) == 0:
            continue
        scores_ood = deployed_scores(ood_feats, w, b, bp_global, class_bp,
                                     maha_sets)
        failure = np.concatenate([fail_id, np.ones(len(ood_feats))])
        for key, (base_arm, var_arm) in DEPLOYED_TRIAL_KEYS.items():
            family, variant = FAMILY_VARIANT[key]
            bk = trial_arm_key(key, base_arm)
            vk = trial_arm_key(key, var_arm)
            if bk not in scores_id or vk not in scores_id:
                continue
            vals = {}
            for tag, sk in (("base", bk), ("var", vk)):
                conf = np.concatenate([scores_id[sk], scores_ood[sk]])
                vals[tag] = 1000.0 * batch_augrc(conf, failure)
            rows.append({"encoder": encoder, "source": source,
                         "n_per_class": n_pc, "seed": seed,
                         "family": family, "variant": variant, "ood": ood,
                         "trial_key": key,
                         "augrc_base": round(vals["base"], 4),
                         "augrc_var": round(vals["var"], 4),
                         "delta_augrc": round(vals["base"] - vals["var"],
                                              4)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="X6 Pool A outcome generation (post-lock only)")
    parser.add_argument("--features-dir", type=str, default="pool_a_features")
    parser.add_argument("--pool", choices=list(POOLS), default="l14",
                        help="l14 = pass-5.1 confirmatory rerun (manifest-"
                             "verified lock, disjoint evaluation); main = "
                             "historical pool (legacy --locked, full files)")
    parser.add_argument("--out", type=str, default=None,
                        help="override the per-pool outcomes.csv path")
    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--expect-tag", type=str, default=None)
    parser.add_argument("--locked", action="store_true",
                        help="legacy attestation, accepted for --pool main "
                             "only (superseded by the manifest protocol)")
    parser.add_argument("--no-git-checks", action="store_true",
                        help="hash-only lock verification; synthetic "
                             "self-test only")
    args = parser.parse_args()
    features_dir = Path(args.features_dir)
    if args.pool == "main":
        if not args.locked:
            sys.exit("main pool: refusing without --locked (historical "
                     "protocol; the l14 pool uses manifest verification)")
        skip_rows = 0
    else:
        tag = args.expect_tag or DEFAULT_TAG[args.pool]
        problems = verify_lock(args.pool, features_dir, tag,
                               git_checks=not args.no_git_checks)
        if problems:
            sys.exit("LOCK VERIFICATION FAILED (no outcomes generated):\n"
                     + "\n".join(f"  - {p}" for p in problems))
        print(f"lock verified (manifest + tag {tag} + feature hashes); "
              f"evaluation excludes the first {ADAPT_ROWS} adaptation rows "
              "of every OOD file")
        skip_rows = ADAPT_ROWS
    pool_encoders, _ = POOLS[args.pool]
    if args.encoder and args.encoder not in pool_encoders:
        sys.exit(f"--encoder {args.encoder} not in pool '{args.pool}'")
    out_path = Path(args.out) if args.out \
        else pred_dir_for(args.pool) / "outcomes.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoders = [args.encoder] if args.encoder else pool_encoders
    all_rows: list[dict] = []
    for encoder in encoders:
        for source in ID_SOURCES:
            for n_pc in N_PER_CLASS:
                for seed in SEEDS:
                    t0 = time.time()
                    rows = outcome_cell(features_dir, encoder, source,
                                        n_pc, seed, skip_rows=skip_rows)
                    if rows is None:
                        print(f"[missing ] {encoder} {source} n{n_pc} "
                              f"s{seed}")
                        continue
                    all_rows.extend(rows)
                    print(f"[outcomes] {encoder} {source} n{n_pc} s{seed} "
                          f"({len(rows)} rows, {time.time()-t0:.0f}s)")
    if not all_rows:
        sys.exit("No outcome rows generated")
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"{len(all_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()

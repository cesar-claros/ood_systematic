#!/usr/bin/env python
"""Verify Theorem E1 on real pipeline outputs (Pilot 0 end-to-end check).

Reads the analysis CSVs a csf_fit/csf_eval run produced for one experiment
and checks, per eval mode:

  1. The confids columns 'GradPCA_head_sum', 'GradPCA_head_max', and
     'ActPCA_cmeans' exist. Auto-discovered modes without them are SKIPPED
     (they are confids files from earlier campaigns that did not include
     these families); a mode explicitly requested via --modes without them
     is a FAIL.
  2. Theorem E1: GradPCA_head_sum == ActPCA_cmeans within --tol (the two
     detectors consume the same cached activations, so any gap beyond float
     accumulation noise is a bug).
  3. All three score columns lie in [0, 1] (retained-energy ratios).
  4. Stats-row E1: the stats CSV rows for GradPCA_head_sum and
     ActPCA_cmeans are identical across all metric columns. This check is
     computed by stats() from in-memory arrays BEFORE any CSV merging, so
     it is immune to the merge caveat below.
  5. Informational: mean score for non-failure vs failure rows.

Merge caveat: stats() appends new confids columns onto any existing
confids CSV with `pd.concat(..., axis=1)` (_merge_csv_cols). If the stale
file has MORE rows than the fresh run (e.g. the correct-ID count of a
joint OOD eval drifted between run vintages), the fresh columns get
NaN-padded on the tail. This script detects that signature (all three new
columns NaN on a contiguous tail), reports it loudly, and evaluates E1 on
the fresh rows. Scattered or partial NaNs are NOT excused and fail.

Exit code 0 iff every checked (non-skipped) mode passes.

Usage (on the HPC, from the code/ repo root):
  python tests/check_gradpca_e1_confids.py \
      --model_path cifar100_paper_sweep/confidnet_bbvgg13_do0_run1_rew2.2
  # restrict to the modes you actually ran:
  python tests/check_gradpca_e1_confids.py --model_path ... \
      --modes iid_test,iid_val,ood_nsncs_svhn,ood_nsncs_textures

--model_path is resolved against $EXPERIMENT_ROOT_DIR (env, or code/.env).
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

E1_A, E1_B, MAXV = "GradPCA_head_sum", "ActPCA_cmeans", "GradPCA_head_max"


def resolve_exp_dir(args):
    if args.exp_dir:
        return args.exp_dir
    root = os.environ.get("EXPERIMENT_ROOT_DIR")
    if not root:
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        if os.path.exists(env_file):
            for line in open(env_file):
                if line.strip().startswith("EXPERIMENT_ROOT_DIR="):
                    root = line.strip().split("=", 1)[1]
    if not root:
        sys.exit("ERROR: set --exp_dir, or --model_path with EXPERIMENT_ROOT_DIR (env or code/.env)")
    return os.path.join(root, args.model_path)


def check_mode(analysis, mode, model_opts, tol, explicit):
    """Returns 'PASS', 'SKIP', or 'FAIL' for one eval mode."""
    cfile = os.path.join(analysis, f"confids{model_opts}_{mode}.csv")
    if not os.path.exists(cfile):
        print(f"[{mode}] FAIL: {cfile} missing")
        return "FAIL"
    df = pd.read_csv(cfile, index_col=0)
    missing = [c for c in (E1_A, E1_B, MAXV) if c not in df.columns]
    if missing:
        if explicit:
            print(f"[{mode}] FAIL: requested mode lacks confids columns {missing}")
            return "FAIL"
        print(f"[{mode}] SKIP: columns {missing} absent "
              f"(file predates the gradpca run; mode was not in this run's TEST_MODES)")
        return "SKIP"

    sub = df[[E1_A, E1_B, MAXV]]
    all_nan = sub.isna().all(axis=1)
    any_nan = sub.isna().any(axis=1)
    if (any_nan & ~all_nan).any():
        print(f"[{mode}] FAIL: {(any_nan & ~all_nan).sum()} rows with PARTIAL NaNs "
              f"across the three columns (not a merge-padding signature)")
        return "FAIL"
    n_total, n_pad = len(df), int(all_nan.sum())
    if n_pad:
        pad_pos = np.flatnonzero(all_nan.to_numpy())
        contiguous_tail = pad_pos[0] == n_total - n_pad and pad_pos[-1] == n_total - 1
        if not contiguous_tail:
            print(f"[{mode}] FAIL: {n_pad} all-NaN rows NOT forming a contiguous tail; "
                  f"this is not _merge_csv_cols padding, investigate")
            return "FAIL"
        print(f"[{mode}] NOTE: {n_pad} tail rows NaN-padded by _merge_csv_cols "
              f"(stale confids file has {n_total} rows, fresh run produced {n_total - n_pad}); "
              f"E1 checked on the {n_total - n_pad} fresh rows. Cross-family row-aligned "
              f"analyses on this merged file are NOT valid.")
        df = df.loc[~all_nan]

    a, b, m = df[E1_A].to_numpy(), df[E1_B].to_numpy(), df[MAXV].to_numpy()
    gap = np.abs(a - b).max()
    e1_ok = gap <= tol
    rng_ok = all(((x >= -1e-9) & (x <= 1 + 1e-9)).all() for x in (a, b, m))

    # Stats-row E1 (computed pre-merge by stats(), so authoritative for the fresh run).
    stats_ok, stats_msg = True, "stats file missing"
    sfile = os.path.join(analysis, f"stats{model_opts}_{mode}.csv")
    sdf = None
    if os.path.exists(sfile):
        sdf = pd.read_csv(sfile, index_col=0)
        if E1_A in sdf.index and E1_B in sdf.index:
            ra, rb = sdf.loc[E1_A].to_numpy(float), sdf.loc[E1_B].to_numpy(float)
            stats_ok = bool(np.all(np.isclose(ra, rb, rtol=1e-9, atol=1e-12, equal_nan=True)))
            stats_msg = "identical" if stats_ok else f"DIFFER: {dict(zip(sdf.columns, ra - rb))}"
        else:
            stats_ok, stats_msg = False, "rows missing from stats CSV"

    ok = e1_ok and rng_ok and stats_ok
    print(f"[{mode}] {'PASS' if ok else 'FAIL'}: E1 max|head_sum - cmeans| = {gap:.3e} "
          f"(tol {tol:.0e}), n = {len(df)}, range_ok = {rng_ok}, stats rows {stats_msg}")
    if "residuals" in df.columns:
        res = df["residuals"].to_numpy()
        for name, x in ((E1_A, a), (MAXV, m)):
            mu_ok = x[res == 0].mean() if (res == 0).any() else float("nan")
            mu_bad = x[res == 1].mean() if (res == 1).any() else float("nan")
            print(f"    {name}: mean score non-failure {mu_ok:.4f} vs failure {mu_bad:.4f}")
    if sdf is not None:
        keys = [k for k in (E1_A, E1_B, MAXV) if k in sdf.index]
        cols = [c for c in ("AUGRC", "AUROC_f", "FPR@95TPR") if c in sdf.columns]
        if keys:
            print(sdf.loc[keys, cols].to_string())
    return "PASS" if ok else "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=None, help="experiment name relative to EXPERIMENT_ROOT_DIR")
    ap.add_argument("--exp_dir", type=str, default=None, help="absolute path to the experiment dir")
    ap.add_argument("--model_opts", type=str, default="_RW0_RF0_ASHNone")
    ap.add_argument("--modes", type=str, default=None, help="comma-separated eval modes (default: all found; found-but-not-run modes are skipped)")
    ap.add_argument("--tol", type=float, default=1e-8, help="max |head_sum - cmeans| allowed (scores live in [0,1])")
    args = ap.parse_args()
    if not args.exp_dir and not args.model_path:
        ap.error("provide --model_path or --exp_dir")

    exp_dir = resolve_exp_dir(args)
    analysis = os.path.join(exp_dir, "analysis")
    if not os.path.isdir(analysis):
        sys.exit(f"ERROR: no analysis/ dir at {analysis}")

    explicit = args.modes is not None
    if explicit:
        modes = [m.strip() for m in args.modes.split(",")]
    else:
        prefix = f"confids{args.model_opts}_"
        modes = sorted(os.path.basename(f)[len(prefix):-len(".csv")]
                       for f in glob.glob(os.path.join(analysis, f"{prefix}*.csv")))
    if not modes:
        sys.exit(f"ERROR: no confids{args.model_opts}_*.csv files in {analysis}")

    verdicts = {m: check_mode(analysis, m, args.model_opts, args.tol, explicit) for m in modes}
    n_pass = sum(v == "PASS" for v in verdicts.values())
    n_skip = sum(v == "SKIP" for v in verdicts.values())
    n_fail = sum(v == "FAIL" for v in verdicts.values())
    print(f"\nE1 end-to-end check: {'PASS' if n_fail == 0 else 'FAIL'} "
          f"({n_pass} passed, {n_skip} skipped, {n_fail} failed)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Verify Theorem E1 on real pipeline outputs (Pilot 0 end-to-end check).

Reads the analysis CSVs a csf_fit/csf_eval run produced for one experiment
and checks, per eval mode:

  1. The confids columns 'GradPCA_head_sum', 'GradPCA_head_max', and
     'ActPCA_cmeans' exist.
  2. Theorem E1: GradPCA_head_sum == ActPCA_cmeans within --tol (the two
     detectors consume the same cached activations, so any gap beyond float
     accumulation noise is a bug).
  3. All three score columns lie in [0, 1] (retained-energy ratios).
  4. Informational: mean score for non-failure vs failure rows (ID-correct
     vs OOD/misclassified; the ratio should be higher on non-failures), and
     the stats CSV rows (AUGRC / AUROC_f / FPR@95TPR) for the three keys.
     E1 predicts identical stats rows for head_sum and cmeans.

Exit code 0 iff checks 1-3 pass for every mode examined.

Usage (on the HPC, from the code/ repo root):
  python tests/check_gradpca_e1_confids.py \
      --model_path cifar100_paper_sweep/confidnet_bbvgg13_do0_run1_rew2.2
  # or point directly at the experiment dir:
  python tests/check_gradpca_e1_confids.py --exp_dir /path/to/experiment

--model_path is resolved against $EXPERIMENT_ROOT_DIR (env, or code/.env).
By default every confids<opts>_*.csv in analysis/ is checked; restrict with
--modes iid_test,ood_nsncs_svhn.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=None, help="experiment name relative to EXPERIMENT_ROOT_DIR")
    ap.add_argument("--exp_dir", type=str, default=None, help="absolute path to the experiment dir")
    ap.add_argument("--model_opts", type=str, default="_RW0_RF0_ASHNone")
    ap.add_argument("--modes", type=str, default=None, help="comma-separated eval modes (default: all found)")
    ap.add_argument("--tol", type=float, default=1e-8, help="max |head_sum - cmeans| allowed (scores live in [0,1])")
    args = ap.parse_args()
    if not args.exp_dir and not args.model_path:
        ap.error("provide --model_path or --exp_dir")

    exp_dir = resolve_exp_dir(args)
    analysis = os.path.join(exp_dir, "analysis")
    if not os.path.isdir(analysis):
        sys.exit(f"ERROR: no analysis/ dir at {analysis}")

    if args.modes:
        confids_files = [os.path.join(analysis, f"confids{args.model_opts}_{m.strip()}.csv")
                         for m in args.modes.split(",")]
    else:
        confids_files = sorted(glob.glob(os.path.join(analysis, f"confids{args.model_opts}_*.csv")))
    if not confids_files:
        sys.exit(f"ERROR: no confids{args.model_opts}_*.csv files in {analysis}")

    all_ok = True
    for cfile in confids_files:
        mode = os.path.basename(cfile)[len(f"confids{args.model_opts}_"):-len(".csv")]
        if not os.path.exists(cfile):
            print(f"[{mode}] FAIL: {cfile} missing")
            all_ok = False
            continue
        df = pd.read_csv(cfile, index_col=0)
        missing = [c for c in (E1_A, E1_B, MAXV) if c not in df.columns]
        if missing:
            print(f"[{mode}] FAIL: missing confids columns {missing}")
            all_ok = False
            continue
        a, b, m = df[E1_A].to_numpy(), df[E1_B].to_numpy(), df[MAXV].to_numpy()
        gap = np.abs(a - b).max()
        e1_ok = gap <= args.tol
        rng_ok = all(((x >= -1e-9) & (x <= 1 + 1e-9)).all() for x in (a, b, m))
        status = "PASS" if (e1_ok and rng_ok) else "FAIL"
        print(f"[{mode}] {status}: E1 max|head_sum - cmeans| = {gap:.3e} "
              f"(tol {args.tol:.0e}), n = {len(df)}, range_ok = {rng_ok}")
        if not (e1_ok and rng_ok):
            all_ok = False
        if "residuals" in df.columns:
            res = df["residuals"].to_numpy()
            for name, x in ((E1_A, a), (MAXV, m)):
                mu_ok, mu_bad = x[res == 0].mean(), x[res == 1].mean() if (res == 1).any() else float("nan")
                print(f"    {name}: mean score non-failure {mu_ok:.4f} vs failure {mu_bad:.4f}")
        sfile = os.path.join(analysis, f"stats{args.model_opts}_{mode}.csv")
        if os.path.exists(sfile):
            sdf = pd.read_csv(sfile, index_col=0)
            keys = [k for k in (E1_A, E1_B, MAXV) if k in sdf.index]
            cols = [c for c in ("AUGRC", "AUROC_f", "FPR@95TPR") if c in sdf.columns]
            if keys:
                print(sdf.loc[keys, cols].to_string())

    print("\nE1 end-to-end check:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

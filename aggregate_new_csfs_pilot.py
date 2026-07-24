"""Aggregate the E-F pilot: ASH / ReAct / Mahalanobis++ / NCI vs baselines.

Reads the per-checkpoint stats CSVs written by the pilot runner
(`stats_RW0_RF0_ASH<method>_<mode>.csv` under each experiment's analysis/
dir) and builds the rebuttal comparison table: mean AUGRC per (arm, CSF,
eval mode) across checkpoints, plus each new method's rank against the
baseline CSFs evaluated on the same checkpoints.

Run on the HPC, or locally after rsyncing the analysis/ dirs:
  python aggregate_new_csfs_pilot.py --experiments-file pilot_experiments.txt
Outputs: new_csfs_pilot_summary.{csv,md} in the working directory.
"""

from __future__ import annotations

import argparse
import os
import pathlib

import pandas as pd

KNOWN_MODES = [
    "iid_test", "ood_nsncs_svhn", "ood_nsncs_ti", "ood_nsncs_lsun_cropped",
    "ood_nsncs_lsun_resize", "ood_nsncs_isun", "ood_nsncs_textures",
    "ood_nsncs_places365", "ood_sncs_c100", "ood_nsncs_c10",
    "ood_nsncs_c100",
]
NEW_METHODS = {"MahaPP", "NCI"}
REFERENCE_METHODS = {"Maha", "NeCo", "CTM", "ViM", "Residual", "Energy",
                     "MSR", "MLS", "fDBD", "NNGuide"}


def parse_stats_name(name: str) -> tuple[str, str] | None:
    """stats_RW0_RF0_ASH<method>_<mode>.csv -> (ash_method, mode)."""
    if not (name.startswith("stats_RW0_RF0_ASH") and name.endswith(".csv")):
        return None
    rest = name[len("stats_RW0_RF0_ASH"):-len(".csv")]
    for mode in sorted(KNOWN_MODES, key=len, reverse=True):
        if rest.endswith("_" + mode):
            return rest[:-len(mode) - 1], mode
    return None


def arm_label(ash_method: str) -> str:
    if ash_method == "None":
        return "base"
    if ash_method.startswith("ash"):
        return f"ASH ({ash_method})"
    if ash_method.startswith("react_and_ash"):
        return "ReAct+ASH"
    if ash_method.startswith("react"):
        return "ReAct"
    return ash_method


def load_experiment(root: pathlib.Path, exp: str) -> list[dict]:
    rows = []
    analysis = root / exp / "analysis"
    if not analysis.is_dir():
        print(f"WARNING: no analysis dir for {exp}")
        return rows
    for f in sorted(analysis.iterdir()):
        parsed = parse_stats_name(f.name)
        if parsed is None:
            continue
        ash_method, mode = parsed
        df = pd.read_csv(f, index_col=0)
        for csf, r in df.iterrows():
            rows.append({
                "experiment": exp, "arm": arm_label(ash_method),
                "ash_method": ash_method, "csf": csf, "mode": mode,
                "AUGRC": r.get("AUGRC"), "AURC": r.get("AURC"),
                "AUROC_f": r.get("AUROC_f"),
                "FPR@95TPR": r.get("FPR@95TPR"),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--experiment-root",
                    default=os.environ.get("EXPERIMENT_ROOT_DIR", "."))
    ap.add_argument("--experiments", nargs="*", default=None)
    ap.add_argument("--experiments-file", default=None)
    ap.add_argument("--out-prefix", default="new_csfs_pilot_summary")
    args = ap.parse_args()

    if args.experiments_file:
        lines = pathlib.Path(args.experiments_file).read_text().splitlines()
        exps = [ln.strip() for ln in lines
                if ln.strip() and not ln.strip().startswith("#")]
    elif args.experiments:
        exps = args.experiments
    else:
        raise SystemExit("Provide --experiments or --experiments-file")

    root = pathlib.Path(args.experiment_root)
    rows = [r for exp in exps for r in load_experiment(root, exp)]
    if not rows:
        raise SystemExit("No stats files found")
    long_df = pd.DataFrame(rows)
    long_df.to_csv(f"{args.out_prefix}_long.csv", index=False)

    focus = long_df[
        ((long_df["arm"] == "base")
         & long_df["csf"].isin(NEW_METHODS | REFERENCE_METHODS))
        | (long_df["arm"] != "base")
    ]
    summary = (focus.groupby(["arm", "csf", "mode"])
               .agg(n=("AUGRC", "size"), AUGRC=("AUGRC", "mean"),
                    AUROC_f=("AUROC_f", "mean"),
                    FPR95=("FPR@95TPR", "mean"))
               .round(3).reset_index())

    base = long_df[long_df["arm"] == "base"]
    rank_rows = []
    for (csf, mode), _ in summary[summary["csf"].isin(NEW_METHODS)
                                  ].groupby(["csf", "mode"]):
        pool = (base[base["mode"] == mode]
                .groupby("csf")["AUGRC"].mean().sort_values())
        if csf in pool.index:
            rank_rows.append({
                "csf": csf, "mode": mode,
                "rank": int(pool.index.get_loc(csf)) + 1,
                "of": len(pool),
                "best_in_mode": pool.index[0],
                "best_AUGRC": round(float(pool.iloc[0]), 3),
            })
    ranks = pd.DataFrame(rank_rows)

    md = ["# E-F pilot summary: ASH / ReAct / Mahalanobis++ / NCI\n",
          f"\nExperiments: {len(exps)}; rows: {len(long_df):,}.\n",
          "\n## Mean metrics per (arm, CSF, mode)\n\n",
          summary.to_markdown(index=False),
          "\n\n## New-method AUGRC rank among all base-arm CSFs\n\n",
          ranks.to_markdown(index=False) if not ranks.empty else "(none)",
          "\n"]
    pathlib.Path(f"{args.out_prefix}.md").write_text("".join(md))
    summary.to_csv(f"{args.out_prefix}.csv", index=False)
    print(f"wrote {args.out_prefix}.md / .csv / _long.csv")


if __name__ == "__main__":
    main()

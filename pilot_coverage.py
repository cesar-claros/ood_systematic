"""Report which (experiment, arm) combinations of the E-F pilot are complete.

Ground truth is what exists on disk, not the runner logs (which can
interleave when multiple sweeps share a filename). For every experiment in
the given list(s) it checks, per arm:

  newcsf : MahaPP and NCI rows present in stats_RW0_RF0_ASHNone_iid_test.csv
  ash    : stats_RW0_RF0_ASHash*_iid_test.csv exists
  react  : stats_RW0_RF0_ASHreact@*_iid_test.csv exists

and writes one rerun list per arm containing the incomplete experiments
(rerun_newcsf.txt, rerun_ash.txt, rerun_react.txt).

  python pilot_coverage.py --experiments-file pilot_all.txt
"""

from __future__ import annotations

import argparse
import os
import pathlib

import pandas as pd

ARMS = ["newcsf", "ash", "react"]


def newcsf_done(analysis: pathlib.Path) -> bool:
    f = analysis / "stats_RW0_RF0_ASHNone_iid_test.csv"
    if not f.exists():
        return False
    idx = pd.read_csv(f, index_col=0).index
    return "MahaPP" in idx and "NCI" in idx


def glob_done(analysis: pathlib.Path, pattern: str) -> bool:
    return any(analysis.glob(pattern))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--experiments-file", nargs="+", required=True)
    ap.add_argument("--experiment-root",
                    default=os.environ.get("EXPERIMENT_ROOT_DIR", "."))
    args = ap.parse_args()

    exps = []
    for ef in args.experiments_file:
        for ln in pathlib.Path(ef).read_text().splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#") and ln not in exps:
                exps.append(ln)

    root = pathlib.Path(args.experiment_root)
    missing: dict[str, list[str]] = {arm: [] for arm in ARMS}
    for exp in exps:
        analysis = root / exp / "analysis"
        if not newcsf_done(analysis):
            missing["newcsf"].append(exp)
        if not glob_done(analysis, "stats_RW0_RF0_ASHash*_iid_test.csv"):
            missing["ash"].append(exp)
        if not glob_done(analysis, "stats_RW0_RF0_ASHreact@*_iid_test.csv"):
            missing["react"].append(exp)

    print(f"experiments checked: {len(exps)}")
    for arm in ARMS:
        n = len(missing[arm])
        print(f"  {arm:<7s} complete: {len(exps) - n:>4d}   missing: {n}")
        out = pathlib.Path(f"rerun_{arm}.txt")
        out.write_text("\n".join(missing[arm]) + "\n" if missing[arm] else "")
        if missing[arm]:
            print(f"          -> {out}")


if __name__ == "__main__":
    main()

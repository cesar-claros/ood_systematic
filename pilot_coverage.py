"""Report which (experiment, arm) combinations of the E-F pilot are complete.

Ground truth is what exists on disk, not the runner logs (which can
interleave when multiple sweeps share a filename). Completion is checked for
EVERY eval mode of the experiment's source (mode lists derived from the
configs_exp/ sweep manifests; corruptions excluded), per arm:

  newcsf : stats_RW0_RF0_ASHNone_<mode>.csv contains MahaPP and NCI rows
  ash    : stats_RW0_RF0_ASHash*_<mode>.csv exists
  react  : stats_RW0_RF0_ASHreact@*_<mode>.csv exists

Incomplete experiments are written to one rerun list per arm
(rerun_newcsf.txt, rerun_ash.txt, rerun_react.txt) for direct use as
EXPERIMENTS_FILE with ARMS=<arm> (split per source before launching, since
TEST_MODES differs per source).

  python pilot_coverage.py --experiments-file pilot_all.txt
"""

from __future__ import annotations

import argparse
import os
import pathlib

import pandas as pd

CODE_DIR = pathlib.Path(__file__).resolve().parent
ARMS = ["newcsf", "ash", "react"]
SOURCES = ["cifar10", "cifar100", "supercifar", "tinyimagenet"]


def source_catalog(configs_dir: pathlib.Path) -> dict[str, dict]:
    """Sweep-path prefix -> {source, modes} from the sweep manifests."""
    catalog: dict[str, dict] = {}
    for source in SOURCES:
        train = configs_dir / f"configs_{source}_iid_train.txt"
        if not train.exists():
            continue
        first = train.read_text().splitlines()[1].split()[1]
        prefix = first.split("/")[0]
        modes = []
        for f in sorted(configs_dir.glob(f"configs_{source}_iid_test_*.txt")) \
                + sorted(configs_dir.glob(f"configs_{source}_ood_test_*.txt")):
            rows = f.read_text().splitlines()[1:]
            if rows:
                mode = rows[0].split()[-1]
                if mode != "iid_test_corruptions" and mode not in modes:
                    modes.append(mode)
        catalog[prefix] = {"source": source, "modes": modes}
    return catalog


def stats_index(path: pathlib.Path) -> set[str]:
    """Method-name index of a stats CSV (first column only, fast)."""
    try:
        return set(pd.read_csv(path, usecols=[0], index_col=0).index)
    except (OSError, ValueError, pd.errors.ParserError):
        return set()


def arm_missing_modes(analysis: pathlib.Path, arm: str,
                      modes: list[str]) -> list[str]:
    """Modes whose stats are absent or incomplete for this arm."""
    out = []
    for mode in modes:
        if arm == "newcsf":
            f = analysis / f"stats_RW0_RF0_ASHNone_{mode}.csv"
            if not f.exists() or not {"MahaPP", "NCI"} <= stats_index(f):
                out.append(mode)
        elif arm == "ash":
            if not any(analysis.glob(f"stats_RW0_RF0_ASHash*_{mode}.csv")):
                out.append(mode)
        elif arm == "react":
            if not any(analysis.glob(f"stats_RW0_RF0_ASHreact@*_{mode}.csv")):
                out.append(mode)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--experiments-file", nargs="+", required=True)
    ap.add_argument("--experiment-root",
                    default=os.environ.get("EXPERIMENT_ROOT_DIR", "."))
    ap.add_argument("--configs-dir", default=str(CODE_DIR / "configs_exp"))
    ap.add_argument("--detail", action="store_true",
                    help="also print, per (source, arm), how many "
                         "experiments are missing each mode")
    args = ap.parse_args()

    catalog = source_catalog(pathlib.Path(args.configs_dir))
    if not catalog:
        raise SystemExit(f"No sweep manifests found in {args.configs_dir}; "
                         "cannot derive per-source mode lists")

    exps = []
    for ef in args.experiments_file:
        for ln in pathlib.Path(ef).read_text().splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#") and ln not in exps:
                exps.append(ln)

    root = pathlib.Path(args.experiment_root)
    missing: dict[str, list[str]] = {arm: [] for arm in ARMS}
    totals: dict[str, int] = {}
    miss_by_source: dict[str, dict[str, int]] = {arm: {} for arm in ARMS}
    mode_counts: dict[tuple[str, str], dict[str, int]] = {}
    for source in SOURCES:
        for arm in ARMS:
            mode_counts[(source, arm)] = {}
    unknown = []
    for exp in exps:
        entry = catalog.get(exp.split("/")[0])
        if entry is None:
            unknown.append(exp)
            continue
        source = entry["source"]
        totals[source] = totals.get(source, 0) + 1
        analysis = root / exp / "analysis"
        for arm in ARMS:
            missing_modes = arm_missing_modes(analysis, arm, entry["modes"])
            if missing_modes:
                missing[arm].append(exp)
                miss_by_source[arm][source] = (
                    miss_by_source[arm].get(source, 0) + 1)
                for mode in missing_modes:
                    mode_counts[(source, arm)][mode] = (
                        mode_counts[(source, arm)].get(mode, 0) + 1)

    print(f"experiments checked: {len(exps)} "
          f"(modes verified per source: "
          f"{ {v['source']: len(v['modes']) for v in catalog.values()} })")
    for exp in unknown:
        print(f"UNKNOWN SWEEP PREFIX (skipped): {exp}")

    per_source = pd.DataFrame([
        {"source": source, "experiments": n_total,
         **{arm: f"{n_total - miss_by_source[arm].get(source, 0)} ok / "
                 f"{miss_by_source[arm].get(source, 0)} miss"
            for arm in ARMS}}
        for source, n_total in sorted(totals.items())
    ])
    print("\nper source:\n")
    print(per_source.to_string(index=False))

    if args.detail:
        print("\nmissing modes per (source, arm):")
        for (source, arm), counts in mode_counts.items():
            if counts:
                summary = ", ".join(f"{m}: {n}" for m, n in
                                    sorted(counts.items(),
                                           key=lambda kv: -kv[1]))
                print(f"  {source}/{arm}: {summary}")

    print("\noverall:")
    for arm in ARMS:
        n = len(missing[arm])
        print(f"  {arm:<7s} complete: {len(exps) - n:>4d}   missing: {n}")
        out = pathlib.Path(f"rerun_{arm}.txt")
        out.write_text("\n".join(missing[arm]) + "\n" if missing[arm] else "")
        if missing[arm]:
            print(f"          -> {out}")


if __name__ == "__main__":
    main()

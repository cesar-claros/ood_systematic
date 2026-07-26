"""Derive the E-F pilot inputs from the paper's sweep manifests (configs_exp/).

Reads `configs_<source>_iid_train.txt` for the checkpoint inventory and the
`configs_<source>_{iid,ood}_test_*.txt` manifests for the exact TestMode
vocabulary, applies substring filters (e.g. bbvgg13, run1), writes the
experiment list for `run_new_csfs_pilot.sh`, and prints the matching
TEST_MODES value plus a ready-to-paste launch command. This replaces
hand-built experiment lists and guessed mode names with the same inputs the
paper's own sweeps used.

Examples:
  python pilot_from_configs.py --source cifar100 --filter bbvgg13 run1
  python pilot_from_configs.py --source cifar100 --filter bbvgg13 \
      --check-dirs   # on the cluster: also verify experiment dirs exist
"""

from __future__ import annotations

import argparse
import os
import pathlib

import pandas as pd

CODE_DIR = pathlib.Path(__file__).resolve().parent

#: Eval mode -> dataset name as used in the clip_distances CSVs (Table 6).
MODE_DATASET = {
    "ood_sncs_c10": "cifar10", "ood_sncs_c100": "cifar100",
    "ood_sncs_sc100": "supercifar100", "ood_nsncs_ti": "tinyimagenet",
    "ood_nsncs_svhn": "svhn", "ood_nsncs_lsun_cropped": "lsun cropped",
    "ood_nsncs_lsun_resize": "lsun resize", "ood_nsncs_isun": "isun",
    "ood_nsncs_textures": "textures", "ood_nsncs_places365": "places365",
}
#: Manifest source key -> clip_distances file key.
CLIP_SOURCE_NAME = {"supercifar": "supercifar100"}


def paper_suite(source: str, clip_dir: pathlib.Path) -> set[str]:
    """OOD dataset names in the paper's Table 6 grouping for one source."""
    clip_name = CLIP_SOURCE_NAME.get(source, source)
    df = pd.read_csv(clip_dir / f"clip_distances_{clip_name}.csv",
                     header=[0, 1], index_col=0)
    group_col = [c for c in df.columns if c[0] == "group"][0]
    groups = df[group_col].astype(int)
    return {ds for ds, g in groups.items() if g > 0}


def filter_modes_to_paper(
        modes: list[str], source: str, clip_dir: pathlib.Path
) -> tuple[list[str], list[str], list[str]]:
    """Reconcile manifest modes with the paper's Table 6 suite.

    iid_test is always kept; an OOD mode is kept iff its dataset appears in
    the source's CLIP-derived near/mid/far grouping.

    Returns:
        (kept, dropped, gaps): kept modes, manifest modes outside the suite,
        and suite datasets covered by NO manifest mode (a gap means the
        pilot cannot reproduce the paper's clique blocks for this source).
    """
    suite = paper_suite(source, clip_dir)
    kept, dropped = [], []
    for mode in modes:
        if mode == "iid_test" or MODE_DATASET.get(mode) in suite:
            kept.append(mode)
        else:
            dropped.append(mode)
    covered = {MODE_DATASET[m] for m in kept if m in MODE_DATASET}
    gaps = sorted(suite - covered)
    return kept, dropped, gaps


def read_manifest(path: pathlib.Path) -> list[list[str]]:
    """Data rows (whitespace-split) of one manifest, header dropped."""
    rows = []
    for line in path.read_text().splitlines()[1:]:
        parts = line.split()
        if parts:
            rows.append(parts)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--configs-dir", default=str(CODE_DIR / "configs_exp"))
    ap.add_argument("--source", required=True,
                    choices=["cifar10", "cifar100", "supercifar",
                             "tinyimagenet"])
    ap.add_argument("--filter", nargs="*", default=["bbvgg13"],
                    help="substrings a ModelPath must ALL contain "
                         "(default: bbvgg13; add run1 for the 1-seed slice)")
    ap.add_argument("--include-corruptions", action="store_true",
                    help="keep iid_test_corruptions (excluded by default; "
                         "not needed for the OOD pilot)")
    ap.add_argument("--clip-dir", default=str(CODE_DIR / "clip_scores"))
    ap.add_argument("--all-manifest-modes", action="store_true",
                    help="keep every manifest mode instead of restricting to "
                         "the paper's Table 6 OOD suite (clip_distances)")
    ap.add_argument("--out", default="pilot_experiments.txt")
    ap.add_argument("--check-dirs", action="store_true",
                    help="verify each experiment dir exists under "
                         "$EXPERIMENT_ROOT_DIR (cluster only)")
    args = ap.parse_args()

    cdir = pathlib.Path(args.configs_dir)
    train_manifest = cdir / f"configs_{args.source}_iid_train.txt"
    exps = sorted({r[1] for r in read_manifest(train_manifest)
                   if all(f in r[1] for f in args.filter)})
    if not exps:
        raise SystemExit(f"No experiments match filters {args.filter} "
                         f"in {train_manifest}")

    modes = []
    for f in sorted(cdir.glob(f"configs_{args.source}_iid_test_*.txt")) \
            + sorted(cdir.glob(f"configs_{args.source}_ood_test_*.txt")):
        rows = read_manifest(f)
        if rows:
            mode = rows[0][-1]
            if mode == "iid_test_corruptions" and not args.include_corruptions:
                continue
            if mode not in modes:
                modes.append(mode)
    if not args.all_manifest_modes:
        modes, dropped, gaps = filter_modes_to_paper(
            modes, args.source, pathlib.Path(args.clip_dir))
        if dropped:
            print(f"dropped (not in the paper's Table 6 suite for "
                  f"{args.source}): {' '.join(dropped)}")
        if gaps:
            raise SystemExit(
                f"GAP: Table 6 datasets for {args.source} with no manifest "
                f"eval mode: {gaps}. The pilot cannot match the paper's "
                "clique blocks; check configs_exp/ and MODE_DATASET.")

    missing = []
    if args.check_dirs:
        root = os.environ.get("EXPERIMENT_ROOT_DIR")
        if not root:
            raise SystemExit("--check-dirs needs EXPERIMENT_ROOT_DIR set")
        missing = [e for e in exps
                   if not pathlib.Path(root, e).is_dir()]
        for e in missing:
            print(f"MISSING DIR: {e}")

    pathlib.Path(args.out).write_text("\n".join(exps) + "\n")
    print(f"wrote {args.out}: {len(exps)} experiments "
          f"({len(missing)} missing dirs)" if args.check_dirs
          else f"wrote {args.out}: {len(exps)} experiments")
    print(f"modes ({len(modes)}): {' '.join(modes)}")
    print("\nlaunch:")
    print(f"  CUDA_VISIBLE_DEVICES=1 VENV=/nonexistent "
          f"EXPERIMENTS_FILE={args.out} \\")
    print(f"    TEST_MODES=\"{' '.join(modes)}\" \\")
    print("    nohup singularity exec --nv systematic_ood.sif "
          "bash run_new_csfs_pilot.sh > new_csfs_pilot.log 2>&1 &")


if __name__ == "__main__":
    main()

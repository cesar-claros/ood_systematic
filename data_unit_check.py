"""R6 (audit #5, section 13): deterministic data-unit and fDBD regression check.

Frozen purpose: settle the observation-unit facts of the harmonized VGG-13 pool
and the sourcewise fDBD comparisons, so the manuscript's wording can be pinned
to reproducible numbers and the factual errors flagged by the 2026-08-24
evaluation (280 "checkpoint-shift cells"; "fDBD beats CTM only on
TinyImageNet") cannot silently return.

Also covers R5 support: reports whether any per-population sample-size or
prevalence column exists in the harmonized table (it decides how the AUGRC
fixed-prevalence corollary may be invoked).

Usage (from code/):  python data_unit_check.py
Outputs: nc_csf_predictivity/outputs/track1/data_unit_report.md (+ .json)
Deterministic: pure aggregation, no randomness.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CODE = Path(__file__).resolve().parent
PARQUET = CODE / "nc_csf_predictivity/outputs/track1/dataset/long_harmonized.parquet"
OUT_DIR = CODE / "nc_csf_predictivity/outputs/track1"

CELL_KEYS = ["paradigm", "source", "run", "reward", "dropout"]


def checkpoint_key(df: pd.DataFrame) -> pd.Series:
    return (df.paradigm.astype(str) + "|" + df.source.astype(str) + "|"
            + df["run"].astype(str) + "|" + df.reward.astype(str) + "|"
            + df.dropout.astype(str))


def main() -> None:
    df = pd.read_parquet(PARQUET)
    report: dict = {"parquet": str(PARQUET), "total_rows": int(len(df))}
    lines = ["# Data-unit and fDBD regression check (R6; deterministic)", "",
             f"Source table: `{PARQUET.relative_to(CODE.parent)}`; total rows {len(df)}.", ""]

    report["columns"] = list(df.columns)
    arch_counts = df.architecture.value_counts().to_dict()
    report["architecture_rows"] = {str(k): int(v) for k, v in arch_counts.items()}
    lines += ["## Architectures (row counts)", ""]
    lines += [f"- {k}: {v}" for k, v in report["architecture_rows"].items()]
    lines.append("")

    vgg = df[df.architecture == "VGG13"].copy()
    vgg["ckpt"] = checkpoint_key(vgg)
    ood = vgg[vgg.eval_dataset != "test"]

    n_ckpt = int(vgg.ckpt.nunique())
    shifts_per_ckpt = ood.groupby("ckpt").eval_dataset.nunique()
    n_obs = int(ood.groupby(["ckpt", "eval_dataset"]).ngroups)
    n_ood_rows = int(len(ood))
    n_csf = int(vgg.csf.nunique())
    n_ood_names = int(ood.eval_dataset.nunique())
    src_counts = vgg.groupby("source").ckpt.nunique().to_dict()

    report["vgg13"] = {
        "unique_checkpoints": n_ckpt,
        "checkpoints_per_source": {str(k): int(v) for k, v in src_counts.items()},
        "ood_shifts_per_checkpoint_min": int(shifts_per_ckpt.min()),
        "ood_shifts_per_checkpoint_max": int(shifts_per_ckpt.max()),
        "distinct_ood_set_names_poolwide": n_ood_names,
        "checkpoint_shift_observations": n_obs,
        "ood_long_rows_all_csfs": n_ood_rows,
        "csf_count": n_csf,
    }
    lines += ["## VGG-13 observation units", "",
              f"- unique trained checkpoints (paradigm|source|run|reward|dropout): **{n_ckpt}**",
              f"- checkpoints per source: {report['vgg13']['checkpoints_per_source']}",
              f"- OOD shifts per checkpoint: min {shifts_per_ckpt.min()}, max {shifts_per_ckpt.max()}",
              f"- distinct OOD set names pool-wide: {n_ood_names}",
              f"- unique checkpoint-shift observations: **{n_obs}**",
              f"- OOD long-format rows over all {n_csf} CSFs: **{n_ood_rows}**", ""]

    # The crossing-audit unit ("Cells: N" in crossing_robustness_report.md) is
    # one row per (checkpoint, eval_dataset) restricted to Energy/CTM rows.
    from crossing_robustness_audit import build_cells
    cells = build_cells(df)
    audit_rows = int(len(cells))
    audit_ckpts = int(cells.cell.nunique())
    report["crossing_audit_unit"] = {"rows": audit_rows, "unique_checkpoints": audit_ckpts}
    lines += ["## Crossing-audit unit resolution", "",
              f"The crossing audit's report header `Cells: {audit_ckpts}` counts unique checkpoints (bootstrap clusters); its estimator operates on **{audit_rows} (checkpoint, OOD set) rows** with both Energy and CTM present. Manuscript wording must state checkpoints and checkpoint-shift observations separately and must not call either number 'checkpoint-shift cells'.", ""]

    # fDBD sourcewise comparisons (AUGRC: lower is better).
    lines += ["## fDBD sourcewise comparisons (mean AUGRC over OOD rows; lower is better)", ""]
    fd_rows = ood[ood.csf.isin(["fDBD", "MLS", "CTM"])]
    piv = (fd_rows.groupby(["source", "csf"]).augrc.mean().unstack("csf"))
    fdbd = {}
    lines += ["| source | fDBD | MLS | CTM | fDBD-MLS | fDBD-CTM | fDBD beats MLS | fDBD beats CTM |",
              "|---|---|---|---|---|---|---|---|"]
    for src, row in piv.iterrows():
        d_mls = row["fDBD"] - row["MLS"]
        d_ctm = row["fDBD"] - row["CTM"]
        fdbd[str(src)] = {"fDBD": float(row["fDBD"]), "MLS": float(row["MLS"]),
                          "CTM": float(row["CTM"]), "fDBD_minus_MLS": float(d_mls),
                          "fDBD_minus_CTM": float(d_ctm),
                          "fdbd_beats_mls": bool(d_mls < 0),
                          "fdbd_beats_ctm": bool(d_ctm < 0)}
        lines.append(f"| {src} | {row['fDBD']:.2f} | {row['MLS']:.2f} | {row['CTM']:.2f} | "
                     f"{d_mls:+.2f} | {d_ctm:+.2f} | {d_mls < 0} | {d_ctm < 0} |")
    report["fdbd_sourcewise"] = fdbd
    lines.append("")
    beats_mls = [s for s, v in fdbd.items() if v["fdbd_beats_mls"]]
    beats_ctm = [s for s, v in fdbd.items() if v["fdbd_beats_ctm"]]
    lines += [f"- fDBD beats MLS on: {beats_mls or 'no source'}",
              f"- fDBD beats CTM on: {beats_ctm or 'no source'}", ""]

    # R5 support: any sample-size / prevalence columns?
    size_like = [c for c in df.columns if any(t in c.lower() for t in ("n_", "count", "size", "prev", "samples"))]
    report["r5_size_like_columns"] = size_like
    lines += ["## R5 support: prevalence columns", "",
              f"Columns suggesting per-population sample sizes or prevalence: {size_like or 'NONE'}. "
              "If none, the fixed-prevalence premise of the AUGRC crossing-invariance corollary cannot be verified from this table and must be handled at the protocol level (OOD sets differ in size, so prevalence varies across shifts; empirical crossings are computed directly in AUGRC and do not invoke the corollary).", ""]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "data_unit_report.md").write_text("\n".join(lines))
    (OUT_DIR / "data_unit_report.json").write_text(json.dumps(report, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

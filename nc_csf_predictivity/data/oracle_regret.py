"""Step 7: Oracle and ground-truth ranking tables.

For each "regret-evaluable row" — one (model row, eval_dataset) — compute:
  - oracle_csf  = argmin AUGRC across the row's CSF inventory
  - worst_csf   = argmax AUGRC
  - csf_ranking = list of CSFs sorted by AUGRC ascending (oracle first)
  - augrc_ranking = parallel list of AUGRC values
  - per-side oracle/worst restricted to head-side and feature-side pools

Track 1 row key: (architecture, paradigm, source, run, dropout, reward, eval_dataset)
Track 2 row key: (architecture, paradigm, source, eval_dataset)

Outputs:
  outputs/track1/dataset/oracle.parquet
  outputs/track2/dataset/oracle.parquet
  outputs/05_oracle_check.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = DATA_DIR.parent
DEFAULT_OUT_ROOT = PIPELINE_DIR / "outputs"

TRACK1_KEYS = ["architecture", "paradigm", "source", "run", "dropout", "reward", "eval_dataset"]
TRACK2_KEYS = ["architecture", "paradigm", "source", "eval_dataset"]


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(path))


def per_row_oracle(group: pd.DataFrame) -> dict:
    """Compute oracle/worst/ranking for one row's CSF slice."""
    g = group.sort_values("augrc", ascending=True).reset_index(drop=True)
    head = g[g["side"] == "head"]
    feat = g[g["side"] == "feature"]
    out = {
        "n_csfs": len(g),
        "oracle_csf": g.iloc[0]["csf"],
        "oracle_augrc": float(g.iloc[0]["augrc"]),
        "worst_csf": g.iloc[-1]["csf"],
        "worst_augrc": float(g.iloc[-1]["augrc"]),
        "csf_ranking": g["csf"].tolist(),
        "augrc_ranking": [float(x) for x in g["augrc"].tolist()],
        "n_head": len(head),
        "oracle_csf_head": head.iloc[0]["csf"] if len(head) else None,
        "oracle_augrc_head": float(head.iloc[0]["augrc"]) if len(head) else None,
        "worst_csf_head": head.iloc[-1]["csf"] if len(head) else None,
        "worst_augrc_head": float(head.iloc[-1]["augrc"]) if len(head) else None,
        "n_feature": len(feat),
        "oracle_csf_feature": feat.iloc[0]["csf"] if len(feat) else None,
        "oracle_augrc_feature": float(feat.iloc[0]["augrc"]) if len(feat) else None,
        "worst_csf_feature": feat.iloc[-1]["csf"] if len(feat) else None,
        "worst_augrc_feature": float(feat.iloc[-1]["augrc"]) if len(feat) else None,
    }
    return out


def build_oracle_table(long_df: pd.DataFrame, row_keys: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in long_df.groupby(row_keys):
        rec = dict(zip(row_keys, keys))
        rec.update(per_row_oracle(g))
        # carry regime forward (it depends only on (source, eval_dataset))
        rec["regime"] = g.iloc[0]["regime"]
        rows.append(rec)
    return pd.DataFrame(rows)


def worked_example_section() -> str:
    lines = ["## Worked example — oracle, worst, per-side, regret variants\n\n"]
    lines.append(
        "Suppose one row at "
        "`(architecture=VGG13, paradigm=confidnet, source=cifar10, run=1, "
        "dropout=False, reward=2.2, eval_dataset=tinyimagenet)` "
        "contains 6 CSFs (3 head-side + 3 feature-side; toy size for "
        "illustration). Their AUGRC values:\n\n"
    )
    df = pd.DataFrame({
        "csf":   ["MSR", "Energy", "PE", "NeCo", "fDBD", "CTM"],
        "side":  ["head","head",   "head","feature","feature","feature"],
        "augrc": [100,   120,      180,   110,      130,      200],
    })
    lines.append("```\n" + df.to_string(index=False) + "\n```\n\n")

    g = df.sort_values("augrc").reset_index(drop=True)
    head = g[g["side"] == "head"]
    feat = g[g["side"] == "feature"]
    lines.append("### Sorted by AUGRC ascending (oracle first):\n\n")
    lines.append("```\n" + g.to_string(index=False) + "\n```\n\n")

    lines.append(
        f"- **oracle_csf** = `{g.iloc[0]['csf']}` (augrc = {g.iloc[0]['augrc']:g})\n"
        f"- **worst_csf** = `{g.iloc[-1]['csf']}` (augrc = {g.iloc[-1]['augrc']:g})\n"
        f"- **csf_ranking** = `{g['csf'].tolist()}`\n"
        f"- **augrc_ranking** = `{g['augrc'].tolist()}`\n\n"
    )
    lines.append(
        f"### Per-side restrictions (the §10e analysis pool):\n\n"
        f"- Head-side pool sorted: `{head['csf'].tolist()}` → "
        f"`oracle_csf_head` = `{head.iloc[0]['csf']}` "
        f"(augrc = {head.iloc[0]['augrc']:g}), "
        f"`worst_csf_head` = `{head.iloc[-1]['csf']}` "
        f"(augrc = {head.iloc[-1]['augrc']:g})\n"
        f"- Feature-side pool sorted: `{feat['csf'].tolist()}` → "
        f"`oracle_csf_feature` = `{feat.iloc[0]['csf']}` "
        f"(augrc = {feat.iloc[0]['augrc']:g}), "
        f"`worst_csf_feature` = `{feat.iloc[-1]['csf']}` "
        f"(augrc = {feat.iloc[-1]['augrc']:g})\n\n"
    )
    lines.append("### Regret variants computed downstream from this row\n\n")
    lines.append(
        "Suppose the predictor's score-based ranking on this row is "
        "`[NeCo, MSR, fDBD, Energy, PE, CTM]` (so its top-1 pick is NeCo).\n\n"
        "- **Top-1 regret** = `augrc(NeCo) − augrc(oracle)` = 110 − 100 = **10**.\n"
        "- **Normalized top-1 regret** = `10 / (worst − oracle)` = "
        "`10 / (200 − 100)` = **0.10**.\n"
        "- **Top-3 regret** with predicted top-3 = `[NeCo, MSR, fDBD]`: "
        "`min(110, 100, 130) − 100` = **0** (the oracle landed in the "
        "predicted top-3 even though top-1 was wrong).\n"
        "- **Set regret** for predicted competitive set `{NeCo, Energy}`: "
        "`min(110, 120) − 100` = **10**.\n\n"
        "On the head-side restricted pool, the oracle is MSR (100), worst is "
        "PE (180). If the predictor's head-side top-1 is Energy, head-side "
        "regret = 120 − 100 = 20 (normalized 20/80 = 0.25). On the "
        "feature-side pool, oracle is NeCo (110); if predictor's feature-side "
        "top-1 is fDBD, feature-side regret = 130 − 110 = 20 (normalized "
        "20/90 ≈ 0.22).\n\n"
    )
    lines.append(
        "All regret variants are computable from the columns this step writes "
        "(`oracle_*`, `worst_*`, per-side variants, and `csf_ranking` / "
        "`augrc_ranking`); the predictor's per-row score ordering is the only "
        "extra input needed at metric time (step 13).\n\n"
    )
    return "".join(lines)


def report(t1: pd.DataFrame | None, t2: pd.DataFrame | None,
           out_path: Path) -> None:
    lines = ["# Step 7 — Oracle and regret tables\n\n"]
    lines.append("**Date:** 2026-05-02\n")
    lines.append("**Source:** `code/nc_csf_predictivity/data/oracle_regret.py`\n\n")

    lines.append(worked_example_section())

    if t1 is not None:
        lines.append("## Track 1\n\n")
        lines.append(f"- Output: `outputs/track1/dataset/oracle.parquet`\n")
        lines.append(f"- Rows (one per regret-evaluable row): {len(t1):,}\n")
        lines.append(f"- Per-row CSF inventory size — distribution:\n\n```\n"
                     + t1["n_csfs"].describe().round(2).to_string() + "\n```\n\n")

        lines.append("### Oracle CSF frequency (top-10) per regime\n\n")
        for regime in ["near", "mid", "far", "test"]:
            sub = t1[t1["regime"] == regime]
            if sub.empty:
                continue
            counts = sub["oracle_csf"].value_counts().head(10)
            lines.append(f"**{regime}** (n = {len(sub):,}):\n\n")
            lines.append("```\n" + counts.to_string() + "\n```\n\n")

        lines.append("### Oracle CSF frequency by side per regime\n\n")
        for regime in ["near", "mid", "far", "test"]:
            sub = t1[t1["regime"] == regime].copy()
            if sub.empty:
                continue
            sub["oracle_side"] = sub["oracle_csf"].map(
                {**{c: "head" for c in
                    ["REN","PE","PCE","MSR","GEN","MLS","GE","GradNorm","Energy","Confidence","pNML"]},
                 **{c: "feature" for c in
                    ["PCA RecError global","NeCo","NNGuide","CTM","ViM","Maha","fDBD",
                     "KPCA RecError global","Residual"]}}
            )
            counts = sub["oracle_side"].value_counts(normalize=True).round(3)
            lines.append(f"**{regime}** — share of rows where oracle is head vs feature:\n\n")
            lines.append("```\n" + counts.to_string() + "\n```\n\n")

        lines.append("### Per-row dynamic range (worst − oracle) — distribution\n\n")
        t1_dr = (t1["worst_augrc"] - t1["oracle_augrc"])
        lines.append("```\n" + t1_dr.describe().round(2).to_string() + "\n```\n\n")
        lines.append(
            "This is the denominator of normalized regret. Cells with very "
            "small dynamic range (worst ≈ oracle) make normalized regret "
            "unstable — downstream metric code should drop or downweight "
            "those rows.\n\n"
        )

    if t2 is not None:
        lines.append("## Track 2\n\n")
        lines.append(f"- Output: `outputs/track2/dataset/oracle.parquet`\n")
        lines.append(f"- Rows: {len(t2):,}\n\n")
        lines.append("### Oracle CSF frequency (top-10) per regime\n\n")
        for regime in ["near", "mid", "far", "test"]:
            sub = t2[t2["regime"] == regime]
            if sub.empty:
                continue
            counts = sub["oracle_csf"].value_counts().head(10)
            lines.append(f"**{regime}** (n = {len(sub):,}):\n\n")
            lines.append("```\n" + counts.to_string() + "\n```\n\n")

    out_path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--track", choices=["1", "2", "all"], default="all")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    t1, t2 = None, None

    if args.track in ("1", "all"):
        in1 = out_root / "track1" / "dataset" / "long_harmonized.parquet"
        long1 = pq.read_table(in1).to_pandas()
        t1 = build_oracle_table(long1, TRACK1_KEYS)
        out1 = out_root / "track1" / "dataset" / "oracle.parquet"
        write_parquet(t1, out1)
        print(f"wrote {out1} ({len(t1):,} rows)")

    if args.track in ("2", "all"):
        in2 = out_root / "track2" / "dataset" / "long_harmonized.parquet"
        long2 = pq.read_table(in2).to_pandas()
        t2 = build_oracle_table(long2, TRACK2_KEYS)
        out2 = out_root / "track2" / "dataset" / "oracle.parquet"
        write_parquet(t2, out2)
        print(f"wrote {out2} ({len(t2):,} rows)")

    report(t1, t2, out_root / "05_oracle_check.md")
    print(f"wrote {out_root / '05_oracle_check.md'}")


if __name__ == "__main__":
    main()

"""ReAct and ASH-S as evaluated rows (camera-ready item 4.2).

The E-F pilot applied two activation wrappers to the nine head-side CSFs of
the VGG13 and ResNet18 pools at the pre-last-maxpool feature map: ReAct
(clipping at the 90th percentile of the ID validation activations of each
checkpoint) and ASH-S (pruning percentile 65). For every (model, OOD set,
CSF) the wrapped AUGRC is paired with the base-arm AUGRC of the same CSF.

Readouts per (architecture, wrapper, CSF): mean base AUGRC, mean and median
paired difference (wrapper - base, AUGRC x1000, positive = worse), share of
rows where the wrapper is better, and a two-sided Wilcoxon p. Per regime and
per source: the same pooled over the nine CSFs. Clique proxy: share of rows where the
best wrapped variant has a lower AUGRC than the best base CSF of the 22-CSF
roster (the per-row oracle); ties count, so this bounds the effect.

Clique test (swap design): for each wrapper, the nine base head-side CSFs of
the 22-CSF Figure 1 inputs are replaced by their wrapped variants (configs
selected on iid_val like every CSF), so the roster size stays 22, and the
VGG13 top cliques are recomputed with the Figure 1 pipeline
(`figure1_newcsf_cliques.py`). Adding the variants instead of swapping them
would change the Friedman-Conover result through the roster size alone and
would tie near-duplicate variants with their base CSFs. ResNet18 has one run
per cell, so it has no cliques.

Gates (abort before any number is computed):
  GW  the base arm of the aggregate reproduces the harmonized AUGRC of the
      nine CSFs on every paper-pool row (max |dAUGRC| <= 1e-6).
  GC  without wrappers, the pipeline reproduces the stored 22-CSF top
      cliques (outputs/40_figure1_newcsf_cliques_bo.md) in all 36 VGG13 OOD
      cells.

Inputs: code/new_csfs_pilot_summary_long.csv,
        outputs/track1/dataset/long_harmonized.parquet.
Outputs: outputs/react_ash/{per_csf,per_regime,per_source,cliques}.csv,
         outputs/51_react_ash.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy import stats as scstats

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent / "ablations"))
sys.path.insert(0, str(EVAL_DIR))
from calibration_features_clique import DEFAULT_OUT_ROOT, add_model_id  # noqa: E402
from figure1_newcsf_cliques import (  # noqa: E402
    CODE_DIR,
    SOURCES,
    clip_groups,
    load_scores_all,
    panel_cliques,
    select_new_csf_rows,
)
from newcsf_heads import AGGREGATE, ARCH, EXPERIMENT, MODE_TO_EVAL, SOURCE_NAMES  # noqa: E402
from regimefree_wilcoxon import load_parquet  # noqa: E402

WRAPPERS = ["ReAct", "ASH (ash_s@65)"]
ROSTER_22 = ["CTM", "Confidence", "Energy", "GE", "GEN", "GradNorm",
             "KPCA RecError global", "MLS", "MSR", "Maha", "MahaPP", "NCI", "NNGuide",
             "NeCo", "PCA RecError global", "PCE", "PE", "REN", "Residual", "ViM",
             "fDBD", "pNML"]
WRAPPER_SUFFIX = {"ReAct": " + ReAct", "ASH (ash_s@65)": " + ASH-S"}
DISPLAY = {"KPCA RecError global": "KPCA RecError", "PCA RecError global": "PCA RecError"}
STORED_CLIQUES = "40_figure1_newcsf_cliques_bo.md"
TOL = 1e-6


def load_rows(paper_models: set[str]) -> pd.DataFrame:
    """OOD AUGRC rows of the paper's CNN models, all arms, keyed like the long table."""
    agg = pd.read_csv(AGGREGATE, usecols=["experiment", "arm", "csf", "mode", "AUGRC"])
    agg = agg[agg["mode"].isin(MODE_TO_EVAL)].copy()
    agg["csf"] = agg["csf"].str.replace("_RecError_global", " RecError global")
    parts = agg["experiment"].str.extract(EXPERIMENT)
    keep = parts["arch"].notna()
    agg, parts = agg[keep], parts[keep]
    agg["architecture"] = parts["arch"].map(ARCH)
    agg["model_id"] = (agg["architecture"] + "|" + parts["paradigm"] + "|"
                       + parts["source"].replace(SOURCE_NAMES) + "|" + parts["run"]
                       + "|" + parts["do"] + "|"
                       + parts["rew"].astype(float).map(lambda r: f"{r:g}"))
    agg["eval_dataset"] = agg["mode"].map(MODE_TO_EVAL)
    return agg[agg["model_id"].isin(paper_models)].rename(columns={"AUGRC": "augrc"})


def gate_base(rows: pd.DataFrame, long_df: pd.DataFrame, csfs: list[str]) -> None:
    """GW: the base arm reproduces the harmonized AUGRC of the wrapped CSFs.

    Raises:
        RuntimeError: On missing rows or any deviation above TOL.
    """
    ref = long_df[long_df["csf"].isin(csfs) & (long_df["regime"] != "test")]
    base = rows[rows["arm"] == "base"]
    both = ref.merge(base, on=["model_id", "eval_dataset", "csf"], how="left",
                     suffixes=("", "_agg"))
    missing = int(both["augrc_agg"].isna().sum())
    dev = float((both["augrc"] - both["augrc_agg"]).abs().max())
    if missing or dev > TOL:
        raise RuntimeError(f"GW failed: {missing} rows missing, max |dAUGRC| {dev:.3g}")
    print(f"GW: {len(both):,} base rows reproduced, max |dAUGRC| {dev:.1e}")


def summarize(paired: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean base AUGRC, paired difference, wrapper win share, and Wilcoxon p."""
    def one(group: pd.DataFrame) -> pd.Series:
        diff = group["diff"]
        return pd.Series({"rows": len(group), "base": group["base"].mean(),
                          "mean_diff": diff.mean(), "median_diff": diff.median(),
                          "wrapper_better": (diff < 0).mean(),
                          "p_wilcoxon": scstats.wilcoxon(diff).pvalue})
    return paired.groupby(keys).apply(one, include_groups=False).reset_index()


def stored_cliques(out_root: Path) -> dict[tuple[str, str, str], list[str]]:
    """Extended (22-CSF) top cliques of the stored Figure 1 report, CNN panels.

    The report marks new CSFs in bold; the markers are stripped.
    """
    cells = {}
    for line in (out_root / STORED_CLIQUES).read_text().splitlines():
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) == 5 and parts[0] in ("confidnet", "devries", "dg"):
            cells[tuple(parts[:3])] = sorted(m.strip().strip("*") for m in parts[4].split(","))
    return cells


def wrapper_cliques(out_root: Path, wrapped: list[str]) -> pd.DataFrame:
    """VGG13 OOD top cliques of the 22-CSF roster and of each swapped roster.

    Per (wrapper, cell): `gained` lists wrapped variants whose base CSF is not
    in the 22-CSF clique, `lost` lists base CSFs of the clique whose variant
    is not in the swapped clique, `side_effects` lists other CSFs that enter
    (+) or leave (-).

    Raises:
        RuntimeError: If the 22-CSF run differs from the stored cliques
            (gate GC).
    """
    stored = stored_cliques(out_root)
    recs, mismatches = [], []
    for source in SOURCES:
        groups = clip_groups(CODE_DIR / "clip_scores", source)
        rows22 = pd.concat([load_scores_all(source, "Conv"),
                            select_new_csf_rows(AGGREGATE, source, "Conv")])
        plain = panel_cliques(rows22, "Conv", source, groups)
        swapped = {arm: panel_cliques(pd.concat([
            rows22[~rows22["methods"].isin(wrapped)],
            select_new_csf_rows(AGGREGATE, source, "Conv", arm=arm,
                                csfs=tuple(wrapped), suffix=suffix)]),
            "Conv", source, groups) for arm, suffix in WRAPPER_SUFFIX.items()}
        for (panel, regime), res in plain.items():
            if regime == "test":
                continue
            base = sorted(DISPLAY.get(m, m) for m in res["top_clique"])
            if base != stored[(panel, source, regime)]:
                mismatches.append((panel, source, regime))
            for arm, suffix in WRAPPER_SUFFIX.items():
                swap = sorted(DISPLAY.get(m, m)
                              for m in swapped[arm][(panel, regime)]["top_clique"])
                plain_of = {m: m.removesuffix(suffix) for m in swap}
                recs.append({
                    "wrapper": arm, "panel": panel, "source": source, "regime": regime,
                    "clique_22": ", ".join(base), "clique_swapped": ", ".join(swap),
                    "gained": ", ".join(m for m in swap
                                        if m.endswith(suffix) and plain_of[m] not in base),
                    "lost": ", ".join(m for m in base if m in wrapped
                                      and m + suffix not in swap),
                    "side_effects": ", ".join(
                        [f"+{m}" for m in swap if not m.endswith(suffix) and m not in base]
                        + [f"-{m}" for m in base if m not in wrapped and m not in swap])})
    if mismatches or len(recs) != 72:
        raise RuntimeError(f"GC failed: {len(recs)} records, mismatches {mismatches}")
    print("GC: 36 VGG13 OOD cliques of the 22-CSF roster reproduced")
    return pd.DataFrame(recs)


def main() -> None:
    """Gate, paired differences, clique proxy, and the report."""
    out_root = Path(DEFAULT_OUT_ROOT)
    long_df = add_model_id(load_parquet(out_root / "track1" / "dataset"
                                        / "long_harmonized.parquet"))
    long_df = long_df[long_df["architecture"].isin(ARCH.values())]
    rows = load_rows(set(long_df["model_id"]))
    wrapped = sorted(rows.loc[rows["arm"].isin(WRAPPERS), "csf"].unique())
    gate_base(rows, long_df, wrapped)
    cliques = wrapper_cliques(out_root, wrapped)

    regime = long_df[["model_id", "eval_dataset", "regime"]].drop_duplicates()
    key = ["architecture", "model_id", "eval_dataset", "csf"]
    base = rows[rows["arm"] == "base"].set_index(key)["augrc"].rename("base")
    paired = (rows[rows["arm"].isin(WRAPPERS)].join(base, on=key)
              .merge(regime, on=["model_id", "eval_dataset"]))
    paired["diff"] = paired["augrc"] - paired["base"]
    per_csf = summarize(paired, ["architecture", "arm", "csf"])
    per_regime = summarize(paired, ["architecture", "arm", "regime"])
    per_source = summarize(paired.assign(source=paired["model_id"].str.split("|").str[2]),
                           ["architecture", "arm", "source"])

    row_key = ["architecture", "model_id", "eval_dataset"]
    oracle = (rows[(rows["arm"] == "base") & rows["csf"].isin(ROSTER_22)]
              .groupby(row_key)["augrc"].min().rename("oracle"))
    best_wrapped = (paired.groupby(row_key + ["arm"])["augrc"].min()
                    .reset_index().join(oracle, on=row_key))
    proxy = (best_wrapped.assign(beats=best_wrapped["augrc"] < best_wrapped["oracle"])
             .groupby(["architecture", "arm"])["beats"].mean().rename("beats_oracle"))

    csv_dir = out_root / "react_ash"
    csv_dir.mkdir(parents=True, exist_ok=True)
    per_csf.to_csv(csv_dir / "per_csf.csv", index=False)
    per_regime.to_csv(csv_dir / "per_regime.csv", index=False)
    per_source.to_csv(csv_dir / "per_source.csv", index=False)
    cliques.to_csv(csv_dir / "cliques.csv", index=False)
    counts = cliques.groupby("wrapper")[["gained", "lost", "side_effects"]].agg(
        lambda c: int((c != "").sum()))
    lines = ["# ReAct and ASH-S on the CNN pools (camera-ready item 4.2)\n\n",
             "**Source:** `nc_csf_predictivity/evaluation/react_ash_table.py`. Paired "
             "AUGRC x1000 (wrapper - base, positive = worse) over the OOD rows of the "
             "paper's VGG13 and ResNet18 models; Wilcoxon two-sided.\n\n",
             f"Gates GW and GC passed. Wrapped CSFs: {', '.join(wrapped)}.\n\n",
             "## Per CSF\n\n```\n", per_csf.round(4).to_string(index=False), "\n```\n\n",
             "## Per regime (nine CSFs pooled)\n\n```\n",
             per_regime.round(4).to_string(index=False), "\n```\n\n",
             "## Per source (nine CSFs pooled)\n\n```\n",
             per_source.round(4).to_string(index=False), "\n```\n\n",
             "## Share of rows where the best wrapped variant beats the 22-CSF oracle\n\n```\n",
             proxy.round(4).to_string(), "\n```\n\n",
             "## VGG13 top cliques, swap design: cells (of 36) with a gained variant, "
             "a lost base CSF, or a side effect\n\n```\n", counts.to_string(),
             "\n```\n\n```\n",
             cliques[(cliques[["gained", "lost", "side_effects"]] != "").any(axis=1)]
             .drop(columns=["clique_22", "clique_swapped"]).to_string(index=False),
             "\n```\n"]
    (out_root / "51_react_ash.md").write_text("".join(lines))
    print(f"wrote {out_root / '51_react_ash.md'}")


if __name__ == "__main__":
    main()

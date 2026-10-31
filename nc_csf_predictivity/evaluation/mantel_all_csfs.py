"""Mantel results of Appendix G on the paper's CSF roster, against the reported ones.

The reported Mantel tests (`mantel_analysis.py`, `mantel_partial_convergence.py`)
filter only the projection variants, so they rank 22 methods in every stratum:
the paper's 20 CSFs plus CTMmean and CTMmeanOC, which the paper does not define
and `stats_eval.py` excludes. On ViT they also keep the Confidence row, which
the ViT panel of Figure 1 excludes. This script reruns them, with only the
roster changed, on:

  reported   as in the paper: 22 methods in every stratum;
  20         the paper's roster without Maha++ and NCI (19 on ViT);
  22         the paper's full roster (21 on ViT): the sandbox scores of
             `figure1_render_newcsf.py`, which has to run first.

Decision rule (fixed before the run; camera-ready checklist, 2026-10-30):
  M1  all four global Mantel tests p <= 0.05 (rank distance, all OOD sets).
  M2  the ID-performance control stays significant in all four strata, and the
      source-identity control in the same three strata (not Deep Gamblers).
  M3  mid is the strongest regime in the strata where the paper reports it.
  If M1 to M3 hold on the 22 roster, moving Appendix G to that roster changes
  numbers but no claim, and it also removes the two undefined CTM variants. If
  one fails, the failure is reported and the choice stays open.

Gates (abort before the comparison is written):
  GP  the reported roster reproduces every stored partial Mantel r and p.
  GT  the reported roster reproduces Tables tab:mantel_global and
      tab:mantel_per_metric of Appendix G to their printed precision.
  GN  every roster ranks the expected number of methods.

The NC1 sensitivity of `nc1_sensitivity.py` is repeated on the 22 roster, with
gate GL: its L-raw row reproduces the 22-roster partial results.

Run from `code/`; `--only-nc1` reuses the stored rosters. Outputs:
outputs/mantel_all_csfs/{global,per_metric,partial,nc1_variants_22}.csv,
outputs/58_mantel_all_csfs.md
"""
from __future__ import annotations

import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

import pandas as pd  # noqa: E402

CODE_DIR = Path(__file__).resolve().parents[2]
OUT_ROOT = CODE_DIR / "nc_csf_predictivity" / "outputs"
OUT_DIR = OUT_ROOT / "mantel_all_csfs"
CTM_VARIANTS = {"CTMmean", "CTMmeanOC"}
SCORES = {"reported": "scores_risk", "20": "scores_risk",
          "22": "fig1_newcsf_sandbox/scores_risk"}
STRATA = [("Conv", "confidnet"), ("Conv", "devries"), ("Conv", "dg"), ("ViT", "vit")]
N_METHODS = {"reported": {"Conv": 22, "ViT": 22}, "20": {"Conv": 20, "ViT": 19},
             "22": {"Conv": 22, "ViT": 21}}
TEX_GLOBAL = {"confidnet": (0.64, 0.55, 0.67), "devries": (0.68, 0.36, 0.52),
              "dg": (0.31, 0.47, 0.34), "vit": (0.59, 0.37, 0.49)}
TEX_PER_METRIC = {
    "var_collapse": (0.38, 0.45, 0.44, 0.51), "equiangular_uc": (0.65, 0.71, 0.20, 0.55),
    "equiangular_wc": (0.73, 0.56, 0.68, 0.44), "equinorm_uc": (0.51, 0.46, 0.25, 0.48),
    "equinorm_wc": (0.25, 0.17, 0.09, 0.34), "max_equiangular_uc": (0.72, 0.55, 0.16, 0.50),
    "max_equiangular_wc": (0.80, 0.75, 0.43, 0.70), "self_duality": (0.34, 0.26, 0.17, 0.58)}
ROUND_TOL = 0.0051


def run_cell(task: tuple[str, str, str]) -> list[dict]:
    """Global, per-regime, per-metric, and partial Mantel for one (roster, stratum)."""
    roster, backbone, study = task
    os.chdir(CODE_DIR)
    sys.path[:0] = [str(CODE_DIR / "archived"), str(CODE_DIR)]
    from loguru import logger
    logger.remove()
    import mantel_analysis
    import mantel_partial_convergence as partial
    from nc_regime_analysis import load_scores as real_load

    def load(_dir, metric, backbone_, mcd, datasets):
        scores = real_load(SCORES[roster], metric, backbone_, mcd, datasets)
        if roster == "reported":
            return scores
        drop = CTM_VARIANTS | ({"Confidence"} if backbone_ == "ViT" else set())
        return scores[~scores["methods"].isin(drop)]

    mantel_analysis.load_scores = load
    partial.load_scores = load
    out = OUT_DIR / "raw" / roster
    out.mkdir(parents=True, exist_ok=True)
    sys.argv = ["mantel_analysis.py", "--backbone", backbone, "--study", study,
                "--filter-methods", "--papyan-only", "--score-metric", "AUGRC",
                "--ood-group", "--output-dir", str(out)]
    mantel_analysis.main()
    return [{**row, "roster": roster} for row in partial.run_stratum(backbone, study)]


def collect(tasks: list[tuple[str, str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the global and per-metric CSVs of every cell."""
    glob, metric = [], []
    for roster, backbone, study in tasks:
        tag = f"AUGRC_{backbone}_MCD-False_{study}.csv"
        raw = OUT_DIR / "raw" / roster
        glob.append(pd.read_csv(raw / f"mantel_group_comparison_{tag}")
                    .assign(roster=roster, study=study))
        metric.append(pd.read_csv(raw / f"mantel_per_metric_group_all_{tag}")
                      .assign(roster=roster, study=study))
    return pd.concat(glob, ignore_index=True), pd.concat(metric, ignore_index=True)


def gates(glob: pd.DataFrame, metric: pd.DataFrame, partial: pd.DataFrame) -> None:
    """GP, GT, and GN.

    Raises:
        RuntimeError: If a gate fails.
    """
    stored = pd.read_csv(CODE_DIR / "mantel_partial_outputs" / "partial_mantel_summary.csv")
    both = stored.merge(partial[partial["roster"] == "reported"],
                        on=["backbone", "study", "control"], suffixes=("", "_new"))
    dev = float((both[["r", "p"]].to_numpy() - both[["r_new", "p_new"]].to_numpy())
                .__abs__().max())
    if len(both) != len(stored) or dev > 1e-12:
        raise RuntimeError(f"GP failed: {len(both)} of {len(stored)} rows, max |d| {dev:.3g}")
    print(f"GP: stored partial Mantel reproduced ({len(both)} rows, max |d| {dev:.1e})")

    rep = glob[(glob["roster"] == "reported") & (glob["ood_group"] == "all")]
    bad = []
    for study, values in TEX_GLOBAL.items():
        for test, value in zip(("spearman_rank", "jaccard_top1", "jaccard_top3"), values):
            r = rep[(rep["study"] == study) & (rep["test"] == f"full_NC_vs_{test}")]["r"].iloc[0]
            if abs(r - value) > ROUND_TOL:
                bad.append((study, test, r, value))
    rep_m = metric[(metric["roster"] == "reported") & (metric["method_distance"] == "spearman_rank")]
    for name, values in TEX_PER_METRIC.items():
        for study, value in zip(TEX_GLOBAL, values):
            r = rep_m[(rep_m["study"] == study) & (rep_m["nc_metric"] == name)]["r"].iloc[0]
            if abs(r - value) > ROUND_TOL:
                bad.append((study, name, r, value))
    if bad:
        raise RuntimeError(f"GT failed: {bad}")
    print("GT: Appendix G global and per-metric tables reproduced to printed precision")

    for roster, by_backbone in N_METHODS.items():
        for _, study in STRATA:
            backbone = "ViT" if study == "vit" else "Conv"
            n = glob[(glob["roster"] == roster) & (glob["study"] == study)]["n_methods"].iloc[0]
            if n != by_backbone[backbone]:
                raise RuntimeError(f"GN failed: {roster}/{study} ranks {n} methods")
    print("GN: method counts as expected")


def rule(glob: pd.DataFrame, partial: pd.DataFrame, roster: str) -> dict[str, bool]:
    """M1, M2, and M3 for one roster."""
    rank = glob[(glob["roster"] == roster) & (glob["test"] == "full_NC_vs_spearman_rank")]
    full = rank[rank["ood_group"] == "all"]
    par = partial[partial["roster"] == roster].set_index(["study", "control"])["p"]
    studies = list(TEX_GLOBAL)
    m2 = (all(par[(s, "iid_msr")] <= 0.05 for s in studies)
          and all((par[(s, "dataset")] <= 0.05) == (s != "dg") for s in studies))
    mid_best = {s: rank[(rank["study"] == s) & (rank["ood_group"].isin(["near", "mid", "far"]))]
                .set_index("ood_group")["r"].idxmax() for s in studies}
    return {"M1": bool((full["p"] <= 0.05).all()), "M2": bool(m2),
            "M3_mid_best": ", ".join(f"{s}:{g}" for s, g in mid_best.items())}


def report(glob: pd.DataFrame, metric: pd.DataFrame, partial: pd.DataFrame) -> str:
    """Markdown tables of the three rosters."""
    rank = glob[glob["test"] == "full_NC_vs_spearman_rank"]
    t_global = rank[rank["ood_group"] == "all"].pivot(index="study", columns="roster", values="r")
    t_p = rank[rank["ood_group"] == "all"].pivot(index="study", columns="roster", values="p")
    jac = glob[(glob["ood_group"] == "all") & glob["test"].isin(
        ["full_NC_vs_jaccard_top1", "full_NC_vs_jaccard_top3"])].pivot_table(
        index=["study", "test"], columns="roster", values="r")
    regime = rank[rank["ood_group"] != "all"].pivot_table(
        index=["study", "ood_group"], columns="roster", values="r")
    m = metric[metric["method_distance"] == "spearman_rank"].pivot_table(
        index=["nc_metric", "study"], columns="roster", values="r")
    par = partial.pivot_table(index=["study", "control"], columns="roster", values="r")
    par_p = partial.pivot_table(index=["study", "control"], columns="roster", values="p")
    order = list(SCORES)
    verdicts = "\n".join(f"- {r}: {rule(glob, partial, r)}" for r in order)
    blocks = [("Global Mantel r, rank distance", t_global), ("Global Mantel p", t_p),
              ("Global Mantel r, Jaccard", jac), ("Per regime r, rank distance", regime),
              ("Per metric r, rank distance", m), ("Partial Mantel r", par), ("Partial Mantel p", par_p)]
    body = "".join(f"## {name}\n\n```\n{frame[order].round(4).to_string()}\n```\n\n"
                   for name, frame in blocks)
    return ("# Mantel results on the paper's CSF roster\n\n**Source:** "
            "`nc_csf_predictivity/evaluation/mantel_all_csfs.py` (decision rule in the "
            "docstring, fixed before the run). Gates GP, GT, GN passed. Rosters: reported = "
            "22 methods including CTMmean and CTMmeanOC (and Confidence on ViT); 20 = paper "
            "roster without Maha++ and NCI (19 on ViT); 22 = paper roster (21 on ViT).\n\n"
            f"## Decision rule\n\n{verdicts}\n\n" + body)


def nc1_variants(partial: pd.DataFrame) -> pd.DataFrame:
    """Global and partial Mantel of every NC1 variant on the 22 roster.

    Raises:
        RuntimeError: If L-raw differs from the 22-roster results (gate GL).
    """
    os.chdir(CODE_DIR)
    for path in (CODE_DIR / "nc_csf_predictivity" / "evaluation",
                 CODE_DIR / "nc_csf_predictivity" / "ablations", CODE_DIR / "archived", CODE_DIR):
        sys.path.insert(0, str(path))
    from loguru import logger
    logger.remove()
    import mantel_partial_convergence as mpc
    from nc1_sensitivity import VARIANTS, mantel_variant
    from nc_regime_analysis import load_nc_metrics, load_scores as real_load

    def load(_dir, metric, backbone, mcd, datasets):
        scores = real_load(SCORES["22"], metric, backbone, mcd, datasets)
        drop = CTM_VARIANTS | ({"Confidence"} if backbone == "ViT" else set())
        return scores[~scores["methods"].isin(drop)]

    mpc.load_scores = load
    panel = (pd.read_csv(OUT_ROOT / "nc1_corrected" / "nc1_corrected.csv")
             .query("pool in ['VGG13', 'ResNet18', 'ViT']").set_index("model_id"))
    inputs = {"papyan": panel["nc1_corrected"],
              "kappa": panel["nc1_corrected"] * panel["n_classes"] / (panel["n_classes"] - 1) ** 2}
    nc = load_nc_metrics("neural_collapse_metrics/nc_metrics.csv")
    stored = partial[partial["roster"] == "22"][["backbone", "study", "control", "r", "p"]]
    rows = pd.concat([mantel_variant(name, nc, inputs, stored) for name in VARIANTS])
    print("GL: L-raw reproduces the 22-roster partial results")
    return rows


def main() -> None:
    """Run all cells, gate, and report."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if "--only-nc1" in sys.argv:
        glob = pd.read_csv(OUT_DIR / "global.csv", dtype={"roster": str})
        metric = pd.read_csv(OUT_DIR / "per_metric.csv", dtype={"roster": str})
        partial = pd.read_csv(OUT_DIR / "partial.csv", dtype={"roster": str})
    else:
        tasks = [(roster, backbone, study) for roster in SCORES for backbone, study in STRATA]
        with ProcessPoolExecutor(max_workers=6) as pool:
            partial = pd.DataFrame([row for rows in pool.map(run_cell, tasks) for row in rows])
        glob, metric = collect(tasks)
        gates(glob, metric, partial)
        glob.to_csv(OUT_DIR / "global.csv", index=False)
        metric.to_csv(OUT_DIR / "per_metric.csv", index=False)
        partial.to_csv(OUT_DIR / "partial.csv", index=False)
        shutil.rmtree(OUT_DIR / "raw")
    variants = nc1_variants(partial)
    variants.to_csv(OUT_DIR / "nc1_variants_22.csv", index=False)
    keep = variants[variants["control"].isin(["(none, replication)", "iid_msr", "dataset"])]
    extra = "".join(f"## NC1 variants on the 22 roster: {name}\n\n```\n"
                    f"{keep.pivot_table(index=['study', 'control'], columns='variant', values=col).round(4).to_string()}"
                    "\n```\n\n" for name, col in (("r", "r"), ("p", "p")))
    (OUT_ROOT / "58_mantel_all_csfs.md").write_text(report(glob, metric, partial) + extra)
    print(f"wrote {OUT_ROOT / '58_mantel_all_csfs.md'}")


if __name__ == "__main__":
    main()

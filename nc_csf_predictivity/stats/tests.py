"""Step 15: Statistical tests menu (protocol §11).

Tests run per (track, split):

  A. Paired Wilcoxon — for every (NC predictor, baseline) pair, on per-row
     regret. alt='less' (NC predictor's regret is hypothesized smaller).
     Holm-Bonferroni within (split, regime, side).
  B. Permutation test — does NC carry non-random information about which CSF
     is the oracle? Shuffle oracle labels R=5000 times; compare to observed
     regression top-1 accuracy.
  C. Conditional Friedman within NC bins — k=3 KMeans on standardized NC;
     per-bin Friedman χ² on AUGRC ranks vs unconditional χ². Reduction in χ²
     supports NC as a relevant covariate.
  D. Mantel test — pairwise NC Euclidean distance vs pairwise per-CSF
     AUGRC-vector cosine distance. Pearson r + permutation p.
  E. Spearman / Kendall ρ on per-row CSF rankings — bootstrap CIs aggregating
     the per-row values from step 13.
  F. McNemar / paired Wilcoxon — multilabel vs per-CSF binary head on the
     same test rows (set-regret).
  G. Multinomial logistic LR — null vs NC-feature multinomial logit on
     oracle CSF; chi² LR test.
  H. Bootstrap CIs on mean regret — already in step 13/14 aggregates;
     summarized here.

Reduced scope (per protocol greenlight): tests A, E, F, H run on all splits;
tests B, C, D, G run on xarch and lopo only.

Outputs:
  outputs/<track>/<split>/stats.json
  outputs/13_stats_check.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats as scstats
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

DATA_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = DATA_DIR.parent
CODE_DIR = PIPELINE_DIR.parent
DEFAULT_OUT_ROOT = PIPELINE_DIR / "outputs"

sys.path.insert(0, str(CODE_DIR))
from src.utils_stats import friedman_blocked  # noqa: E402

NC_PRIMARY = [
    "var_collapse", "equiangular_uc", "equiangular_wc",
    "equinorm_uc", "equinorm_wc", "max_equiangular_uc",
    "max_equiangular_wc", "self_duality",
]

ALL_SPLITS_T1 = ["xarch", "lopo", "lodo_vgg13", "pxs_vgg13",
                 "single_vgg13", "lopo_cnn_only"]
GLOBAL_TEST_SPLITS = ["xarch", "lopo"]
TRACK2_SPLITS = ["track2_loo"]
N_PERM = 5000
N_MANTEL_PERM = 999
SEED = 0


def add_id_track1(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["model_id"] = (
        df["architecture"].astype(str) + "|"
        + df["paradigm"].astype(str) + "|"
        + df["source"].astype(str) + "|"
        + df["run"].astype(int).astype(str) + "|"
        + df["dropout"].astype(int).astype(str) + "|"
        + df["reward"].apply(lambda x: f"{x:g}")
    )
    return df


# ---- A. Paired Wilcoxon (predictor vs baselines) ----

def wilcoxon_predictor_vs_baselines(per_row: pd.DataFrame, id_col: str
                                    ) -> list[dict]:
    """Per (regime, side): paired Wilcoxon of NC predictor vs each baseline."""
    out = []
    pivot_keys = [id_col, "eval_dataset", "regime", "side"]

    for (regime, side), grp in per_row.groupby(["regime", "side"]):
        # Reshape: per (id, eval, side), one column per (kind, name)
        grp_p = grp.copy()
        grp_p["comp"] = grp_p["comparator_kind"] + "::" + grp_p["comparator_name"]
        wide = grp_p.pivot_table(index=pivot_keys, columns="comp",
                                 values="regret_raw", aggfunc="first").reset_index()

        baselines = [c for c in wide.columns if str(c).startswith("baseline::")]
        # Use predictor_imputed for fair always-predicts comparison; fall back to predictor_raw
        predictors = [c for c in wide.columns if str(c).startswith("predictor_imputed::")
                      or str(c).startswith("predictor_raw::regression")]

        rows = []
        for pred in predictors:
            for base in baselines:
                d = wide[[pred, base]].dropna()
                if len(d) < 5:
                    continue
                diff = d[pred].values - d[base].values
                if np.allclose(diff, 0):
                    W, p = float("nan"), 1.0
                else:
                    try:
                        W, p = scstats.wilcoxon(diff, alternative="less",
                                                zero_method="wilcox")
                    except ValueError:
                        W, p = float("nan"), float("nan")
                rows.append({
                    "regime": regime, "side": side,
                    "predictor": pred, "baseline": base,
                    "n": int(len(d)),
                    "median_diff": float(np.median(diff)),
                    "W": float(W) if not np.isnan(W) else None,
                    "p": float(p) if not np.isnan(p) else None,
                })

        # Holm-Bonferroni within this (regime, side)
        if rows:
            ps = [r["p"] for r in rows if r["p"] is not None]
            if ps:
                reject, p_holm, _, _ = multipletests([r["p"] if r["p"] is not None else 1.0
                                                      for r in rows], method="holm")
                for i, r in enumerate(rows):
                    r["p_holm"] = float(p_holm[i])
                    r["reject_holm_05"] = bool(reject[i])
        out.extend(rows)
    return out


# ---- B. Permutation test (NC ↔ oracle association) ----

def permutation_nc_oracle(reg_preds: pd.DataFrame, oracle: pd.DataFrame,
                          id_col: str) -> dict:
    """Observed top-1 accuracy of regression vs accuracy under permuted
    oracle labels."""
    # Per (id, eval), top-1 predicted CSF
    key_cols = [id_col, "eval_dataset"]
    p_sorted = reg_preds.sort_values(key_cols + ["predicted_score"])
    top1 = p_sorted.groupby(key_cols).first().reset_index()[
        key_cols + ["csf"]].rename(columns={"csf": "pred_top1"})
    or_simple = oracle[key_cols + ["oracle_csf"]]
    merged = top1.merge(or_simple, on=key_cols, how="inner")
    if merged.empty:
        return {"status": "no data"}
    obs_acc = float((merged["pred_top1"] == merged["oracle_csf"]).mean())

    rng = np.random.default_rng(SEED)
    perm_accs = np.empty(N_PERM)
    oracle_arr = merged["oracle_csf"].values
    n = len(oracle_arr)
    for i in range(N_PERM):
        shuffled = oracle_arr[rng.permutation(n)]
        perm_accs[i] = (merged["pred_top1"].values == shuffled).mean()
    p = (np.sum(perm_accs >= obs_acc) + 1) / (N_PERM + 1)
    return {
        "n": int(n),
        "observed_acc": obs_acc,
        "perm_acc_mean": float(perm_accs.mean()),
        "perm_acc_p95": float(np.percentile(perm_accs, 95)),
        "perm_p_value": float(p),
        "n_perm": N_PERM,
    }


# ---- C. Conditional Friedman within NC bins ----

def conditional_friedman(long_df: pd.DataFrame, oracle: pd.DataFrame,
                         id_col: str, k_bins: int = 3) -> dict:
    """KMeans on standardized NC at the model level, per-bin Friedman χ²
    on AUGRC ranks across (eval × csf) blocks vs unconditional χ²."""
    nc_cols = NC_PRIMARY
    nc_per_model = (long_df[[id_col] + nc_cols]
                    .drop_duplicates(subset=[id_col]).reset_index(drop=True))
    if len(nc_per_model) < k_bins * 2:
        return {"status": "too few rows"}

    scaler = StandardScaler()
    Xs = scaler.fit_transform(nc_per_model[nc_cols].values)
    km = KMeans(n_clusters=k_bins, random_state=SEED, n_init=10)
    bins = km.fit_predict(Xs)
    nc_per_model["nc_bin"] = bins
    bin_map = nc_per_model.set_index(id_col)["nc_bin"].to_dict()

    work = long_df[[id_col, "eval_dataset", "csf", "augrc"]].copy()
    work["nc_bin"] = work[id_col].map(bin_map)

    # Build score = -augrc so higher = better
    work["score"] = -work["augrc"]
    work["block"] = work[id_col].astype(str) + "|" + work["eval_dataset"].astype(str)

    # Unconditional Friedman χ²
    try:
        chi_un, p_un, pivot_un = friedman_blocked(
            work, entity_col="csf", block_col="block", value_col="score")
    except Exception:
        chi_un, p_un = float("nan"), float("nan")

    per_bin = {}
    for b in sorted(work["nc_bin"].dropna().unique()):
        sub = work[work["nc_bin"] == b]
        try:
            chi_b, p_b, _ = friedman_blocked(
                sub, entity_col="csf", block_col="block", value_col="score")
        except Exception:
            chi_b, p_b = float("nan"), float("nan")
        per_bin[int(b)] = {
            "chi2": float(chi_b) if not np.isnan(chi_b) else None,
            "p": float(p_b) if not np.isnan(p_b) else None,
            "n_models_in_bin": int((nc_per_model["nc_bin"] == b).sum()),
        }
    chi_per_bin = [v["chi2"] for v in per_bin.values() if v["chi2"] is not None]
    avg_chi_bin = float(np.mean(chi_per_bin)) if chi_per_bin else None
    ratio = (avg_chi_bin / chi_un) if (avg_chi_bin and chi_un) else None
    return {
        "k_bins": k_bins,
        "n_models": int(len(nc_per_model)),
        "unconditional_chi2": float(chi_un) if not np.isnan(chi_un) else None,
        "unconditional_p": float(p_un) if not np.isnan(p_un) else None,
        "per_bin": per_bin,
        "avg_per_bin_chi2": avg_chi_bin,
        "ratio_avg_to_unconditional": ratio,
    }


# ---- D. Mantel test ----

def mantel_test(long_df: pd.DataFrame, id_col: str,
                max_models: int = 100) -> dict:
    """Pairwise NC distance vs pairwise per-CSF AUGRC-vector cosine distance.
    Subsample to `max_models` models for tractability."""
    nc_per_model = (long_df[[id_col] + NC_PRIMARY]
                    .drop_duplicates(subset=[id_col]).reset_index(drop=True))
    if len(nc_per_model) > max_models:
        nc_per_model = nc_per_model.sample(max_models, random_state=SEED).reset_index(drop=True)
    ids = nc_per_model[id_col].tolist()

    # NC distance matrix (Euclidean on standardized NC)
    Xs = StandardScaler().fit_transform(nc_per_model[NC_PRIMARY].values)
    n = len(ids)
    D_nc = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D_nc[i, j] = D_nc[j, i] = float(np.linalg.norm(Xs[i] - Xs[j]))

    # AUGRC-vector per model: mean raw_augrc per CSF averaged over eval rows
    aug_long = long_df[long_df[id_col].isin(ids)]
    per_model_csf = (aug_long.groupby([id_col, "csf"])["augrc"].mean()
                     .reset_index())
    pivot = per_model_csf.pivot(index=id_col, columns="csf", values="augrc")
    pivot = pivot.reindex(ids).fillna(pivot.mean(axis=0))  # per-CSF mean fill if absent

    D_perf = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D_perf[i, j] = D_perf[j, i] = float(cosine(pivot.iloc[i].values,
                                                       pivot.iloc[j].values))

    iu = np.triu_indices(n, k=1)
    nc_vec = D_nc[iu]
    perf_vec = D_perf[iu]
    obs_r = float(np.corrcoef(nc_vec, perf_vec)[0, 1])

    rng = np.random.default_rng(SEED)
    perm_rs = np.empty(N_MANTEL_PERM)
    for i in range(N_MANTEL_PERM):
        perm = rng.permutation(n)
        D_perf_p = D_perf[perm][:, perm]
        perf_perm = D_perf_p[iu]
        perm_rs[i] = np.corrcoef(nc_vec, perf_perm)[0, 1]
    p = (np.sum(perm_rs >= obs_r) + 1) / (N_MANTEL_PERM + 1)
    return {
        "n_models": int(n),
        "n_pairs": int(len(nc_vec)),
        "observed_r": obs_r,
        "perm_r_mean": float(perm_rs.mean()),
        "perm_p_value": float(p),
        "n_perm": N_MANTEL_PERM,
    }


# ---- E. Spearman / Kendall (aggregate from step 13) ----

def spearman_aggregate(per_row_metrics: pd.DataFrame) -> list[dict]:
    """Per (regime, side): mean and 95% bootstrap CI of regression per-row
    Spearman ρ and Kendall τ."""
    rows = []
    reg = per_row_metrics[per_row_metrics["predictor"] == "regression"]
    for (regime, side), g in reg.groupby(["regime", "side"]):
        for col in ("spearman_rho", "kendall_tau", "mrr"):
            if col not in g.columns:
                continue
            vals = g[col].dropna().values
            if len(vals) == 0:
                continue
            mean = float(vals.mean())
            rng = np.random.default_rng(SEED)
            idx = rng.integers(0, len(vals), size=(2000, len(vals)))
            boot = vals[idx].mean(axis=1)
            rows.append({
                "regime": regime, "side": side, "metric": col,
                "n": int(len(vals)),
                "mean": mean,
                "ci_lo": float(np.percentile(boot, 2.5)),
                "ci_hi": float(np.percentile(boot, 97.5)),
            })
    return rows


# ---- F. McNemar / paired Wilcoxon (per_csf vs multilabel) ----

def per_csf_vs_multilabel(per_row: pd.DataFrame, id_col: str
                          ) -> list[dict]:
    """Paired Wilcoxon comparing set_regret of per_csf_binary vs multilabel
    on the same (id, eval, regime, side, label_rule). Two-sided test."""
    pr = per_row.copy()
    # Parse "multilabel/<rule>" or "per_csf_binary/<rule>" into family + rule
    parts = pr["comparator_name"].str.split("/", n=1, expand=True)
    pr["family"] = parts[0]
    pr["label_rule"] = parts[1].fillna("")
    pr = pr[pr["family"].isin(["multilabel", "per_csf_binary"])]
    rows = []
    keys = ["regime", "side", "label_rule"]
    for (regime, side, label_rule), grp in pr.groupby(keys):
        if not isinstance(label_rule, str) or label_rule == "":
            continue
        ml = grp[grp["family"] == "multilabel"]
        pc = grp[grp["family"] == "per_csf_binary"]
        if ml.empty or pc.empty:
            continue
        # Use predictor_imputed if available (always-predicts), else raw
        ml_imp = ml[ml["comparator_kind"] == "predictor_imputed"]
        pc_imp = pc[pc["comparator_kind"] == "predictor_imputed"]
        if not ml_imp.empty and not pc_imp.empty:
            ml_use = ml_imp
            pc_use = pc_imp
            kind = "imputed"
        else:
            ml_use = ml[ml["comparator_kind"] == "predictor_raw"]
            pc_use = pc[pc["comparator_kind"] == "predictor_raw"]
            kind = "raw"
        m_a = ml_use[[id_col, "eval_dataset", "regret_raw"]].rename(
            columns={"regret_raw": "ml_r"})
        p_b = pc_use[[id_col, "eval_dataset", "regret_raw"]].rename(
            columns={"regret_raw": "pc_r"})
        merged = m_a.merge(p_b, on=[id_col, "eval_dataset"]).dropna()
        if len(merged) < 5:
            continue
        diff = merged["pc_r"].values - merged["ml_r"].values
        if np.allclose(diff, 0):
            W, p = float("nan"), 1.0
        else:
            try:
                W, p = scstats.wilcoxon(diff, alternative="two-sided",
                                        zero_method="wilcox")
            except ValueError:
                W, p = float("nan"), float("nan")
        rows.append({
            "regime": regime, "side": side, "label_rule": label_rule,
            "comparator_kind": kind, "n": int(len(merged)),
            "median_diff_pc_minus_ml": float(np.median(diff)),
            "W": float(W) if not np.isnan(W) else None,
            "p": float(p) if not np.isnan(p) else None,
        })
    return rows


# ---- G. Multinomial LR test ----

def multinomial_lr(long_df: pd.DataFrame, oracle: pd.DataFrame,
                   id_col: str) -> dict:
    """Null vs full multinomial logit on (NC, oracle CSF). LR statistic and df."""
    or_keys = [id_col, "eval_dataset"]
    or_small = oracle[or_keys + ["oracle_csf", "regime"]]
    nc_per_model = (long_df[[id_col] + NC_PRIMARY]
                    .drop_duplicates(subset=[id_col]).reset_index(drop=True))
    merged = or_small.merge(nc_per_model, on=id_col, how="inner")
    if merged.empty:
        return {"status": "no data"}

    Xs = StandardScaler().fit_transform(merged[NC_PRIMARY].values)
    y = merged["oracle_csf"].values
    classes = np.unique(y)
    K = len(classes)
    if K < 2:
        return {"status": "single class"}

    full = LogisticRegression(max_iter=2000, solver="lbfgs")
    full.fit(Xs, y)
    proba_full = full.predict_proba(Xs)
    ll_full = -log_loss(y, proba_full, labels=classes, normalize=False)

    # Null = intercept only (predict marginal class probability)
    counts = pd.Series(y).value_counts(normalize=True)
    null_proba = np.tile(counts[classes].values, (len(y), 1))
    ll_null = -log_loss(y, null_proba, labels=classes, normalize=False)

    LR = float(2 * (ll_full - ll_null))
    df = (len(NC_PRIMARY)) * (K - 1)
    p = float(scstats.chi2.sf(LR, df))
    return {
        "n": int(len(y)),
        "K_classes": int(K),
        "n_features": int(len(NC_PRIMARY)),
        "LR_statistic": LR,
        "df": int(df),
        "p_value": p,
    }


# ---- Driver ----

def run_track1(out_root: Path) -> dict:
    long_df = pq.read_table(out_root / "track1" / "dataset" / "long_harmonized.parquet").to_pandas()
    long_df = add_id_track1(long_df)
    long_df_ood = long_df[long_df["regime"] != "test"].copy()

    oracle_df = pq.read_table(out_root / "track1" / "dataset" / "oracle.parquet").to_pandas()
    oracle_df = add_id_track1(oracle_df)
    oracle_df_ood = oracle_df[oracle_df["regime"] != "test"].copy()

    summary = {}
    for split in ALL_SPLITS_T1:
        sp_path = out_root / "splits" / f"{split}.parquet"
        if not sp_path.exists():
            continue
        sp = pq.read_table(sp_path).to_pandas()
        test_ids = sp[sp["role"] == "test"]["model_id"].unique()
        eval_test = long_df_ood[long_df_ood["model_id"].isin(test_ids)]
        oracle_test = oracle_df_ood[oracle_df_ood["model_id"].isin(test_ids)]

        baseline_pr_path = out_root / "track1" / split / "baselines" / "per_row.parquet"
        per_row_metrics_path = out_root / "track1" / split / "metrics" / "per_row.parquet"
        if not baseline_pr_path.exists():
            continue
        baseline_pr = pq.read_table(baseline_pr_path).to_pandas()
        per_row_metrics = (pq.read_table(per_row_metrics_path).to_pandas()
                           if per_row_metrics_path.exists() else pd.DataFrame())

        results: dict = {"split": split, "track": 1, "tests": {}}

        # A. Wilcoxon
        results["tests"]["wilcoxon"] = wilcoxon_predictor_vs_baselines(
            baseline_pr, "model_id")

        # E. Spearman aggregate
        if not per_row_metrics.empty:
            results["tests"]["spearman_aggregate"] = spearman_aggregate(
                per_row_metrics)

        # F. per_csf vs multilabel
        results["tests"]["per_csf_vs_multilabel"] = per_csf_vs_multilabel(
            baseline_pr, "model_id")

        if split in GLOBAL_TEST_SPLITS:
            # B. Permutation
            reg_preds_path = out_root / "track1" / split / "regression" / "preds.parquet"
            if reg_preds_path.exists():
                reg_preds = pq.read_table(reg_preds_path).to_pandas()
                results["tests"]["permutation_nc_oracle"] = permutation_nc_oracle(
                    reg_preds, oracle_test, "model_id")

            # C. Conditional Friedman (on TEST pool to mirror evaluation)
            results["tests"]["conditional_friedman"] = conditional_friedman(
                eval_test, oracle_test, "model_id")

            # D. Mantel
            results["tests"]["mantel"] = mantel_test(eval_test, "model_id",
                                                    max_models=100)

            # G. Multinomial LR
            results["tests"]["multinomial_lr"] = multinomial_lr(
                eval_test, oracle_test, "model_id")

        out_dir = out_root / "track1" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "stats.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        summary[split] = {
            "n_wilcoxon_tests": len(results["tests"].get("wilcoxon", [])),
            "n_per_csf_vs_multilabel": len(results["tests"].get("per_csf_vs_multilabel", [])),
            "has_permutation": "permutation_nc_oracle" in results["tests"],
            "has_mantel": "mantel" in results["tests"],
        }
        print(f"  track1/{split}: {summary[split]}")
    return summary


def report(out_root: Path, t1_summary: dict, out_path: Path) -> None:
    lines = ["# Step 15 — Statistical tests menu\n\n"]
    lines.append("**Date:** 2026-05-04\n")
    lines.append("**Source:** `code/nc_csf_predictivity/stats/tests.py`\n")
    lines.append(f"**Permutation R:** {N_PERM}; **Mantel R:** {N_MANTEL_PERM}; "
                 f"**seed:** {SEED}\n\n")

    # Headline: xarch Wilcoxon table — NC predictors that beat Always-CTM
    lines.append("## Headline — xarch paired Wilcoxon (predictor vs Always-CTM)\n\n")
    json_path = out_root / "track1" / "xarch" / "stats.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        wil = data["tests"]["wilcoxon"]
        rows = [r for r in wil if "Always-CTM" in r["baseline"]]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df[["regime", "side", "predictor", "n", "median_diff",
                     "W", "p", "p_holm", "reject_holm_05"]].round(4)
            df["predictor"] = df["predictor"].str.replace("predictor_imputed::", "imp/")\
                .str.replace("predictor_raw::", "raw/")
            df = df.sort_values(["regime", "side", "p"])
            lines.append("```\n" + df.to_string(index=False) + "\n```\n\n")
            lines.append(
                "Reading: `predictor` is `imp/<predictor_name>` (always-"
                "predicts via empty-set imputation) or `raw/regression`. "
                "`median_diff` is `regret(predictor) − regret(Always-CTM)`; "
                "negative = NC predictor wins. `p_holm` is Holm-Bonferroni "
                "corrected within (regime, side). `reject_holm_05` flags "
                "predictors that significantly beat Always-CTM at α=0.05 "
                "after correction.\n\n"
            )

    # Permutation test on xarch
    lines.append("## Permutation test (NC ↔ oracle association)\n\n")
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        if "permutation_nc_oracle" in data["tests"]:
            d = data["tests"]["permutation_nc_oracle"]
            lines.append(
                f"On xarch test set (n={d['n']:,}): observed top-1 acc = "
                f"**{d['observed_acc']:.4f}**. Under permuted oracle labels: "
                f"mean acc = {d['perm_acc_mean']:.4f}, 95% percentile = "
                f"{d['perm_acc_p95']:.4f}. Permutation p-value = "
                f"**{d['perm_p_value']:.4f}**.\n\n"
            )

    # Multinomial LR test
    lines.append("## Multinomial logistic LR test (NC features → oracle CSF)\n\n")
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        if "multinomial_lr" in data["tests"]:
            d = data["tests"]["multinomial_lr"]
            lines.append(
                f"On xarch test rows (n={d['n']:,}, K={d['K_classes']} oracle "
                f"classes, {d['n_features']} NC features): LR statistic = "
                f"**{d['LR_statistic']:.2f}** on df = {d['df']}, "
                f"p = **{d['p_value']:.2e}**.\n\n"
            )

    # Mantel test
    lines.append("## Mantel test (NC pairwise distance vs AUGRC-vector cosine)\n\n")
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        if "mantel" in data["tests"]:
            d = data["tests"]["mantel"]
            lines.append(
                f"On {d['n_models']} sampled models from xarch test pool "
                f"({d['n_pairs']:,} pairs): observed Mantel r = "
                f"**{d['observed_r']:.4f}**. Mean perm r = "
                f"{d['perm_r_mean']:.4f}, perm p = **{d['perm_p_value']:.4f}**.\n\n"
            )

    # Conditional Friedman
    lines.append("## Conditional Friedman within NC bins\n\n")
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        if "conditional_friedman" in data["tests"]:
            d = data["tests"]["conditional_friedman"]
            lines.append(
                f"k=3 KMeans on standardized NC for {d['n_models']} models. "
                f"Unconditional χ² = **{d['unconditional_chi2']:.2f}**, "
                f"avg per-bin χ² = **{d['avg_per_bin_chi2']:.2f}**, "
                f"ratio = {d['ratio_avg_to_unconditional']:.3f}.\n\n"
            )
            for b, v in d["per_bin"].items():
                lines.append(f"- bin {b} (n={v['n_models_in_bin']}): "
                             f"χ² = {v['chi2']:.2f}, p = {v['p']:.2e}\n")
            lines.append("\n")
            ratio = d["ratio_avg_to_unconditional"]
            if ratio is not None and ratio < 1:
                lines.append(
                    "Per-bin χ² < unconditional χ² ⇒ conditioning on NC "
                    "reduces CSF-rank disagreement, consistent with NC being "
                    "a relevant covariate.\n\n"
                )

    # Spearman aggregate (regression ranking)
    lines.append("## Spearman / Kendall ρ (regression per-row rankings)\n\n")
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        if "spearman_aggregate" in data["tests"]:
            sp = pd.DataFrame(data["tests"]["spearman_aggregate"])
            sp = sp[sp["metric"] == "spearman_rho"].sort_values(["regime", "side"])
            lines.append("Spearman ρ on xarch regression per-row CSF orderings, "
                         "with bootstrap 95% CI:\n\n")
            lines.append("```\n" + sp[["regime","side","n","mean","ci_lo","ci_hi"]]
                         .round(3).to_string(index=False) + "\n```\n\n")

    # per_csf vs multilabel
    lines.append("## per_csf_binary vs multilabel (paired Wilcoxon, two-sided)\n\n")
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        if "per_csf_vs_multilabel" in data["tests"]:
            df = pd.DataFrame(data["tests"]["per_csf_vs_multilabel"])
            if not df.empty:
                df = df.sort_values(["regime","side","label_rule"]).round(4)
                lines.append("```\n" + df.to_string(index=False) + "\n```\n\n")
                lines.append(
                    "`median_diff_pc_minus_ml > 0` ⇒ per_csf has higher "
                    "regret (worse). Two-sided p tests whether the two "
                    "predictors give systematically different set-regret "
                    "on the same rows.\n\n"
                )

    out_path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    print("Track 1:")
    t1 = run_track1(out_root)

    report(out_root, t1, out_root / "13_stats_check.md")
    print(f"wrote {out_root / '13_stats_check.md'}")


if __name__ == "__main__":
    main()

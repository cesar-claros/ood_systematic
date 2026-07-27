"""Pool A local analysis: probes, CSFs, cliques, descriptors, H1/H2 tests.

Consumes the cached frozen features written by `extract_features.py`
(`{encoder}_{dataset}_{split}.npz`) and runs the full pilot on CPU:

  1. per (encoder, source, seed): linear probe (last-5k val carve-out),
     temperature scaling, CSF fitting on the validation slice (the pipeline's
     Stage-1 convention) with per-cell hyperparameter selection by validation
     failure-AUGRC;
  2. AUGRC/AURC per (CSF, eval set) under the paper's protocol (OOD vs
     correctly classified ID test samples) via `src.rc_stats`;
  3. Friedman-Conover top cliques per (encoder, source) cell through the
     paper's own pipeline (`cliques_track1.compute_track1_cliques`);
  4. Papyan NC metrics (formulas mirrored from `src/neural_collapse.py`) and
     X8 weak-collapse descriptors per probe model;
  5. H1: within-pool Mantel, classical NC vs extended descriptor vector;
  6. H2/H4: the VGG-13-trained per-CSF clique predictor applied to the probe
     pool without retraining; set-regret vs pool oracle and Always-X baselines.

Excluded CSFs for frozen probes: `Confidence` (needs a trained auxiliary
head; the paper's ViT panels already drop it) and KPCA RecError (deferred).

Usage:
  ./.venv/bin/python x8_pool_a/pool_a_analysis.py --features-dir pool_a_features
  ./.venv/bin/python x8_pool_a/pool_a_analysis.py --synthetic   # self-test
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from loguru import logger

CODE_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "x8_pool_a"))
sys.path.insert(0, str(CODE_DIR / "nc_csf_predictivity" / "data"))
sys.path.insert(0, str(CODE_DIR / "nc_csf_predictivity" / "ablations"))

import pool_a_csfs as csf
from probes_and_descriptors import descriptors
from src.rc_stats import RiskCoverageStats
from cliques_track1 import compute_track1_cliques
from calibration_features_clique import (
    NC_PRIMARY,
    add_model_id,
    build_pipeline,
)

warnings.filterwarnings("ignore")

ENCODERS = ["dinov2_vitb14", "clip_vitb16"]
SOURCES = ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]
CLIP_TO_NPZ = {"lsun resize": "lsun_resize", "lsun cropped": "lsun_cropped"}
VAL_N = 5000
PCA_DIMS = [32, 64, 128, 256]
NNG_KS = [10, 50]
HEAD_CSFS = ["MSR", "MLS", "Energy", "PE", "GEN", "REN", "GE", "PCE",
             "GradNorm", "pNML"]
FEAT_CSFS = ["CTM", "Maha", "NNGuide", "fDBD", "PCA RecError global",
             "Residual", "ViM", "NeCo"]
# Rebuttal additions, evaluated on the pool but excluded from the H2
# predictor's candidate sides (the VGG-trained heads carry no labels for
# them, so including them would inflate the oracle unfairly).
NEW_CSFS = ["MahaPP", "NCI"]
ALWAYS_BASELINES = ["MSR", "Energy", "MLS", "CTM", "fDBD", "NNGuide"]
DESC_COLS = ["rho_res", "n_residue_spikes", "median_cos_own",
             "class_dep_residue", "common_mode_energy"]
OUT_ROOT = CODE_DIR / "nc_csf_predictivity" / "outputs"


def mantel_test(D1: np.ndarray, D2: np.ndarray, n_perms: int = 9999,
                method: str = "spearman") -> dict:
    """Permutation Mantel test (verbatim from mantel_analysis.py, which is
    not importable on the cluster because it depends on the untracked
    archived/ tree): Spearman correlation of upper-triangle distances,
    fixed seed 42, one-sided count with the +1 convention."""
    from scipy.stats import pearsonr, spearmanr
    n = D1.shape[0]
    idx = np.triu_indices(n, k=1)
    d1, d2 = D1[idx], D2[idx]
    corr = spearmanr if method == "spearman" else pearsonr
    r_obs, _ = corr(d1, d2)
    rng = np.random.default_rng(42)
    count_ge = 0
    for _ in range(n_perms):
        perm = rng.permutation(n)
        r_perm, _ = corr(d1, D2[np.ix_(perm, perm)][idx])
        if r_perm >= r_obs:
            count_ge += 1
    return {"r_obs": r_obs, "p_value": (count_ge + 1) / (n_perms + 1),
            "n_perms": n_perms}


def method_rank_distance(rank_matrix: np.ndarray) -> np.ndarray:
    """1 - Spearman correlation between per-model CSF rank vectors."""
    from scipy.stats import spearmanr
    n = rank_matrix.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(rank_matrix[i], rank_matrix[j])
            D[i, j] = D[j, i] = 1.0 - rho
    return D


def regime_map(source: str, clip_dir: pathlib.Path) -> dict[str, str]:
    """npz eval-set name -> near/mid/far for one source (paper Table 6)."""
    df = pd.read_csv(clip_dir / f"clip_distances_{source}.csv",
                     header=[0, 1], index_col=0)
    group_col = [c for c in df.columns if c[0] == "group"][0]
    groups = df[group_col].astype(int)
    names = {0: "test", 1: "near", 2: "mid", 3: "far"}
    return {CLIP_TO_NPZ.get(ds, ds): names[g]
            for ds, g in groups.items() if g > 0}


def papyan_nc(h_fit: np.ndarray, y_fit: np.ndarray, w_eff: np.ndarray,
              n_cls: int) -> dict[str, float]:
    """The 8 Papyan metrics, formulas mirrored from src/neural_collapse.py."""
    gmean = h_fit.mean(axis=0)
    means = np.stack([h_fit[y_fit == c].mean(axis=0) for c in range(n_cls)])
    m_cent = means - gmean
    sigma_b = m_cent.T @ m_cent / n_cls
    centered = h_fit - means[y_fit]
    sigma_w = centered.T @ centered / (len(h_fit) * n_cls)
    var_collapse = float(
        np.trace(sigma_w @ np.linalg.pinv(sigma_b, rcond=1e-6)) / n_cls)

    def cosines(a: np.ndarray) -> np.ndarray:
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        c = an @ an.T
        return c[~np.eye(n_cls, dtype=bool)]

    def eq_stats(a: np.ndarray) -> tuple[float, float, float]:
        off = cosines(a)
        norms = np.linalg.norm(a, axis=1)
        return (float(off.std(ddof=1)),
                float(np.abs(off + 1.0 / (n_cls - 1)).mean()),
                float(norms.std(ddof=1) / norms.mean()))

    eq_uc, max_uc, eqn_uc = eq_stats(m_cent)
    eq_wc, max_wc, eqn_wc = eq_stats(w_eff)
    m_t = m_cent / np.linalg.norm(m_cent)
    w_t = w_eff / np.linalg.norm(w_eff)
    return {"var_collapse": var_collapse,
            "equiangular_uc": eq_uc, "equiangular_wc": eq_wc,
            "equinorm_uc": eqn_uc, "equinorm_wc": eqn_wc,
            "max_equiangular_uc": max_uc, "max_equiangular_wc": max_wc,
            "self_duality": float(((w_t - m_t) ** 2).sum())}


def rc_metrics(confids: np.ndarray, residuals: np.ndarray) -> tuple[float, float]:
    """(AUGRC, AURC); RiskCoverageStats already applies the x1000 display scale."""
    rc = RiskCoverageStats(confids=confids, residuals=residuals.astype(float))
    return float(rc.augrc), float(rc.aurc)


def select_dim(score_fn, dims: list[int], conf_val_args: tuple,
               resid_val: np.ndarray) -> int:
    """Pick the hyperparameter minimizing validation failure-AUGRC."""
    best, best_a = dims[0], np.inf
    for d in dims:
        a, _ = rc_metrics(score_fn(*conf_val_args, d).cpu().numpy(), resid_val)
        if a < best_a:
            best, best_a = d, a
    return best


def run_model(enc: str, source: str, seed: int, feats: dict,
              eval_names: list[str], device: str,
              include_new: bool = False) -> tuple[list[dict], dict]:
    """Probe + CSFs + per-eval-set metrics for one (encoder, source, seed)."""
    t0 = time.perf_counter()
    h_train, y_train = feats[(source, "train")]
    h_test_np, y_test = feats[(source, "test")]
    n_cls = int(y_train.max()) + 1
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(h_train))
    val_n = min(VAL_N, len(h_train) // 5)
    fit_idx, val_idx = perm[:-val_n], perm[-val_n:]
    h_fit_np, y_fit_np = h_train[fit_idx], y_train[fit_idx]

    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(device)
    h_fit = to_t(h_fit_np)
    y_fit = torch.from_numpy(y_fit_np).long().to(device)
    h_val = to_t(h_train[val_idx])
    y_val = torch.from_numpy(y_train[val_idx]).long().to(device)
    h_test = to_t(h_test_np)

    probe = csf.train_probe(h_fit, y_fit, n_cls, seed=seed)
    w, b, mu, sd = probe["W"], probe["b"], probe["mu"], probe["sd"]
    zf = lambda h: (h - mu) / sd
    logits_of = lambda h: zf(h) @ w.T + b
    temp = csf.fit_temperature(logits_of(h_val), y_val)
    logger.info(f"{enc}/{source}/seed{seed}: probe acc={probe['acc']:.3f} "
                f"temp={temp:.2f} ({time.perf_counter() - t0:.1f}s)")

    t1 = time.perf_counter()
    lg_val = logits_of(h_val)
    resid_val = (lg_val.argmax(dim=1) != y_val).float().cpu().numpy()
    maha = csf.Mahalanobis(h_val, y_val, n_cls)
    w_eff = w / sd
    train_mean_raw = h_fit.mean(dim=0)
    # MahaPP/NCI are deferred until the E-F sweep completes on the CNN/ViT
    # benchmark, so the roster stays identical across pools; rerun with
    # --include-new-csfs to add them.
    maha_pp, nci_alpha = None, None
    if include_new:
        maha_pp = csf.Mahalanobis(csf.l2n(h_val), y_val, n_cls)
        nci_alpha = csf.fit_nci_alpha(h_val, lg_val, resid_val, w_eff,
                                      train_mean_raw, rc_metrics)
    sub = csf.Subspace(h_val)
    pnml = csf.PNML(zf(h_val))
    bank_scores = csf.conf_energy(lg_val, 1.0)
    mu_std = zf(h_fit).mean(dim=0)

    d_pca = select_dim(lambda h, d: sub.conf_pca_recerror(h, d),
                       PCA_DIMS, (h_val,), resid_val)
    d_res = select_dim(lambda h, d: sub.conf_residual(h, d),
                       PCA_DIMS, (h_val,), resid_val)
    d_neco = select_dim(lambda h, d: sub.conf_neco(h, d),
                        PCA_DIMS, (h_val,), resid_val)
    d_vim = d_res
    vim_alpha = sub.vim_alpha(h_val, lg_val, d_vim)
    k_nng = NNG_KS[0]
    best_a = np.inf
    for k in NNG_KS:
        nng = csf.NNGuide(h_val, bank_scores, k)
        conf_k = nng.conf(h_val, csf.conf_energy(lg_val, 1.0))
        a, _ = rc_metrics(conf_k.cpu().numpy(), resid_val)
        if a < best_a:
            best_a, k_nng = a, k
    nng = csf.NNGuide(h_val, bank_scores, k_nng)
    nci_note = f", nci_alpha={nci_alpha:g}" if include_new else ""
    logger.info(f"{enc}/{source}/seed{seed}: CSFs fit (dims pca={d_pca} "
                f"res={d_res} neco={d_neco}, nng_k={k_nng}{nci_note}) "
                f"({time.perf_counter() - t1:.1f}s)")

    def all_confs(h: torch.Tensor) -> dict[str, np.ndarray]:
        z = zf(h)
        lg = z @ w.T + b
        p = torch.softmax(lg / temp, dim=1)
        confs = {
            "MSR": csf.conf_msr(p), "MLS": csf.conf_mls(lg),
            "Energy": csf.conf_energy(lg, temp), "PE": csf.conf_pe(p),
            "GEN": csf.conf_gen(p, 0.1, min(100, n_cls)),
            "REN": csf.conf_ren(p, 0.5, n_cls),
            "GE": csf.conf_ge(p), "PCE": csf.conf_pce(p),
            "GradNorm": csf.conf_gradnorm(p, z), "pNML": pnml.conf(z, p),
            "CTM": csf.conf_ctm(z, w), "Maha": maha.conf(h),
            "NNGuide": nng.conf(h, csf.conf_energy(lg, 1.0)),
            "fDBD": csf.conf_fdbd(z, lg, w, mu_std),
            "PCA RecError global": sub.conf_pca_recerror(h, d_pca),
            "Residual": sub.conf_residual(h, d_res),
            "ViM": sub.conf_vim(h, lg, d_vim, vim_alpha, 1.0),
            "NeCo": sub.conf_neco(h, d_neco),
        }
        if include_new:
            confs["MahaPP"] = maha_pp.conf(csf.l2n(h))
            confs["NCI"] = csf.conf_nci(h, lg, w_eff, train_mean_raw,
                                        nci_alpha)
        return {k: v.cpu().numpy() for k, v in confs.items()}

    conf_test = all_confs(h_test)
    correct = (logits_of(h_test).argmax(dim=1).cpu().numpy() == y_test)
    rows = []
    base = {"paradigm": f"probe_{enc}", "source": source, "run": seed,
            "dropout": False, "reward": 0.0}
    for name, c in conf_test.items():
        a, u = rc_metrics(c, (~correct).astype(float))
        rows.append({**base, "csf": name, "eval_dataset": "iid",
                     "regime": "test", "augrc": a, "aurc": u})
    for ev in eval_names:
        t_ev = time.perf_counter()
        h_ood_np = feats[(ev, "eval")][0]
        conf_ood = all_confs(to_t(h_ood_np))
        for name in conf_test:
            confids = np.concatenate([conf_test[name][correct],
                                      conf_ood[name]])
            residuals = np.concatenate([np.zeros(int(correct.sum())),
                                        np.ones(len(h_ood_np))])
            a, u = rc_metrics(confids, residuals)
            rows.append({**base, "csf": name, "eval_dataset": ev,
                         "regime": None, "augrc": a, "aurc": u})
        logger.info(f"{enc}/{source}/seed{seed}: scored {ev} "
                    f"(n={len(h_ood_np):,}, "
                    f"{time.perf_counter() - t_ev:.1f}s)")

    w_eff_np = w_eff.cpu().numpy()
    desc = descriptors(h_fit_np, y_fit_np, n_cls)
    desc["common_mode_energy"] = desc.pop("common_mode")["energy_fraction"]
    model_row = {**base, "acc": float(correct.mean()), "temp": temp,
                 **papyan_nc(h_fit_np, y_fit_np, w_eff_np, n_cls),
                 **{k: v for k, v in desc.items() if k in DESC_COLS}}
    logger.info(f"{enc}/{source}/seed{seed}: done "
                f"({time.perf_counter() - t0:.1f}s total)")
    return rows, model_row


def h1_mantel(models_df: pd.DataFrame, long_df: pd.DataFrame,
              n_perms: int) -> pd.DataFrame:
    """Classical vs extended Mantel within the probe pool."""
    ood = long_df[long_df["regime"] != "test"]
    pivot = ood.pivot_table(index=["paradigm", "source", "run"],
                            columns="csf", values="augrc", aggfunc="mean")
    ranks = pivot.rank(axis=1).values
    key = models_df.set_index(["paradigm", "source", "run"])
    key = key.loc[pivot.index]

    def dist(cols: list[str]) -> np.ndarray:
        x = key[cols].values.astype(float)
        x = (x - x.mean(0)) / (x.std(0) + 1e-12)
        from scipy.spatial.distance import pdist, squareform
        return squareform(pdist(x))

    d_rank = method_rank_distance(ranks)
    rows = []
    for label, cols in [("classical NC (8)", NC_PRIMARY),
                        ("extended (8 + X8 descriptors)",
                         NC_PRIMARY + DESC_COLS)]:
        res = mantel_test(dist(cols), d_rank, n_perms=n_perms)
        rows.append({"vector": label, "n_models": len(key),
                     "r": res["r_obs"], "p": res["p_value"]})
    return pd.DataFrame(rows)


def pool_cliques_for(train_archs: tuple, benchmark_long: pd.DataFrame
                     ) -> pd.DataFrame:
    """Clique labels for the requested training architectures.

    VGG-13 CNN labels come from the published step-5 cliques; ViT labels are
    computed inline with the same Friedman-Conover pipeline (cached to
    cliques_vit.parquet) since that file is not tracked in the repo.
    """
    pieces = []
    if "VGG13" in train_archs:
        pieces.append(pd.read_parquet(
            OUT_ROOT / "track1" / "cliques" / "cliques.parquet"))
    if "ViT" in train_archs:
        vit_path = OUT_ROOT / "track1" / "cliques" / "cliques_vit.parquet"
        if vit_path.exists():
            pieces.append(pd.read_parquet(vit_path))
        else:
            logger.info("Computing ViT clique labels inline...")
            vit_rows = benchmark_long[
                (benchmark_long["architecture"] == "ViT")
                & (benchmark_long["paradigm"] == "modelvit")]
            flat, _ = compute_track1_cliques(vit_rows)
            flat.to_parquet(vit_path, index=False)
            pieces.append(flat)
    return pd.concat(pieces, ignore_index=True)


def h2_predictor(models_df: pd.DataFrame, long_df: pd.DataFrame,
                 n_perms: int, train_archs: tuple = ("VGG13",)
                 ) -> pd.DataFrame:
    """Benchmark-trained clique predictor applied to the probe pool.

    train_archs selects the training pool: the paper's VGG-13 CNNs, the
    fine-tuned ViTs (the weak-collapse regime closest to frozen probes), or
    both combined (the widest NC span).
    """
    vgg_long = pd.read_parquet(
        OUT_ROOT / "track1" / "dataset" / "long_harmonized.parquet")
    vgg_long = add_model_id(vgg_long)
    cliques = pool_cliques_for(train_archs, vgg_long)
    for arch, sub in vgg_long.groupby("architecture"):
        for c in NC_PRIMARY:
            vgg_long.loc[sub.index, c] = (
                (sub[c] - sub[c].mean()) / (sub[c].std() + 1e-12))
    label_wide = (cliques.assign(label=cliques["in_top_clique"].astype(int))
                  .pivot_table(index=["paradigm", "source", "dropout",
                                      "reward", "regime"],
                               columns="csf", values="label", aggfunc="first")
                  .reset_index().fillna(0))
    csf_cols = [c for c in label_wide.columns if c not in
                ["paradigm", "source", "dropout", "reward", "regime"]]
    vgg_models = (vgg_long[vgg_long["architecture"].isin(train_archs)]
                  [["model_id", "paradigm", "source", "dropout", "reward"]
                   + NC_PRIMARY].drop_duplicates("model_id"))
    train_rows = []
    for _, m in vgg_models.iterrows():
        for regime in ["near", "mid", "far", "all"]:
            train_rows.append({**m.to_dict(), "regime": regime})
    tr = pd.DataFrame(train_rows).merge(
        label_wide, on=["paradigm", "source", "dropout", "reward", "regime"],
        how="inner")

    pool = models_df.copy()
    for enc, sub in pool.groupby("paradigm"):
        for c in NC_PRIMARY:
            pool.loc[sub.index, c] = (
                (sub[c] - sub[c].mean()) / (sub[c].std() + 1e-12))
    test_rows = []
    for _, m in pool.iterrows():
        for regime in ["near", "mid", "far"]:
            test_rows.append({**m[["paradigm", "source", "run"]
                                  + NC_PRIMARY].to_dict(), "regime": regime})
    te = pd.DataFrame(test_rows)

    feats_cols = NC_PRIMARY + ["source", "regime"]
    preds = []
    for name in csf_cols:
        y = tr[name].astype(int).values
        if y.min() == y.max() or min(np.bincount(y)) < 5:
            continue
        pipe = build_pipeline("source")
        pipe.fit(tr[feats_cols], y)
        pr = te[["paradigm", "source", "run", "regime"]].copy()
        pr["csf"] = name
        pr["hit"] = pipe.predict_proba(te[feats_cols])[:, 1] >= 0.5
        preds.append(pr)
    preds = pd.concat(preds, ignore_index=True)
    logger.info(f"H2[{'+'.join(train_archs)}]: "
                f"{preds['csf'].nunique()} per-CSF heads trained")

    ood = long_df[long_df["regime"] != "test"].copy()
    sides = {"all": HEAD_CSFS + FEAT_CSFS, "head": HEAD_CSFS,
             "feature": FEAT_CSFS}
    out = []
    for side, side_csfs in sides.items():
        sub = ood[ood["csf"].isin(side_csfs)]
        oracle = (sub.groupby(["paradigm", "source", "run", "eval_dataset",
                               "regime"])["augrc"]
                  .agg(["min", "max"]).reset_index()
                  .rename(columns={"min": "o", "max": "wst"}))
        chosen = sub.merge(
            preds[preds["hit"]][["paradigm", "source", "run", "regime", "csf"]],
            on=["paradigm", "source", "run", "regime", "csf"], how="inner")
        smin = (chosen.groupby(["paradigm", "source", "run", "eval_dataset",
                                "regime"])["augrc"].min()
                .rename("set_min").reset_index())
        m = oracle.merge(smin, how="left",
                         on=["paradigm", "source", "run", "eval_dataset",
                             "regime"])
        m["regret"] = (m["set_min"].fillna(m["wst"]) - m["o"]).clip(lower=0)
        for regime, g in m.groupby("regime"):
            rec = {"train_pool": "+".join(train_archs),
                   "side": side, "regime": regime,
                   "predictor_regret": round(float(g["regret"].mean()), 2),
                   "empty_pct": round(100 * float(g["set_min"].isna().mean()), 1)}
            best_name, best_val = None, np.inf
            for x in ALWAYS_BASELINES:
                if x not in side_csfs:
                    continue
                bx = sub[(sub["csf"] == x) & (sub["regime"] == regime)]
                bxm = bx.merge(oracle, on=["paradigm", "source", "run",
                                           "eval_dataset", "regime"])
                val = float((bxm["augrc"] - bxm["o"]).clip(lower=0).mean())
                if val < best_val:
                    best_name, best_val = x, val
            rec["best_baseline"] = best_name
            rec["baseline_regret"] = round(best_val, 2)
            out.append(rec)
    return pd.DataFrame(out)


def make_synthetic(tmp: pathlib.Path, rng: np.random.Generator) -> None:
    """Small fake feature cache exercising the full pipeline."""
    dims = {"dinov2_vitb14": 96, "clip_vitb16": 80}
    n_cls = {"cifar10": 10, "cifar100": 20, "supercifar100": 19,
             "tinyimagenet": 30}
    oods = ["isun", "lsun_cropped", "lsun_resize", "svhn", "places365",
            "textures"]
    for enc, d in dims.items():
        residue = 1.4 if enc == "clip_vitb16" else 0.0
        q = np.linalg.qr(rng.standard_normal((d, d)))[0]
        bank = q[:, 60:75]
        for src, c in n_cls.items():
            mu = q[:, :c] * 3.0
            def draw(n, cls, shift=0.0, res=residue):
                y = rng.integers(0, cls, n)
                h = (mu[:, y].T + shift
                     + (rng.standard_normal((n, bank.shape[1])) * res) @ bank.T
                     + rng.standard_normal((n, d)) * 0.5)
                return h.astype(np.float32), y
            for split, n in [("train", 3000), ("test", 900)]:
                h, y = draw(n, c)
                np.savez(tmp / f"{enc}_{src}_{split}.npz", features=h, labels=y)
        for ev in oods:
            h = (rng.standard_normal((700, d)) * 1.2 + 1.5
                 + (rng.standard_normal((700, bank.shape[1])) * residue)
                 @ bank.T).astype(np.float32)
            np.savez(tmp / f"{enc}_{ev}_test.npz", features=h,
                     labels=np.zeros(700, dtype=int))


def main() -> None:
    """Run the Pool A pilot end to end and write the report."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features-dir", default=str(CODE_DIR / "pool_a_features"))
    ap.add_argument("--clip-dir", default=str(CODE_DIR / "clip_scores"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-perms", type=int, default=9999)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--include-new-csfs", action="store_true",
                    help="also evaluate MahaPP and NCI on the pool "
                         "(deferred by default until the E-F sweep completes "
                         "on the CNN/ViT benchmark, keeping rosters "
                         "identical across pools)")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    logger.info(f"Pool A analysis on device={args.device} "
                f"(torch {torch.__version__})")

    fdir = pathlib.Path(args.features_dir)
    if args.synthetic:
        fdir = pathlib.Path(__file__).parent / "_synthetic_features"
        fdir.mkdir(exist_ok=True)
        make_synthetic(fdir, np.random.default_rng(0))
        args.seeds, args.n_perms = 2, 999

    all_rows, model_rows = [], []
    for enc in ENCODERS:
        feats: dict = {}
        for f in fdir.glob(f"{enc}_*.npz"):
            stem = f.stem.replace(f"{enc}_", "")
            ds, split = stem.rsplit("_", 1)
            d = np.load(f)
            key = (ds, split if ds in SOURCES else "eval")
            feats[key] = (d["features"], d["labels"])
        for source in SOURCES:
            if (source, "train") not in feats:
                continue
            rmap = regime_map(source, pathlib.Path(args.clip_dir))
            eval_names = [e for e in rmap
                          if (e, "eval") in feats or (e, "test") in feats]
            for e in list(eval_names):
                if (e, "eval") not in feats:
                    feats[(e, "eval")] = feats[(e, "test")]
            for seed in range(args.seeds):
                rows, mrow = run_model(enc, source, seed, feats, eval_names,
                                       args.device,
                                       include_new=args.include_new_csfs)
                for r in rows:
                    if r["regime"] is None:
                        r["regime"] = rmap[r["eval_dataset"]]
                all_rows.extend(rows)
                model_rows.append(mrow)
            logger.info(f"=== {enc}/{source}: {args.seeds} probes done "
                        f"(mean acc {np.mean([m['acc'] for m in model_rows[-args.seeds:]]):.3f}) ===")

    long_df = pd.DataFrame(all_rows)
    models_df = pd.DataFrame(model_rows)
    out_dir = OUT_ROOT / "pool_a"
    out_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(out_dir / "long_pool_a.parquet", index=False)
    models_df.to_parquet(out_dir / "models_pool_a.parquet", index=False)

    logger.info("Computing pool cliques (Friedman-Conover per cell)...")
    flat, _ = compute_track1_cliques(long_df)
    flat.to_parquet(out_dir / "cliques_pool_a.parquet", index=False)
    top = (flat[flat["in_top_clique"] & flat["regime"].isin(
        ["near", "mid", "far"])]
        .groupby(["paradigm", "source", "regime"])["csf"]
        .apply(lambda s: ", ".join(sorted(s))).reset_index())

    logger.info("H1: Mantel classical vs extended descriptors...")
    h1 = h1_mantel(models_df, long_df, args.n_perms)
    h2_parts = []
    for pool in [("VGG13",), ("ViT",), ("VGG13", "ViT")]:
        logger.info(f"H2[{'+'.join(pool)}]: benchmark-trained predictor on "
                    "the probe pool (sklearn, CPU; the slowest stage)...")
        h2_parts.append(h2_predictor(models_df, long_df, args.n_perms,
                                     train_archs=pool))
    h2 = pd.concat(h2_parts, ignore_index=True)

    lines = ["# Pool A pilot (X8): frozen DINOv2/CLIP probes\n\n",
             "**Source:** `x8_pool_a/pool_a_analysis.py`"
             + (" (SYNTHETIC self-test)" if args.synthetic else "") + "\n\n",
             "## Probe accuracy and descriptors (mean per encoder x source)\n\n",
             "```\n" + models_df.groupby(["paradigm", "source"])
             [["acc", "var_collapse", "self_duality", "rho_res"]]
             .mean().round(3).to_string() + "\n```\n\n",
             "## Top cliques per (encoder, source, regime)\n\n",
             "```\n" + top.to_string(index=False) + "\n```\n\n",
             *(["## New CSFs on the SSL pool (mean AUGRC per encoder x regime)\n\n",
                "```\n" + (long_df[long_df["regime"].isin(["near", "mid", "far"])
                           & long_df["csf"].isin(NEW_CSFS + ["Maha", "NeCo",
                                                             "CTM", "Residual",
                                                             "ViM", "Energy"])]
                           .pivot_table(index="csf",
                                        columns=["paradigm", "regime"],
                                        values="augrc", aggfunc="mean")
                           .round(2).to_string()) + "\n```\n\n"]
               if long_df["csf"].isin(NEW_CSFS).any() else []),
             "## H1: Mantel, classical vs extended descriptors\n\n",
             "```\n" + h1.round(4).to_string(index=False) + "\n```\n\n",
             "## H2/H4: benchmark-trained predictors on the probe pool\n"
             "(train pools: VGG-13 CNNs, fine-tuned ViTs, and both; the ViT "
             "pool is the weak-collapse regime closest to frozen probes)\n\n",
             "```\n" + h2.to_string(index=False) + "\n```\n"]
    report = out_dir / ("27_pool_a_pilot_SYNTHETIC.md" if args.synthetic
                        else "27_pool_a_pilot.md")
    report.write_text("".join(lines))
    print(f"wrote {report}")


if __name__ == "__main__":
    main()

"""Harmonized SSL (Pool A) rerun: the paper-faithful CSF protocol on the
cached probe features (documentation/imagenet_scale_plan.md, harmonization
queue item 1).

What changes vs the submitted pilot (pool_a_analysis.py), per the
simplification ledger: temperature LBFGS (was 41-point grid); GEN/REN BO
over (M, gamma) (were fixed (0.1, 100) / (0.5)); NNGuide BO over
(k, bank proportion) (was full-bank k in {10, 50}); PCA RecError on
standardized features with BO'd explained variance and the relative error
(was raw + dim grid); Residual/ViM fixed ladder (DINOv2 d=768 -> 512,
CLIP d=512 -> 256; was dim grid); NeCo fixed 100 components, raw input
(probes are ViT-family) TIMES max logit (factor was missing); NCI alpha BO
over (0, 3e-2) (was 7-point grid); KPCA RecError ADDED (was deferred),
on raw features with probe-equivalent weights W_eff = W/sd,
b_eff = b - (mu/sd) @ W.T so its landmark energies use the probe's own
logits. Roster: 21 CSFs. What does NOT change: probe training (identical
seeds -> identical probes), the val carve-out serving as both the class-
statistics and tuning slice (the paper's Stage-1 convention), and the
untouched CSFs (Maha, MahaPP, CTM, fDBD), which double as replication
gates.

Gates before the report is written:
  G-a  probe accuracies match models_pool_a_newcsfs.parquet exactly;
  G-b  invariant CSFs (Maha, MahaPP, CTM, fDBD) reproduce
       long_pool_a_newcsfs.parquet per row (tol 0.02).

Report (41_pool_a_harmonized.md): per-CSF pooled drift; the regime-free
selector table on the SUBMITTED 18-CSF roster (drift vs GLiC-2's
4.96/3.99/5.07 and 1.59/1.38/0.59) and on the 21-CSF revision roster;
H1 Mantel old vs new; clique-change summary.

Run inside the x9 container (bayes_opt, torch 2.x; the chain is
fd-shifts-free) on a GPU node:
  python x8_pool_a/pool_a_harmonized.py --features-dir <pool_a_features>
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
from loguru import logger

CODE_DIR = pathlib.Path(__file__).resolve().parents[1]
for p in (CODE_DIR, CODE_DIR / "x8_pool_a", CODE_DIR / "x9_imagenet",
          CODE_DIR / "nc_csf_predictivity" / "data",
          CODE_DIR / "nc_csf_predictivity" / "ablations",
          CODE_DIR / "nc_csf_predictivity" / "evaluation"):
    sys.path.insert(0, str(p))

import pool_a_csfs as csf  # noqa: E402
from pool_a_analysis import (  # noqa: E402
    ENCODERS, OUT_ROOT, SOURCES, VAL_N, h1_mantel, pool_cliques_for,
    regime_map)
from cliques_track1 import compute_track1_cliques  # noqa: E402
from calibration_features_clique import NC_PRIMARY, add_model_id  # noqa: E402
from input_ablation_grid import REGIMES, evaluate, ssl_shortlists  # noqa: E402
from src.rc_stats import RiskCoverageStats  # noqa: E402
from kernel_pca_port import KernelPCAPort  # noqa: E402
from csf_protocol import (  # noqa: E402
    fit_temperature_lbfgs, ladder_dim, make_neco_conf, neco_flags,
    tune_entropy_pair, tune_nci_alpha, tune_nnguide, tune_pca_re)

INVARIANT_CSFS = ["Maha", "MahaPP", "CTM", "fDBD"]
SUBMITTED = {"source_nr": (4.96, 3.99, 5.07), "none_nr": (1.59, 1.38, 0.59)}


def rc_metrics(confids, residuals) -> tuple[float, float]:
    rc = RiskCoverageStats(confids=confids,
                           residuals=np.asarray(residuals, dtype=float))
    return float(rc.augrc), float(rc.aurc)


def run_model(enc, source, seed, feats, eval_names, device, args):
    t0 = time.perf_counter()
    h_train, y_train = feats[(source, "train")]
    h_test_np, y_test = feats[(source, "test")]
    n_cls = int(y_train.max()) + 1
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(h_train))
    val_n = min(VAL_N, len(h_train) // 5)
    fit_idx, val_idx = perm[:-val_n], perm[-val_n:]

    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(device)
    h_fit = to_t(h_train[fit_idx])
    y_fit = torch.from_numpy(y_train[fit_idx]).long().to(device)
    h_val = to_t(h_train[val_idx])
    y_val = torch.from_numpy(y_train[val_idx]).long().to(device)
    y_val_np = y_train[val_idx]
    h_test = to_t(h_test_np)

    probe = csf.train_probe(h_fit, y_fit, n_cls, seed=seed)
    w, b, mu, sd = probe["W"], probe["b"], probe["mu"], probe["sd"]
    zf = lambda h: (h - mu) / sd            # probe-native space
    logits_of = lambda h: zf(h) @ w.T + b
    lg_val = logits_of(h_val)
    resid_val = (lg_val.argmax(dim=1) != y_val).float().cpu().numpy()
    temp = fit_temperature_lbfgs(lg_val, y_val)          # CHANGED: LBFGS
    name = f"probe_{enc}"

    # unchanged fits (double as replication gates)
    maha = csf.Mahalanobis(h_val, y_val, n_cls)
    maha_pp = csf.Mahalanobis(csf.l2n(h_val), y_val, n_cls)
    w_eff = w / sd
    b_eff = b - (mu / sd) @ w.T
    train_mean_raw = h_fit.mean(dim=0)
    mu_std = zf(h_fit).mean(dim=0)
    pnml = csf.PNML(zf(h_val))

    # harmonized protocol (stats slice == tuning slice: Stage-1 convention)
    d_lad = ladder_dim(h_val.shape[1])
    sub = csf.Subspace(h_val)
    mu_v, sd_v = h_val.mean(0), h_val.std(0) + 1e-12
    zv = lambda h: (h - mu_v) / sd_v        # PCA-RE standardization (its own
    sub_z = csf.Subspace(zv(h_val))         # tss on the fitting slice)
    vim_alpha = sub.vim_alpha(h_val, lg_val, d_lad)
    pca_var, pca_dim = tune_pca_re(sub_z, zv(h_val), resid_val, rc_metrics,
                                   args.bo_init, args.bo_iters)
    p_val = torch.softmax(lg_val / temp, dim=1)
    gen_p, ren_p = tune_entropy_pair(csf, p_val, resid_val, n_cls,
                                     rc_metrics, args.bo_init, args.bo_iters)
    bank_scores = csf.conf_energy(lg_val, 1.0)
    nng, nng_p = tune_nnguide(csf, h_val, y_val_np, bank_scores, n_cls,
                              h_val, lg_val, resid_val, rc_metrics,
                              args.bo_init, args.bo_iters)
    nci_alpha = tune_nci_alpha(csf, h_val, lg_val, resid_val, w_eff,
                               train_mean_raw, rc_metrics,
                               args.bo_init, args.bo_iters)
    neco_raw, neco_mult = neco_flags(name)
    neco_conf = make_neco_conf(sub, sub_z, neco_raw, neco_mult)
    kpca = KernelPCAPort(w_eff, b_eff, n_cls)
    kpca_params = kpca.tune_hyperparameters(h_val, h_val, resid_val,
                                            rc_metrics, temperature=1.0,
                                            n_iters=args.bo_iters,
                                            n_init=args.bo_init)
    logger.info(
        f"{enc}/{source}/seed{seed}: harmonized fits (D={d_lad}, "
        f"pca {pca_var:.3f}/{pca_dim}, gen M={int(gen_p['M'])} "
        f"g={gen_p['gamma']:.3f}, nng k={int(nng_p['k_clusters'])} "
        f"prop={nng_p['proportion']:.2f}, nci a={nci_alpha:.4f}, "
        f"temp={temp:.2f}) ({time.perf_counter() - t0:.0f}s)")

    def all_confs(h: torch.Tensor) -> dict[str, np.ndarray]:
        z = zf(h)
        lg = z @ w.T + b
        p = torch.softmax(lg / temp, dim=1)
        confs = {
            "MSR": csf.conf_msr(p), "MLS": csf.conf_mls(lg),
            "Energy": csf.conf_energy(lg, temp), "PE": csf.conf_pe(p),
            "GEN": csf.conf_gen(p, gen_p["gamma"], int(gen_p["M"])),
            "REN": csf.conf_ren(p, ren_p["gamma"], int(ren_p["M"])),
            "GE": csf.conf_ge(p), "PCE": csf.conf_pce(p),
            "GradNorm": csf.conf_gradnorm(p, z), "pNML": pnml.conf(z, p),
            "CTM": csf.conf_ctm(z, w), "Maha": maha.conf(h),
            "MahaPP": maha_pp.conf(csf.l2n(h)),
            "NNGuide": nng.conf(h, csf.conf_energy(lg, 1.0)),
            "fDBD": csf.conf_fdbd(z, lg, w, mu_std),
            "PCA RecError global": sub_z.conf_pca_recerror(zv(h), pca_dim),
            "Residual": sub.conf_residual(h, d_lad),
            "ViM": sub.conf_vim(h, lg, d_lad, vim_alpha, 1.0),
            "NeCo": neco_conf(h, zv(h), lg),
            "NCI": csf.conf_nci(h, lg, w_eff, train_mean_raw, nci_alpha),
            "KPCA RecError global": kpca.get_scores(h),
        }
        return {k: v.cpu().numpy() for k, v in confs.items()}

    conf_test = all_confs(h_test)
    correct = (logits_of(h_test).argmax(dim=1).cpu().numpy() == y_test)
    rows, base = [], {"paradigm": name, "source": source, "run": seed,
                      "dropout": False, "reward": 0.0}
    for cname, c in conf_test.items():
        a, u = rc_metrics(c, (~correct).astype(float))
        rows.append({**base, "csf": cname, "eval_dataset": "iid",
                     "regime": "test", "augrc": a, "aurc": u})
    for ev in eval_names:
        conf_ood = all_confs(to_t(feats[(ev, "eval")][0]))
        n_ood = len(feats[(ev, "eval")][0])
        for cname in conf_test:
            confids = np.concatenate([conf_test[cname][correct],
                                      conf_ood[cname]])
            residuals = np.concatenate([np.zeros(int(correct.sum())),
                                        np.ones(n_ood)])
            a, u = rc_metrics(confids, residuals)
            rows.append({**base, "csf": cname, "eval_dataset": ev,
                         "regime": None, "augrc": a, "aurc": u})
    model_row = {**base, "acc": float(correct.mean()), "temp": temp,
                 "d_ladder": d_lad, "pca_var": pca_var, "pca_dim": pca_dim,
                 "gen_M": int(gen_p["M"]), "gen_gamma": gen_p["gamma"],
                 "ren_M": int(ren_p["M"]), "ren_gamma": ren_p["gamma"],
                 "nng_k": int(nng_p["k_clusters"]),
                 "nng_proportion": nng_p["proportion"],
                 "nci_alpha": nci_alpha,
                 **{f"kpca_{k}": float(v) for k, v in kpca_params.items()}}
    logger.info(f"{enc}/{source}/seed{seed}: done "
                f"({time.perf_counter() - t0:.0f}s)")
    return rows, model_row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--clip-dir", default=str(CODE_DIR / "clip_scores"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-perms", type=int, default=9999)
    ap.add_argument("--bo-init", type=int, default=20)
    ap.add_argument("--bo-iters", type=int, default=80)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    fdir = pathlib.Path(args.features_dir)
    if not fdir.is_dir():
        raise SystemExit(f"features dir not found: {fdir}")

    all_rows, model_rows = [], []
    for enc in ENCODERS:
        feats: dict = {}
        for f in fdir.glob(f"{enc}_*.npz"):
            ds, split = f.stem.replace(f"{enc}_", "").rsplit("_", 1)
            d = np.load(f)
            feats[(ds, split if ds in SOURCES else "eval")] = (
                d["features"], d["labels"])
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
                                       args.device, args)
                for r in rows:
                    if r["regime"] is None:
                        r["regime"] = rmap[r["eval_dataset"]]
                all_rows.extend(rows)
                model_rows.append(mrow)

    long_df = pd.DataFrame(all_rows)
    models_df = pd.DataFrame(model_rows)
    out_dir = OUT_ROOT / "pool_a"

    # ---- gates ----
    ref_models = pd.read_parquet(out_dir / "models_pool_a_newcsfs.parquet")
    ref_long = pd.read_parquet(out_dir / "long_pool_a_newcsfs.parquet")
    key = ["paradigm", "source", "run"]
    gm = models_df.merge(ref_models[key + ["acc"] + NC_PRIMARY], on=key,
                         suffixes=("", "_ref"))
    assert len(gm) == len(models_df) == 40, "model merge incomplete"
    d_acc = (gm["acc"] - gm["acc_ref"]).abs().max()
    if d_acc > 1e-6:
        raise SystemExit(f"GATE G-a FAILED: probe accuracy drift {d_acc}")
    logger.info("G-a PASSED: probe accuracies identical (same probes)")
    models_df = models_df.merge(ref_models[key + NC_PRIMARY +
                                           ["rho_res", "n_residue_spikes",
                                            "median_cos_own",
                                            "class_dep_residue",
                                            "common_mode_energy"]], on=key)

    inv = long_df[long_df["csf"].isin(INVARIANT_CSFS)].merge(
        ref_long, on=key + ["csf", "eval_dataset", "regime"],
        suffixes=("", "_ref"))
    d_inv = (inv["augrc"] - inv["augrc_ref"]).abs()
    logger.info(f"G-b invariant-CSF drift: max {d_inv.max():.4f} over "
                f"{len(inv):,} rows")
    if d_inv.max() > 0.02:
        bad = inv.loc[d_inv.idxmax()]
        raise SystemExit(f"GATE G-b FAILED: {bad['csf']} "
                         f"{bad['eval_dataset']} {d_inv.max():.4f}")
    logger.info("G-b PASSED: Maha/MahaPP/CTM/fDBD reproduce the pilot")

    long_df.to_parquet(out_dir / "long_pool_a_harmonized.parquet",
                       index=False)
    models_df.to_parquet(out_dir / "models_pool_a_harmonized.parquet",
                         index=False)
    flat, _ = compute_track1_cliques(long_df)
    flat.to_parquet(out_dir / "cliques_pool_a_harmonized.parquet",
                    index=False)

    # ---- drift analyses ----
    per_csf = (long_df[long_df["regime"] != "test"]
               .groupby("csf")["augrc"].mean().rename("harmonized")
               .to_frame().join(
                   ref_long[ref_long["regime"] != "test"]
                   .groupby("csf")["augrc"].mean().rename("pilot"))
               .assign(delta=lambda d: d["harmonized"] - d["pilot"])
               .round(2).sort_values("delta"))

    # regime-free selector drift (VGG-trained marginal heads, as submitted)
    vgg_long = pd.read_parquet(
        OUT_ROOT / "track1" / "dataset" / "long_harmonized.parquet")
    vgg_long = add_model_id(vgg_long)
    cliques = pool_cliques_for(("VGG13",), vgg_long)
    for arch, sub_a in vgg_long.groupby("architecture"):
        for c in NC_PRIMARY:
            vgg_long.loc[sub_a.index, c] = (
                (sub_a[c] - sub_a[c].mean()) / (sub_a[c].std() + 1e-12))
    label_wide = (cliques.assign(label=cliques["in_top_clique"].astype(int))
                  .pivot_table(index=["paradigm", "source", "dropout",
                                      "reward", "regime"],
                               columns="csf", values="label",
                               aggfunc="first").reset_index().fillna(0))
    csf_cols = [c for c in label_wide.columns if c not in
                ["paradigm", "source", "dropout", "reward", "regime"]]
    vgg_models = (vgg_long[vgg_long["architecture"] == "VGG13"]
                  [["model_id", "paradigm", "source", "dropout", "reward"]
                   + NC_PRIMARY].drop_duplicates("model_id"))
    tr_marginal = pd.DataFrame(
        [{**m.to_dict(), "regime": r} for _, m in vgg_models.iterrows()
         for r in REGIMES]).merge(
        label_wide, on=["paradigm", "source", "dropout", "reward", "regime"],
        how="inner")

    long_df["model_id"] = (long_df["paradigm"] + "|" + long_df["source"]
                           + "|" + long_df["run"].astype(str))
    rows18 = long_df[long_df["csf"].isin(sorted(ref_long["csf"].unique()))]
    rows18 = rows18[~rows18["csf"].isin(["KPCA RecError global"])]
    rows21 = long_df
    sel_lines = ["| roster | config | near | mid | far | submitted |\n",
                 "|---|---|---|---|---|---|\n"]
    for roster_name, rows_r in [("18 (submitted)", rows18),
                                ("21 (revision)", rows21)]:
        fam = sorted(rows_r["csf"].unique())
        ood = rows_r[rows_r["regime"].isin(REGIMES)][
            ["model_id", "eval_dataset", "source", "regime", "csf", "augrc"]]
        for config, ref in SUBMITTED.items():
            sl = ssl_shortlists(config, models_df, tr_marginal, tr_marginal,
                                csf_cols)
            res = evaluate(ood, sl, always=fam)
            pooled = tuple(res[("all", r)]["predictor"] for r in REGIMES)
            sel_lines.append(
                f"| {roster_name} | {config} | "
                + " | ".join(f"{v:.2f}" for v in pooled)
                + f" | {'/'.join(f'{v:.2f}' for v in ref)} |\n")
            logger.info(f"selector {roster_name} {config}: {pooled}")

    h1 = h1_mantel(models_df, long_df, args.n_perms)

    old_tops = (pd.read_parquet(out_dir / "cliques_pool_a_newcsfs.parquet"))
    def tops(df):
        t = df[df["in_top_clique"] & df["regime"].isin(["near", "mid", "far"])]
        return t.groupby(["paradigm", "source", "regime"])["csf"].apply(
            frozenset)
    a, bnew = tops(old_tops), tops(flat)
    common = sorted(set(a.index) & set(bnew.index))
    n_same = sum(a[k] == bnew[k] for k in common)

    lines = [
        "# Pool A harmonized rerun (paper-faithful protocol, 21 CSFs)\n\n",
        "**Source:** `x8_pool_a/pool_a_harmonized.py`. Gates: probe "
        "accuracies identical; invariant CSFs (Maha, MahaPP, CTM, fDBD) "
        f"reproduce the pilot (max |dAUGRC| {d_inv.max():.4f}). Protocol "
        "changes and BO settings in the module docstring.\n\n",
        "## Regime-free selector, pooled cells (drift vs submitted GLiC-2)\n\n",
        *sel_lines,
        "\n## Per-CSF pooled OOD AUGRC drift (harmonized - pilot)\n\n",
        "```\n" + per_csf.to_string() + "\n```\n\n",
        "## H1 Mantel (harmonized, 21-CSF ranks)\n\n",
        "```\n" + h1.round(4).to_string(index=False) + "\n```\n\n",
        f"## Cliques: {n_same} of {len(common)} (encoder, source, regime) "
        "cells identical to the newcsfs pilot cliques\n",
    ]
    report = out_dir / "41_pool_a_harmonized.md"
    report.write_text("".join(lines))
    logger.info(f"wrote {report}")
    print("".join(lines))


if __name__ == "__main__":
    main()

"""x9 per-model driver: features -> CSF fitting -> per-eval-set metrics.

Mirrors `x8_pool_a/pool_a_analysis.run_model` with the changes real
classifiers require (no probe training; the model's own head supplies
logits and weights):

  - penultimate features = model.forward_head(forward_features, pre_logits=
    True); classifier must be a single nn.Linear (models violating this,
    e.g. distilled DeiT with twin heads, are rejected loudly and recorded);
  - spaces: feature-side CSFs (Maha, MahaPP, NNGuide, PCA/Residual/ViM/NeCo,
    KPCA) operate on RAW penultimate features with the real (W, b), exactly
    as in pool_a where w_eff mapped to raw space; z-space head confs (CTM,
    fDBD, GradNorm, pNML) use z = (h - mu_fit)/sd_fit with the equivalent
    z-space weights W_z = W * sd (identical logits);
  - fitting protocol: class statistics on the FIT draw (100/class from the
    superset, --fit-seed selects the draw; G3 = rerun with another seed);
    everything tunable selects on the SELECTION draw by failure-AUGRC (the
    pipeline's Stage-1 convention), with Bayesian optimization exactly
    where the original pipeline used it (GEN and REN over (M, gamma),
    NNGuide over (k_clusters, bank proportion), PCA RecError over explained
    variance, NCI over alpha in (0, 3e-2) (x9 upgrade of the original's
    7-point grid), KPCA over (variance, gamma, landmarks); 20+80 evals,
    random_state=1) and the original fixed rules elsewhere (Residual/ViM
    dimension ladder 1000/512/d//2; NeCo at 100 components with the
    ViT-raw/standardized and resnet-no-mls conventions);
  - metrics per (csf, eval set): AUGRC/AURC via src.rc_stats (x1000 scale;
    OOD scored against correctly-classified ID test) plus AUROC and
    FPR@95TPR for the G1 OpenOOD comparison.

Outputs per model under --out-dir: <tag>_rows.parquet, <tag>_model.parquet
(accuracy, temperature, NC metrics, X8 descriptors, chosen hyperparams,
skipped CSFs). Per-model failure isolation; --skip-existing resumes.

Run inside the x9 container on a GPU node:
  singularity exec --nv x9_imagenet.sif env HF_HOME=$DATASET_ROOT_DIR/hf \
    python x9_imagenet/extract_and_score.py \
      --data-root $DATASET_ROOT_DIR --models resnet50.tv_in1k
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
from loguru import logger

CODE_DIR = pathlib.Path(__file__).resolve().parents[1]
for p in (CODE_DIR, CODE_DIR / "x8_pool_a",
          CODE_DIR / "x9_imagenet"):
    sys.path.insert(0, str(p))

import pool_a_csfs as csf  # noqa: E402  (torch-only)
from probes_and_descriptors import descriptors  # noqa: E402  (numpy-only)
from src.rc_stats import RiskCoverageStats  # noqa: E402
from data import ParquetImageDataset, make_loader, model_transform  # noqa: E402
from kernel_pca_port import KernelPCAPort  # noqa: E402
from nc_metrics import papyan_nc  # noqa: E402

N_CLS = 1000


def bo_max(f, pbounds: dict, n_init: int, n_iters: int) -> dict:
    """The original pipeline's BO convention (verbose=0, random_state=1)."""
    from bayes_opt import BayesianOptimization
    bo = BayesianOptimization(f=f, pbounds=pbounds, verbose=0,
                              random_state=1)
    bo.maximize(init_points=n_init, n_iter=n_iters)
    return bo.max["params"]
DESC_COLS = ["rho_res", "n_residue_spikes", "median_cos_own",
             "class_dep_residue", "common_mode_energy"]


def rc_metrics(confids: np.ndarray, residuals: np.ndarray) -> tuple[float, float]:
    rc = RiskCoverageStats(confids=confids, residuals=residuals.astype(float))
    return float(rc.augrc), float(rc.aurc)


def fit_temperature_lbfgs(logits_val: torch.Tensor,
                          y_val: torch.Tensor) -> float:
    """Original convention (src/csfs/temperature_scaling.py): LBFGS on
    validation cross-entropy, init T=1, lr=0.01, max_iter=2000 (pool_a's
    41-point log grid was a pilot simplification of the same objective)."""
    t = torch.ones(1, device=logits_val.device).requires_grad_(True)
    opt = torch.optim.LBFGS([t], lr=0.01, max_iter=2000)

    def _eval():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits_val / t, y_val)
        loss.backward()
        return loss

    opt.step(_eval)
    return float(t.detach().item())


def det_metrics(conf_id: np.ndarray, conf_ood: np.ndarray) -> tuple[float, float]:
    """(AUROC, FPR@95TPR) for ID-vs-OOD with confidence scores (higher=ID)."""
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.ones(len(conf_id)), np.zeros(len(conf_ood))])
    s = np.concatenate([conf_id, conf_ood])
    auroc = float(roc_auc_score(y, s))
    thr = np.quantile(conf_id, 0.05)  # 95% TPR on ID
    fpr = float((conf_ood >= thr).mean())
    return auroc, fpr


def draw_uids(superset_dir: pathlib.Path, per_class_fit: int,
              per_class_sel: int, seed: int) -> tuple[set, set]:
    m = pd.read_parquet(superset_dir / "superset_manifest.parquet")
    m["out_row"] = m.groupby("out_shard").cumcount()
    m["uid"] = (m["out_shard"].map(lambda s: f"superset-{s:05d}.parquet")
                + ":" + m["out_row"].astype(str))
    rng = np.random.default_rng(seed)
    fit, sel = [], []
    for _, g in m.groupby("label"):
        idx = rng.permutation(len(g))
        take = g.iloc[idx]
        fit.extend(take["uid"].iloc[:per_class_fit])
        sel.extend(take["uid"].iloc[per_class_fit:per_class_fit + per_class_sel])
    return set(fit), set(sel)


@torch.no_grad()
def collect(model, transform, shards, device, row_filter=None,
            batch_size=256, num_workers=8) -> tuple[torch.Tensor, np.ndarray]:
    loader = make_loader(shards, transform, row_filter=row_filter,
                         batch_size=batch_size, num_workers=num_workers)
    feats, labels = [], []
    for x, y, _ in loader:
        f = model.forward_features(x.to(device, non_blocking=True))
        f = model.forward_head(f, pre_logits=True)
        feats.append(f.float().cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels).numpy()


def process_model(tag: str, family: str, args, paths: dict,
                  fit_uids: set, sel_uids: set, device: str) -> None:
    import timm
    t0 = time.perf_counter()
    model = timm.create_model(tag, pretrained=True).eval().to(device)
    clf = model.get_classifier()
    if not isinstance(clf, torch.nn.Linear):
        raise RuntimeError(f"unsupported head {type(clf).__name__} "
                           "(needs a single nn.Linear)")
    W = clf.weight.detach().float().cpu()
    b = (clf.bias.detach().float().cpu() if clf.bias is not None
         else torch.zeros(N_CLS))
    tfm = model_transform(model)

    h_fit, y_fit = collect(model, tfm, paths["superset"], device, fit_uids)
    h_sel, y_sel = collect(model, tfm, paths["superset"], device, sel_uids)
    h_val, y_val = collect(model, tfm, paths["val"], device)
    ood_feats = {name: collect(model, tfm, shards, device)[0]
                 for name, shards in paths["ood"].items()}
    del model
    torch.cuda.empty_cache()
    logger.info(f"{tag}: features done ({time.perf_counter() - t0:.0f}s; "
                f"fit {len(h_fit):,} sel {len(h_sel):,} val {len(h_val):,} "
                f"ood {sum(len(v) for v in ood_feats.values()):,})")

    dev = torch.device(device)
    W_d, b_d = W.to(dev), b.to(dev)
    logits = lambda h: h @ W_d.T + b_d
    h_fit_d, h_sel_d = h_fit.to(dev), h_sel.to(dev)
    y_fit_t = torch.from_numpy(y_fit).long().to(dev)
    y_sel_t = torch.from_numpy(y_sel).long().to(dev)

    mu, sd = h_fit_d.mean(0), h_fit_d.std(0) + 1e-12
    zf = lambda h: (h - mu) / sd
    W_z = W_d * sd  # z-space weights: z @ W_z.T + (b + mu@W.T) = raw logits
    b_z = b_d + mu @ W_d.T
    mu_std = zf(h_fit_d).mean(dim=0)

    lg_sel = logits(h_sel_d)
    resid_sel = (lg_sel.argmax(1) != y_sel_t).float().cpu().numpy()
    temp = fit_temperature_lbfgs(lg_sel, y_sel_t)

    fitted, skipped = {}, []

    def try_fit(name, fn):
        try:
            fitted[name] = fn()
        except Exception as e:  # noqa: BLE001
            skipped.append(name)
            logger.warning(f"{tag}: {name} fit failed: {e}")

    try_fit("maha", lambda: csf.Mahalanobis(h_fit_d, y_fit_t, N_CLS))
    try_fit("maha_pp", lambda: csf.Mahalanobis(csf.l2n(h_fit_d), y_fit_t, N_CLS))

    # Subspace family, original conventions (src/csfs/):
    #  - Residual/ViM (residual.py, vim.py): centered RAW features, FIXED
    #    dimension ladder d>=2048 -> 1000, d>=768 -> 512, else d//2.
    #  - PCA RecError (projection_filtering.py): STANDARDIZED features,
    #    components by explained variance, BO-tuned over (0.85, 0.99),
    #    relative reconstruction error.
    #  - NeCo (neco.py): standardized for non-ViT / raw for ViT-family,
    #    fixed 100 components, ratio multiplied by max logit except on
    #    resnet-named models.
    d_feat = h_fit_d.shape[1]
    d_lad = 1000 if d_feat >= 2048 else (512 if d_feat >= 768 else d_feat // 2)
    sub = csf.Subspace(h_fit_d)
    sub_z = csf.Subspace(zf(h_fit_d))
    vim_alpha = sub.vim_alpha(h_fit_d, logits(h_fit_d), d_lad)

    s2 = (sub_z.s ** 2)
    cum_ratio = (s2.cumsum(0) / s2.sum()).cpu()
    z_sel = zf(h_sel_d)

    def pca_dim_of(v: float) -> int:
        return int((cum_ratio <= v).sum().item()) + 1

    pca_var = bo_max(
        lambda explained_variance: -rc_metrics(
            sub_z.conf_pca_recerror(z_sel, pca_dim_of(explained_variance))
            .cpu().numpy(), resid_sel)[0],
        {"explained_variance": (0.85, 0.99)}, args.bo_init, args.bo_iters
    )["explained_variance"]
    pca_dim = pca_dim_of(pca_var)

    neco_raw = tag.startswith(("vit_", "deit"))
    neco_mult = "resnet" not in tag
    neco_sub = sub if neco_raw else sub_z
    NECO_DIM = 100

    def conf_neco_orig(hd: torch.Tensor, z: torch.Tensor,
                       lg: torch.Tensor) -> torch.Tensor:
        x = hd if neco_raw else z
        c = x - neco_sub.mu
        ratio = ((c @ neco_sub.vt[:NECO_DIM].T).norm(dim=1)
                 / (x.norm(dim=1) + 1e-12))
        return ratio * lg.max(dim=1).values if neco_mult else ratio

    try_fit("pnml", lambda: csf.PNML(zf(h_fit_d)))
    # BO-tuned CSFs, mirroring the original pipeline (src/csfs/entropy.py,
    # nnguide.py, kernel_pca.py): GEN and REN tune (M, gamma) with
    # M in (1, num_classes), gamma in (1e-6, 0.999999); NNGuide tunes
    # (k_clusters in (10, 500), per-class bank proportion in (0.1, 0.5));
    # objective = selection-set failure-AUGRC throughout.
    p_sel = torch.softmax(lg_sel / temp, dim=1)
    ent_bounds = {"M": (1, N_CLS), "gamma": (1e-6, 0.999999)}
    gen_p = bo_max(
        lambda M, gamma: -rc_metrics(
            csf.conf_gen(p_sel, gamma, int(M)).cpu().numpy(), resid_sel)[0],
        ent_bounds, args.bo_init, args.bo_iters)
    ren_p = bo_max(
        lambda M, gamma: -rc_metrics(
            csf.conf_ren(p_sel, gamma, int(M)).cpu().numpy(), resid_sel)[0],
        ent_bounds, args.bo_init, args.bo_iters)

    bank_scores = csf.conf_energy(logits(h_fit_d), 1.0)

    def build_nng(k_clusters, proportion):
        rng = np.random.default_rng(0)  # fixed: keeps the BO objective
        keep = []                       # deterministic in proportion
        for c in range(N_CLS):
            idx = np.flatnonzero(y_fit == c)
            n = max(1, int(round(proportion * len(idx))))
            keep.append(rng.choice(idx, n, replace=False))
        keep = torch.from_numpy(np.concatenate(keep)).to(dev)
        return csf.NNGuide(h_fit_d[keep], bank_scores[keep], int(k_clusters))

    def nng_obj(k_clusters, proportion):
        m = build_nng(k_clusters, proportion)
        a, _ = rc_metrics(m.conf(h_sel_d, csf.conf_energy(lg_sel, 1.0))
                          .cpu().numpy(), resid_sel)
        return -a

    nng_p = bo_max(nng_obj, {"k_clusters": (10, 500),
                             "proportion": (0.1, 0.5)},
                   args.bo_init, args.bo_iters)
    nng = build_nng(nng_p["k_clusters"], nng_p["proportion"])
    # NCI alpha: BO over the original grid's span (0, 3e-2), including the
    # alpha=0 pure-alignment endpoint (x9 decision 2026-08-06; the original
    # nci.py and the E-F benchmark runs used the fixed 7-point grid).
    train_mean = h_fit_d.mean(dim=0)
    nci_alpha = bo_max(
        lambda alpha: -rc_metrics(
            csf.conf_nci(h_sel_d, lg_sel, W_d, train_mean, alpha)
            .cpu().numpy(), resid_sel)[0],
        {"alpha": (0.0, 3e-2)}, args.bo_init, args.bo_iters)["alpha"]
    kpca, kpca_params = None, {}
    if not args.no_kpca:
        try:
            kpca = KernelPCAPort(W_d, b_d, N_CLS)
            kpca_params = kpca.tune_hyperparameters(
                h_fit_d, h_sel_d, resid_sel, rc_metrics, temperature=1.0,
                n_iters=args.bo_iters, n_init=args.bo_init)
        except Exception as e:  # noqa: BLE001
            kpca, kpca_params = None, {}
            skipped.append("KPCA RecError global")
            logger.warning(f"{tag}: KPCA failed: {e}")
    logger.info(f"{tag}: CSFs fit (ladder D={d_lad}, "
                f"pca var={pca_var:.3f} dim={pca_dim}, "
                f"neco {'raw' if neco_raw else 'std'}"
                f"{' x mls' if neco_mult else ''}; "
                f"BO: gen M={int(gen_p['M'])} g={gen_p['gamma']:.3f}, "
                f"ren M={int(ren_p['M'])} g={ren_p['gamma']:.3f}, "
                f"nng k={int(nng_p['k_clusters'])} "
                f"prop={nng_p['proportion']:.2f}; temp={temp:.2f}, "
                f"skipped={skipped or 'none'})")

    def all_confs(h: torch.Tensor) -> dict[str, np.ndarray]:
        out: dict[str, torch.Tensor] = {}
        for i in range(0, len(h), 8192):
            hd = h[i:i + 8192].to(dev)
            z = zf(hd)
            lg = logits(hd)
            p = torch.softmax(lg / temp, dim=1)
            confs = {
                "MSR": csf.conf_msr(p), "MLS": csf.conf_mls(lg),
                "Energy": csf.conf_energy(lg, temp), "PE": csf.conf_pe(p),
                "GEN": csf.conf_gen(p, gen_p["gamma"], int(gen_p["M"])),
                "REN": csf.conf_ren(p, ren_p["gamma"], int(ren_p["M"])),
                "GE": csf.conf_ge(p), "PCE": csf.conf_pce(p),
                "GradNorm": csf.conf_gradnorm(p, z),
                "CTM": csf.conf_ctm(z, W_z),
                "NNGuide": nng.conf(hd, csf.conf_energy(lg, 1.0)),
                "fDBD": csf.conf_fdbd(z, lg, W_z, mu_std),
                "PCA RecError global": sub_z.conf_pca_recerror(z, pca_dim),
                "Residual": sub.conf_residual(hd, d_lad),
                "ViM": sub.conf_vim(hd, lg, d_lad, vim_alpha, 1.0),
                "NeCo": conf_neco_orig(hd, z, lg),
            }
            if "maha" in fitted:
                confs["Maha"] = fitted["maha"].conf(hd)
            if "maha_pp" in fitted:
                confs["MahaPP"] = fitted["maha_pp"].conf(csf.l2n(hd))
            if "pnml" in fitted:
                confs["pNML"] = fitted["pnml"].conf(z, p)
            confs["NCI"] = csf.conf_nci(hd, lg, W_d, train_mean,
                                        nci_alpha)
            if kpca is not None:
                confs["KPCA RecError global"] = kpca.get_scores(hd)
            for name, v in confs.items():
                out.setdefault(name, []).append(v.float().cpu())
        return {k: torch.cat(v).numpy() for k, v in out.items()}

    conf_val = all_confs(h_val)
    lg_val = logits(h_val.to(dev)).cpu()
    correct = (lg_val.argmax(1).numpy() == y_val)
    acc = float(correct.mean())
    rows = []
    for name, c in conf_val.items():
        a, u = rc_metrics(c, (~correct).astype(float))
        rows.append({"tag": tag, "family": family, "csf": name,
                     "eval_dataset": "iid_test", "augrc": a, "aurc": u,
                     "auroc": np.nan, "fpr95": np.nan,
                     "n": len(h_val)})
    for ev, h_ood in ood_feats.items():
        conf_ood = all_confs(h_ood)
        for name in conf_val:
            ci, co = conf_val[name][correct], conf_ood[name]
            a, u = rc_metrics(np.concatenate([ci, co]),
                              np.concatenate([np.zeros(len(ci)),
                                              np.ones(len(co))]))
            auroc, fpr = det_metrics(conf_val[name], conf_ood[name])
            rows.append({"tag": tag, "family": family, "csf": name,
                         "eval_dataset": ev, "augrc": a, "aurc": u,
                         "auroc": auroc, "fpr95": fpr, "n": len(co)})
        logger.info(f"{tag}: scored {ev} ({len(h_ood):,})")

    h_fit_np, w_np = h_fit.numpy(), W.numpy()
    desc = descriptors(h_fit_np, y_fit, N_CLS)
    desc["common_mode_energy"] = desc.pop("common_mode")["energy_fraction"]
    model_row = {"tag": tag, "family": family, "acc": acc,
                 "temp": float(temp), "feature_dim": h_fit.shape[1],
                 "fit_seed": args.fit_seed, "d_ladder": d_lad,
                 "pca_var": float(pca_var), "pca_dim": pca_dim,
                 "neco_raw": neco_raw, "neco_mult": neco_mult,
                 "gen_M": int(gen_p["M"]), "gen_gamma": float(gen_p["gamma"]),
                 "ren_M": int(ren_p["M"]), "ren_gamma": float(ren_p["gamma"]),
                 "nng_k": int(nng_p["k_clusters"]),
                 "nng_proportion": float(nng_p["proportion"]),
                 "nci_alpha": float(nci_alpha),
                 "skipped_csfs": ",".join(skipped),
                 **{f"kpca_{k}": float(v) for k, v in kpca_params.items()},
                 **papyan_nc(h_fit_np, y_fit, w_np, N_CLS),
                 **{k: v for k, v in desc.items() if k in DESC_COLS}}

    out = pathlib.Path(args.out_dir)
    pd.DataFrame(rows).to_parquet(out / f"{tag}_rows.parquet", index=False)
    pd.DataFrame([model_row]).to_parquet(out / f"{tag}_model.parquet",
                                         index=False)
    logger.info(f"{tag}: DONE acc={acc:.3f} "
                f"({time.perf_counter() - t0:.0f}s total)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--models", nargs="*", default=["all"])
    ap.add_argument("--fit-seed", type=int, default=0)
    ap.add_argument("--fit-per-class", type=int, default=100)
    ap.add_argument("--sel-per-class", type=int, default=25)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-kpca", action="store_true")
    ap.add_argument("--bo-init", type=int, default=20,
                    help="BO init points for GEN/REN/NNGuide/KPCA (paper: 20)")
    ap.add_argument("--bo-iters", type=int, default=80,
                    help="BO iterations for GEN/REN/NNGuide/KPCA (paper: 80)")
    args = ap.parse_args()

    root = pathlib.Path(args.data_root)
    args.out_dir = args.out_dir or str(root / "x9_outputs")
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    superset = sorted((root / "imagenet1k_superset").glob("superset-*.parquet"))
    val = sorted((root / "imagenet1k_raw" / "data").glob("validation-*.parquet"))
    ood_dir = root / "ood_parquet"
    ood = {}
    for f in sorted(ood_dir.glob("*-*.parquet")):
        ood.setdefault(f.name.rsplit("-", 1)[0], []).append(f)
    if not (superset and val and ood):
        raise SystemExit(f"missing data under {root} "
                         f"(superset {len(superset)}, val {len(val)}, "
                         f"ood sets {len(ood)})")
    paths = {"superset": superset, "val": val, "ood": ood}
    logger.info(f"data: {len(superset)} superset shards, {len(val)} val "
                f"shards, OOD sets {sorted(ood)}")

    manifest = pd.read_csv(
        pathlib.Path(__file__).resolve().parent / "manifest_verified.csv")
    manifest = manifest[manifest["status"] == "ok"]
    if args.models != ["all"]:
        manifest = manifest[manifest["tag"].isin(args.models)]
    fit_uids, sel_uids = draw_uids(root / "imagenet1k_superset",
                                   args.fit_per_class, args.sel_per_class,
                                   args.fit_seed)
    logger.info(f"{len(manifest)} models; fit {len(fit_uids):,} uids, "
                f"sel {len(sel_uids):,} uids (seed {args.fit_seed})")

    failed = []
    for _, r in manifest.iterrows():
        out_f = pathlib.Path(args.out_dir) / f"{r['tag']}_rows.parquet"
        if args.skip_existing and out_f.exists():
            logger.info(f"[skip] {r['tag']}")
            continue
        try:
            process_model(r["tag"], r["family"], args, paths,
                          fit_uids, sel_uids, args.device)
        except Exception as e:  # noqa: BLE001
            failed.append(r["tag"])
            logger.error(f"{r['tag']}: FAILED: {e}\n{traceback.format_exc()}")
    if failed:
        logger.error(f"{len(failed)} models failed: {failed}")
    else:
        logger.info("all models completed")


if __name__ == "__main__":
    main()

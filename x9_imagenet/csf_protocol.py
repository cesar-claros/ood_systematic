"""The paper-faithful CSF fitting/tuning protocol, defined ONCE.

Shared by the x9 ImageNet driver (extract_and_score.py) and the harmonized
SSL rerun (x8_pool_a/pool_a_harmonized.py). Everything here was audited
line-by-line against src/csfs/ (2026-08-06):

  BO-tuned (20+80 evals, random_state=1, objective = failure-AUGRC on the
  tuning slice): GEN and REN over (M in (1, n_cls), gamma in (1e-6,
  0.999999)); NNGuide over (k_clusters in (10, 500), per-class bank
  proportion in (0.1, 0.5)); PCA RecError over explained variance in
  (0.85, 0.99) on STANDARDIZED features; NCI over alpha in (0, 3e-2)
  (x9 upgrade of the original 7-point grid); KPCA via kernel_pca_port.
  Fixed rules: temperature by LBFGS on tuning-slice cross-entropy;
  Residual/ViM dimension ladder (d>=2048 -> 1000, d>=768 -> 512, else
  d//2) on centered raw features; NeCo at 100 components, raw input for
  ViT-family models and standardized otherwise, ratio multiplied by the
  max logit except on resnet-named models.

Data conventions stay pool-specific and are the CALLER's job: x9 uses
disjoint fit (class statistics) and selection (tuning) draws; the SSL pool
follows the paper's Stage-1 convention where one validation carve-out
serves both roles.
"""
from __future__ import annotations

import numpy as np
import torch

NECO_DIM = 100


def bo_max(f, pbounds: dict, n_init: int = 20, n_iters: int = 80) -> dict:
    """The original pipeline's BO convention (verbose=0, random_state=1)."""
    from bayes_opt import BayesianOptimization
    bo = BayesianOptimization(f=f, pbounds=pbounds, verbose=0,
                              random_state=1)
    bo.maximize(init_points=n_init, n_iter=n_iters)
    return bo.max["params"]


def fit_temperature_lbfgs(logits_val: torch.Tensor,
                          y_val: torch.Tensor) -> float:
    """Original convention (src/csfs/temperature_scaling.py): LBFGS on
    validation cross-entropy, init T=1, lr=0.01, max_iter=2000."""
    t = torch.ones(1, device=logits_val.device).requires_grad_(True)
    opt = torch.optim.LBFGS([t], lr=0.01, max_iter=2000)

    def _eval():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits_val / t, y_val)
        loss.backward()
        return loss

    opt.step(_eval)
    return float(t.detach().item())


def ladder_dim(d_feat: int) -> int:
    """Residual/ViM fixed dimension ladder (src/csfs/{residual,vim}.py)."""
    return 1000 if d_feat >= 2048 else (512 if d_feat >= 768 else d_feat // 2)


def tune_entropy_pair(csf_mod, p_tune: torch.Tensor, resid_tune: np.ndarray,
                      n_cls: int, rc_metrics, bo_init: int = 20,
                      bo_iters: int = 80) -> tuple[dict, dict]:
    """GEN and REN (M, gamma) BO, bounds from src/csfs/entropy.py."""
    bounds = {"M": (1, n_cls), "gamma": (1e-6, 0.999999)}
    gen_p = bo_max(
        lambda M, gamma: -rc_metrics(
            csf_mod.conf_gen(p_tune, gamma, int(M)).cpu().numpy(),
            resid_tune)[0], bounds, bo_init, bo_iters)
    ren_p = bo_max(
        lambda M, gamma: -rc_metrics(
            csf_mod.conf_ren(p_tune, gamma, int(M)).cpu().numpy(),
            resid_tune)[0], bounds, bo_init, bo_iters)
    return gen_p, ren_p


def tune_nnguide(csf_mod, h_stats: torch.Tensor, y_stats: np.ndarray,
                 bank_scores: torch.Tensor, n_cls: int,
                 h_tune: torch.Tensor, lg_tune: torch.Tensor,
                 resid_tune: np.ndarray, rc_metrics,
                 bo_init: int = 20, bo_iters: int = 80):
    """NNGuide (k_clusters, per-class bank proportion) BO, bounds from
    src/csfs/nnguide.py; fixed inner rng keeps the objective deterministic
    in the proportion. Returns (fitted NNGuide, best params)."""
    dev = h_stats.device

    def build(k_clusters, proportion):
        rng = np.random.default_rng(0)
        keep = []
        for c in range(n_cls):
            idx = np.flatnonzero(y_stats == c)
            if len(idx) == 0:
                continue
            n = max(1, int(round(proportion * len(idx))))
            keep.append(rng.choice(idx, n, replace=False))
        keep = torch.from_numpy(np.concatenate(keep)).to(dev)
        return csf_mod.NNGuide(h_stats[keep], bank_scores[keep],
                               int(k_clusters))

    def obj(k_clusters, proportion):
        m = build(k_clusters, proportion)
        a, _ = rc_metrics(
            m.conf(h_tune, csf_mod.conf_energy(lg_tune, 1.0)).cpu().numpy(),
            resid_tune)
        return -a

    params = bo_max(obj, {"k_clusters": (10, 500), "proportion": (0.1, 0.5)},
                    bo_init, bo_iters)
    return build(params["k_clusters"], params["proportion"]), params


def tune_pca_re(sub_z, z_tune: torch.Tensor, resid_tune: np.ndarray,
                rc_metrics, bo_init: int = 20,
                bo_iters: int = 80) -> tuple[float, int]:
    """PCA RecError explained-variance BO on the standardized-feature
    subspace (src/csfs/projection_filtering.py); returns (variance, dim)."""
    s2 = sub_z.s ** 2
    cum_ratio = (s2.cumsum(0) / s2.sum()).cpu()

    def dim_of(v: float) -> int:
        return int((cum_ratio <= v).sum().item()) + 1

    var = bo_max(
        lambda explained_variance: -rc_metrics(
            sub_z.conf_pca_recerror(z_tune, dim_of(explained_variance))
            .cpu().numpy(), resid_tune)[0],
        {"explained_variance": (0.85, 0.99)}, bo_init, bo_iters
    )["explained_variance"]
    return float(var), dim_of(var)


def tune_nci_alpha(csf_mod, h_tune: torch.Tensor, lg_tune: torch.Tensor,
                   resid_tune: np.ndarray, W: torch.Tensor,
                   train_mean: torch.Tensor, rc_metrics,
                   bo_init: int = 20, bo_iters: int = 80) -> float:
    """NCI alpha BO over the original grid's span (0, 3e-2), alpha=0
    (pure alignment) reachable."""
    return float(bo_max(
        lambda alpha: -rc_metrics(
            csf_mod.conf_nci(h_tune, lg_tune, W, train_mean, alpha)
            .cpu().numpy(), resid_tune)[0],
        {"alpha": (0.0, 3e-2)}, bo_init, bo_iters)["alpha"])


def neco_flags(model_name: str) -> tuple[bool, bool]:
    """(raw_input, multiply_by_mls) per src/csfs/neco.py conventions:
    ViT-family models use raw features; every non-resnet model multiplies
    the ratio by the max logit. The startswith rule matches the x9 sweep's
    behavior exactly (maxvit is NOT ViT-family there); dinov2/clip covers
    the SSL probes, whose encoders are ViTs."""
    lower = model_name.lower()
    raw = (lower.startswith(("vit_", "deit"))
           or "dinov2" in lower or "clip" in lower)
    mult = "resnet" not in lower
    return raw, mult


def make_neco_conf(sub_raw, sub_z, raw: bool, mult: bool):
    """NeCo score closure: 100-component norm ratio in the chosen space
    (projection centered by the subspace mean, input norm uncentered, as in
    the original), optionally times the max logit."""
    space = sub_raw if raw else sub_z

    def conf(hd: torch.Tensor, z: torch.Tensor,
             lg: torch.Tensor) -> torch.Tensor:
        x = hd if raw else z
        c = x - space.mu
        ratio = ((c @ space.vt[:NECO_DIM].T).norm(dim=1)
                 / (x.norm(dim=1) + 1e-12))
        return ratio * lg.max(dim=1).values if mult else ratio

    return conf

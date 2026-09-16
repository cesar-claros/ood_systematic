"""Adaptation-trajectory development driver (2026-09-15; preparation for the pilot, not a frozen registration).

One trajectory: backbone + downstream head, adapted by full fine-tuning or LoRA from a downstream-compatible reference head
(a linear probe on frozen reference features). At every listed checkpoint (0 = reference) it extracts features on FIXED image lists
with stable ids, measures geometry (corrected Papyan panel via pilot0.geometry, projector residue energy and paired linear CKA via
descriptors_v3, total drift, principal angles), refits the probe-pool detector inventory under one rule, scores every detector on
all-ID-versus-OOD (primary) and the historical correct-ID-only population (secondary), evaluates the reference-feature detectors and a
fixed permitted-ID-normalized reference/adapted combination as baselines, and writes a timing and storage ledger.

Backbones: dinov2_vitb14 (torch.hub, HPC) or toy (tiny ViT, random init, CPU smoke test). Data: cifar100 (torchvision, HPC) or
synthetic (CPU smoke test). Run the smoke test:
  .venv/bin/python x8_pool_a/adaptation_trajectory.py --backbone toy --data synthetic --method lora --steps 30 --checkpoints 0,15,30 --out /tmp/traj_smoke
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR)); sys.path.insert(0, str(CODE_DIR / "x8_pool_a")); sys.path.insert(0, str(CODE_DIR / "pilot0"))
import pool_a_csfs as csf  # noqa: E402
import descriptors_v3 as d3  # noqa: E402
from geometry import fit_feature_model, papyan_metrics  # noqa: E402
from src.rc_stats import RiskCoverageStats  # noqa: E402

PCA_DIMS = [32, 64, 128, 256]; NNG_KS = [10, 50]


# ----------------------------------------------------------------------------- models
class ToyAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__(); self.heads = heads; self.qkv = nn.Linear(dim, 3 * dim); self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, D = x.shape; qkv = self.qkv(x).reshape(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]; a = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        return self.proj((a.softmax(-1) @ v).transpose(1, 2).reshape(B, N, D))


class ToyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.norm1 = nn.LayerNorm(dim); self.attn = ToyAttention(dim); self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
    def forward(self, x):
        x = x + self.attn(self.norm1(x)); return x + self.mlp(self.norm2(x))


class ToyViT(nn.Module):
    """Tiny ViT with dinov2-style module names (blocks[i].attn.qkv / attn.proj) so the LoRA targeting is identical."""
    def __init__(self, img=32, patch=8, dim=64, depth=2):
        super().__init__(); self.patch_embed = nn.Conv2d(3, dim, patch, patch); n = (img // patch) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)); self.pos_embed = nn.Parameter(torch.randn(1, n + 1, dim) * 0.02)
        self.blocks = nn.ModuleList([ToyBlock(dim) for _ in range(depth)]); self.norm = nn.LayerNorm(dim); self.embed_dim = dim
    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2); x = torch.cat([self.cls_token.expand(len(x), -1, -1), x], 1) + self.pos_embed
        for b in self.blocks: x = b(x)
        return self.norm(x)[:, 0]


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__(); self.base = base; self.scale = alpha / rank
        self.A = nn.Parameter(torch.randn(rank, base.in_features) * 0.01); self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        for p in self.base.parameters(): p.requires_grad_(False)
    def forward(self, x): return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


def apply_lora(model: nn.Module, rank: int, alpha: float, pattern: str) -> int:
    n = 0
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and re.search(pattern, full):
                setattr(mod, child_name, LoRALinear(child, rank, alpha)); n += 1
    return n


def build_backbone(name: str):
    if name == "toy": return ToyViT()
    if name == "dinov2_vitb14":
        m = torch.hub.load("facebookresearch/dinov2:81b2b64", "dinov2_vitb14", skip_validation=True); return m
    raise ValueError(name)


class Classifier(nn.Module):
    """Backbone features -> fixed standardization from the reference -> linear head (downstream-compatible at the reference)."""
    def __init__(self, backbone, mu, sd, w, b):
        super().__init__(); self.backbone = backbone
        self.register_buffer("mu", mu); self.register_buffer("sd", sd); self.head = nn.Linear(len(mu), w.shape[0])
        with torch.no_grad(): self.head.weight.copy_(w); self.head.bias.copy_(b)
    def features(self, x): return self.backbone(x)
    def forward(self, x): return self.head((self.features(x) - self.mu) / self.sd)


# ----------------------------------------------------------------------------- data
def synthetic_data(n_cls, n_fit, n_val, n_test, n_ood, seed, img=32):
    g = torch.Generator().manual_seed(seed); protos = torch.randn(n_cls + n_cls, 3, img, img, generator=g) * 0.6
    def draw(classes, n):
        y = torch.randint(0, len(classes), (n,), generator=g); x = protos[classes][y] + torch.randn(n, 3, img, img, generator=g); return x, y
    x_fit, y_fit = draw(list(range(n_cls)), n_fit); x_val, y_val = draw(list(range(n_cls)), n_val); x_te, y_te = draw(list(range(n_cls)), n_test)
    x_shift, _ = draw(list(range(n_cls, 2 * n_cls)), n_ood); x_noise = torch.randn(n_ood, 3, img, img, generator=g) * 2
    return {"fit": (x_fit, y_fit), "val": (x_val, y_val), "test": (x_te, y_te)}, {"shift": x_shift, "noise": x_noise}


def cifar100_data(n_cls, n_fit, n_val, n_test, n_ood, seed, root):
    """HPC path. Images are kept as uint8 tensors at native size (CIFAR/SVHN 32 px, DTD resized to 224 px) and resized plus
    normalized on the device per batch, so host memory stays at a few hundred megabytes instead of tens of gigabytes."""
    import torchvision, torchvision.transforms as T  # noqa: E401
    to_u8 = T.Compose([T.PILToTensor()]); to_u8_224 = T.Compose([T.Resize(224), T.CenterCrop(224), T.PILToTensor()])
    tr = torchvision.datasets.CIFAR100(root, train=True, download=False, transform=to_u8); te = torchvision.datasets.CIFAR100(root, train=False, download=False, transform=to_u8)
    g = np.random.default_rng(seed); idx = g.permutation(len(tr)); fit_idx, val_idx = idx[:n_fit], idx[n_fit:n_fit + n_val]
    stack = lambda ds, ii: (torch.stack([ds[i][0] for i in ii]), torch.tensor([ds[i][1] for i in ii]))
    ood = {}
    for name, ctor, tfm in [("svhn", lambda: torchvision.datasets.SVHN(root, split="test", download=False, transform=to_u8), None),
                            ("dtd", lambda: torchvision.datasets.DTD(root, split="test", download=False, transform=to_u8_224), None)]:
        try:
            ds = ctor(); jj = g.permutation(len(ds))[:n_ood]; ood[name] = torch.stack([ds[i][0] for i in jj])
        except Exception as e: print(f"OOD set {name} unavailable: {e}")
    return {"fit": stack(tr, fit_idx), "val": stack(tr, val_idx), "test": stack(te, g.permutation(len(te))[:n_test])}, ood


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def preprocess(x: torch.Tensor, backbone: str) -> torch.Tensor:
    """uint8 or float images -> float, resized to the backbone's input size, ImageNet-normalized for dinov2."""
    x = x.float() / 255.0 if x.dtype == torch.uint8 else x
    if backbone == "toy": return x if x.shape[-1] == 32 else F.interpolate(x, size=32, mode="bilinear", align_corners=False)
    if x.shape[-1] != 224: x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


# ----------------------------------------------------------------------------- metrics
def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUROC with ties at half credit; pos = ID scores (higher = more ID), neg = OOD scores."""
    s = np.concatenate([pos, neg]); r = s.argsort().argsort().astype(float)
    # average ranks for ties
    order = np.argsort(s); ss = s[order]; ranks = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]: j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1; i = j + 1
    rp = ranks[:len(pos)]; return float((rp.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def augrc(conf: np.ndarray, resid: np.ndarray) -> float:
    return float(RiskCoverageStats(confids=conf, residuals=resid.astype(float)).augrc)


def fit_detectors(h_fit, y_fit, h_val, y_val, w_raw, b, mu, sd, n_cls, device):
    """Probe-pool recipe with the CURRENT head; detectors fitted on the validation slice (Stage-1 convention)."""
    zf = lambda h: (h - mu) / sd; w = w_raw * sd  # head acts on standardized features
    logits_of = lambda h: zf(h) @ w.T + b
    temp = csf.fit_temperature(logits_of(h_val), y_val); lg_val = logits_of(h_val)
    resid_val = (lg_val.argmax(1) != y_val).float().cpu().numpy()
    maha = csf.Mahalanobis(h_val, y_val, n_cls); maha_pp = csf.Mahalanobis(csf.l2n(h_val), y_val, n_cls)
    w_eff = w / sd; train_mean_raw = h_fit.mean(0)
    nci_alpha = csf.fit_nci_alpha(h_val, lg_val, resid_val, w_eff, train_mean_raw, lambda c, r: (augrc(c, r), 0.0))
    sub = csf.Subspace(h_val); pnml = csf.PNML(zf(h_val)); bank = csf.conf_energy(lg_val, 1.0); mu_std = zf(h_fit).mean(0)
    dims = [d for d in PCA_DIMS if d < h_val.shape[1]] or [max(2, h_val.shape[1] // 2)]
    def sel(fn):
        best, ba = dims[0], np.inf
        for d in dims:
            a = augrc(fn(h_val, d).cpu().numpy(), resid_val)
            if a < ba: best, ba = d, a
        return best
    d_pca, d_res, d_neco = sel(sub.conf_pca_recerror), sel(sub.conf_residual), sel(sub.conf_neco); vim_alpha = sub.vim_alpha(h_val, lg_val, d_res)
    ks = [k for k in NNG_KS if k < len(h_val)] or [max(1, len(h_val) // 10)]; k_best, ba = ks[0], np.inf
    for k in ks:
        a = augrc(csf.NNGuide(h_val, bank, k).conf(h_val, csf.conf_energy(lg_val, 1.0)).cpu().numpy(), resid_val)
        if a < ba: k_best, ba = k, a
    nng = csf.NNGuide(h_val, bank, k_best)
    def all_confs(h):
        z = zf(h); lg = z @ w.T + b; p = torch.softmax(lg / temp, 1)
        c = {"MSR": csf.conf_msr(p), "MLS": csf.conf_mls(lg), "Energy": csf.conf_energy(lg, temp), "PE": csf.conf_pe(p),
             "GEN": csf.conf_gen(p, 0.1, min(100, n_cls)), "REN": csf.conf_ren(p, 0.5, n_cls), "GE": csf.conf_ge(p), "PCE": csf.conf_pce(p),
             "GradNorm": csf.conf_gradnorm(p, z), "pNML": pnml.conf(z, p), "CTM": csf.conf_ctm(z, w), "Maha": maha.conf(h),
             "MahaPP": maha_pp.conf(csf.l2n(h)), "NNGuide": nng.conf(h, csf.conf_energy(lg, 1.0)), "fDBD": csf.conf_fdbd(z, lg, w, mu_std),
             "PCA RecError global": sub.conf_pca_recerror(h, d_pca), "Residual": sub.conf_residual(h, d_res),
             "ViM": sub.conf_vim(h, lg, d_res, vim_alpha, 1.0), "NeCo": sub.conf_neco(h, d_neco),
             "NCI": csf.conf_nci(h, lg, w_eff, train_mean_raw, nci_alpha)}
        return {k: v.detach().cpu().numpy() for k, v in c.items()}, lg.argmax(1)
    return all_confs, {"temp": float(temp), "d_pca": d_pca, "d_res": d_res, "d_neco": d_neco, "k_nng": k_best, "nci_alpha": float(nci_alpha)}


@torch.no_grad()
def extract(model, x, batch, device, backbone="toy", amp=False):
    out = []
    for i in range(0, len(x), batch):
        xb = preprocess(x[i:i + batch].to(device), backbone)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=bool(amp and device.startswith("cuda"))):
            out.append(model.features(xb).float().cpu())
    return torch.cat(out)


AMP_DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16


def principal_angle_cos(h_ref, h_cur, k):
    def basis(h):
        c = h - h.mean(0); _, _, vt = np.linalg.svd(c, full_matrices=False); return vt[:k].T
    s = np.linalg.svd(basis(h_ref).T @ basis(h_cur), compute_uv=False); return float(np.mean(np.clip(s, 0, 1)))


def rank_normalize(scores_fit, scores):
    """Map scores to [0,1] by their rank among the permitted ID fit-set scores (GNOME-style fixed combination input)."""
    s = np.sort(scores_fit); return np.searchsorted(s, scores, side="right") / len(s)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="toy"); ap.add_argument("--data", default="synthetic"); ap.add_argument("--data-root", default=str(pathlib.Path.home() / "data"))
    ap.add_argument("--method", choices=["full", "lora"], default="lora"); ap.add_argument("--lora-rank", type=int, default=8); ap.add_argument("--lora-alpha", type=float, default=16)
    ap.add_argument("--lora-pattern", default=r"attn\.(qkv|proj)$"); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--checkpoints", default="0,15,30"); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-cls", type=int, default=5); ap.add_argument("--n-fit", type=int, default=400); ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=200); ap.add_argument("--n-ood", type=int, default=200); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cpu"); ap.add_argument("--amp", action="store_true", help="mixed precision on CUDA: bf16 where supported (A100), else fp16 with loss scaling (V100)")
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); dev = a.device
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); ledger = {"config": vars(a), "stages": []}
    def tick(stage, t0, **extra): ledger["stages"].append({"stage": stage, "seconds": round(time.perf_counter() - t0, 3), **extra})

    t0 = time.perf_counter()
    if a.data == "synthetic": splits, ood = synthetic_data(a.n_cls, a.n_fit, a.n_val, a.n_test, a.n_ood, a.seed)
    else: splits, ood = cifar100_data(a.n_cls, a.n_fit, a.n_val, a.n_test, a.n_ood, a.seed, a.data_root)
    n_cls = int(splits["fit"][1].max()) + 1; ids = {k: np.arange(len(v[0])) for k, v in splits.items()}; tick("data", t0, n_cls=n_cls)

    t0 = time.perf_counter(); backbone = build_backbone(a.backbone).to(dev)
    class Wrap(nn.Module):
        def __init__(s, b): super().__init__(); s.backbone = b
        def features(s, x): return s.backbone(x)
    h_fit0 = extract(Wrap(backbone), splits["fit"][0], a.batch, dev, a.backbone, a.amp); tick("reference_extract_fit", t0, n=len(h_fit0), dim=h_fit0.shape[1])
    t0 = time.perf_counter(); probe = csf.train_probe(h_fit0.to(dev), splits["fit"][1].to(dev), n_cls, seed=a.seed)
    model = Classifier(backbone, probe["mu"].detach(), probe["sd"].detach(), probe["W"].detach(), probe["b"].detach()).to(dev); tick("reference_probe_head", t0, probe_acc=float(probe["acc"]))
    if a.method == "lora":
        n_lora = apply_lora(model.backbone, a.lora_rank, a.lora_alpha, a.lora_pattern); model.to(dev)
        params = [p for n, p in model.named_parameters() if p.requires_grad and (".A" in n or ".B" in n or n.startswith("head"))]
        ledger["lora_modules"] = n_lora
    else: params = list(model.parameters())
    ledger["trainable_params"] = int(sum(p.numel() for p in params)); ledger["total_params"] = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01); scaler = torch.amp.GradScaler("cuda", enabled=bool(a.amp and dev.startswith("cuda") and AMP_DTYPE == torch.float16))
    ledger["amp_dtype"] = str(AMP_DTYPE) if a.amp else "fp32"
    cps = sorted({int(c) for c in a.checkpoints.split(",")} | {0}); x_fit, y_fit = splits["fit"]; g = torch.Generator().manual_seed(a.seed)
    ref = {}; rows = []; meas = []
    def evaluate(step):
        t0 = time.perf_counter(); model.eval(); feats = {k: extract(model, v[0], a.batch, dev, a.backbone, a.amp) for k, v in splits.items()}; feats_ood = {k: extract(model, v, a.batch, dev, a.backbone, a.amp) for k, v in ood.items()}
        tick(f"extract@{step}", t0, n=sum(len(v) for v in feats.values()) + sum(len(v) for v in feats_ood.values()))
        np.savez_compressed(out / f"features_step{step}.npz", **{f"{k}_h": v.numpy() for k, v in feats.items()}, **{f"{k}_y": splits[k][1].numpy() for k in splits}, **{f"ood_{k}": v.numpy() for k, v in feats_ood.items()}, **{f"{k}_ids": ids[k] for k in ids})
        w_raw = (model.head.weight / model.sd).detach().cpu(); b = model.head.bias.detach().cpu(); mu, sd = model.mu.cpu(), model.sd.cpu()
        t0 = time.perf_counter(); hf = feats["fit"].numpy().astype(np.float64); yf = splits["fit"][1].numpy()
        fm = fit_feature_model(hf, yf, n_cls); pap = papyan_metrics(w_raw.numpy(), fm)
        m = {"step": step, **{f"nc_{k}": v for k, v in pap.items()}, "residue_energy_projector": d3.residue_energy_projector(hf, yf, n_cls),
             "id_test_acc": float((model.head((feats["test"] - mu) / sd).argmax(1) == splits["test"][1]).float().mean())}
        if step == 0: ref["fit"] = hf
        else:
            m["paired_cka_fit"] = float(d3.paired_linear_cka(ref["fit"], hf, ids["fit"], ids["fit"]) if "paired_linear_cka" in dir(d3) else np.nan)
            m["total_drift_fit"] = float(np.linalg.norm(hf - ref["fit"]) / np.linalg.norm(ref["fit"])); m["principal_angle_cos_fit"] = principal_angle_cos(ref["fit"], hf, max(1, n_cls - 1))
        meas.append(m); tick(f"measure@{step}", t0)
        t0 = time.perf_counter(); all_confs, hp = fit_detectors(feats["fit"].to(dev), splits["fit"][1].to(dev), feats["val"].to(dev), splits["val"][1].to(dev), w_raw.to(dev), b.to(dev), mu.to(dev), sd.to(dev), n_cls, dev)
        tick(f"fit_detectors@{step}", t0, **hp)
        t0 = time.perf_counter(); c_fit, _ = all_confs(feats["fit"].to(dev)); c_te, pred_te = all_confs(feats["test"].to(dev)); correct = (pred_te.cpu() == splits["test"][1]).numpy()
        if step == 0: ref["c_fit"], ref["c_te"], ref["c_ood"] = c_fit, c_te, {k: all_confs(v.to(dev))[0] for k, v in feats_ood.items()}
        for name, ho in feats_ood.items():
            c_ood, _ = all_confs(ho.to(dev))
            for det in c_te:
                variants = {"adapted": (c_te[det], c_ood[det])}
                if step > 0:
                    variants["reference"] = (ref["c_te"][det], ref["c_ood"][name][det])
                    comb = lambda s_ref, s_cur, d=det: 0.5 * rank_normalize(ref["c_fit"][d], s_ref) + 0.5 * rank_normalize(c_fit[d], s_cur)
                    variants["combined"] = (comb(ref["c_te"][det], c_te[det]), comb(ref["c_ood"][name][det], c_ood[det]))
                for var, (s_id, s_ood) in variants.items():
                    resid_all = np.concatenate([np.zeros(len(s_id)), np.ones(len(s_ood))]); conf_all = np.concatenate([s_id, s_ood])
                    rows.append({"step": step, "ood_set": name, "detector": det, "variant": var, "auroc_allid": auroc(s_id, s_ood),
                                 "augrc_allid": augrc(conf_all, resid_all), "augrc_correct_only": augrc(np.concatenate([s_id[correct], s_ood]), np.concatenate([np.zeros(correct.sum()), np.ones(len(s_ood))])),
                                 "id_failure_augrc": augrc(s_id, (~correct).astype(float))})
        tick(f"score@{step}", t0, n_rows=len(rows)); model.train()
    evaluate(0); model.train(); step = 0
    while step < max(cps):
        idx = torch.randint(0, len(x_fit), (a.batch,), generator=g); t0 = time.perf_counter()
        use_amp = bool(a.amp and dev.startswith("cuda"))
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=use_amp):
            loss = F.cross_entropy(model(preprocess(x_fit[idx].to(dev), a.backbone)), y_fit[idx].to(dev))
        opt.zero_grad()
        if use_amp and AMP_DTYPE == torch.float16: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else: loss.backward(); opt.step()
        step += 1
        ledger.setdefault("train_seconds", 0.0); ledger["train_seconds"] += time.perf_counter() - t0
        if step in cps:
            torch.save({k: v.cpu() for k, v in model.state_dict().items() if a.method == "full" or ".A" in k or ".B" in k or k.startswith("head") or k in ("mu", "sd")}, out / f"ckpt_step{step}.pt")
            ledger["last_loss"] = float(loss.detach()); evaluate(step)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "outcomes.csv", index=False); pd.DataFrame(meas).to_csv(out / "measurements.csv", index=False)
    ledger["artifact_bytes"] = int(sum(p.stat().st_size for p in out.iterdir())); ledger["total_seconds"] = round(sum(s["seconds"] for s in ledger["stages"]) + ledger.get("train_seconds", 0.0), 3)
    json.dump(ledger, open(out / "ledger.json", "w"), indent=1); print(json.dumps({k: ledger[k] for k in ("trainable_params", "total_params", "train_seconds", "total_seconds", "artifact_bytes")}, indent=1))
    print(pd.DataFrame(meas).round(4).to_string(index=False))
    df = pd.DataFrame(rows); print(df[df.variant == "adapted"].groupby(["step", "ood_set"]).auroc_allid.agg(["mean", "max"]).round(3).to_string())


if __name__ == "__main__":
    main()

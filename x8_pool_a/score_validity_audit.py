"""Raw-score validity audit for finished adaptation runs (CPU is enough; no training).

For every run directory and every saved checkpoint it rebuilds the head from the checkpoint, refits the detector inventory under the
driver's rule on the saved features, and checks:
  (1) finiteness of the saved features (fit, val, test, every OOD set, including rescored OOD features);
  (2) finiteness of every detector's raw confidences on fit, test and each OOD set, for the adapted and the reference variant;
  (3) strict recomputation of the adapted and reference all-ID AUROC against the recorded outcomes (max abs difference);
  (4) detector orientation: recomputed AUROC below 0.5 (an inverted score, e.g. pNML on Pets).
Writes <run>/score_validity.csv (one row per step x variant x detector x set) and prints a summary per run. Exit code 1 if any
nonfinite feature or score, or any AUROC mismatch above --tol, is found in any run.

  python x8_pool_a/score_validity_audit.py $EXPERIMENT_ROOT_DIR/adaptation_ladder/*_seed0 --device cpu
  python x8_pool_a/score_validity_audit.py $EXPERIMENT_ROOT_DIR/adaptation_pilot/B_full_seed1 --device cuda
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

CODE_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR)); sys.path.insert(0, str(CODE_DIR / "x8_pool_a"))
from adaptation_trajectory import auroc, fit_detectors  # noqa: E402
import pool_a_csfs as csf  # noqa: E402


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def audit_run(run: pathlib.Path, device: str, tol: float):
    import json
    cfg = json.load(open(run / "ledger.json"))["config"]
    steps = sorted(int(p.stem.split("step")[1]) for p in run.glob("features_step*.npz"))
    recorded = pd.concat([pd.read_csv(f) for f in sorted(run.glob("outcomes*.csv"))], ignore_index=True) if list(run.glob("outcomes*.csv")) else pd.DataFrame()
    to_t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).float().to(device)
    z0 = np.load(run / "features_step0.npz"); n_cls = int(z0["fit_y"].max()) + 1
    probe = csf.train_probe(to_t(z0["fit_h"]), torch.from_numpy(z0["fit_y"]).long().to(device), n_cls, seed=cfg["seed"])
    mu, sd = probe["mu"].detach(), probe["sd"].detach()
    rows, ref, flags = [], {}, []
    for step in steps:
        z = np.load(run / f"features_step{step}.npz")
        ood = {k[4:]: z[k] for k in z.files if k.startswith("ood_")}
        for f in sorted(run.glob(f"rescore_features_*_step{step}.npz")):   # rescored OOD sets, same checkpoint
            zr = np.load(f); ood.update({k[4:]: zr[k] for k in zr.files if k.startswith("ood_")})
        for k in ("fit", "val", "test"):
            rows.append(dict(step=step, variant="features", detector="", set=k, n=len(z[f"{k}_h"]), n_nonfinite=int((~np.isfinite(z[f"{k}_h"])).sum())))
        for k, v in ood.items():
            rows.append(dict(step=step, variant="features", detector="", set=k, n=len(v), n_nonfinite=int((~np.isfinite(v)).sum())))
        if step == 0: W, b = probe["W"].detach(), probe["b"].detach()
        else:
            ck = torch.load(run / f"ckpt_step{step}.pt", map_location=device); W, b = ck["head.weight"].to(device), ck["head.bias"].to(device)
        w_raw = (W / sd).detach(); h = {k: to_t(z[f"{k}_h"]) for k in ("fit", "val", "test")}; y = {k: torch.from_numpy(z[f"{k}_y"]).long().to(device) for k in ("fit", "val")}
        all_confs, _ = fit_detectors(h["fit"], y["fit"], h["val"], y["val"], w_raw, b, mu, sd, n_cls, device)
        c_fit, _ = all_confs(h["fit"]); c_te, _ = all_confs(h["test"]); c_ood = {k: all_confs(to_t(v))[0] for k, v in ood.items()}
        if step == 0: ref = {"c_te": c_te, "c_ood": c_ood}
        variants = {"adapted": (c_te, c_ood)} if step == 0 else {"adapted": (c_te, c_ood), "reference": (ref["c_te"], {k: ref["c_ood"][k] for k in c_ood if k in ref["c_ood"]})}
        for var, (cte, cood) in variants.items():
            for det in cte:
                s_fit, s_te = _np(c_fit[det]) if var == "adapted" else None, _np(cte[det])
                if s_fit is not None: rows.append(dict(step=step, variant=var, detector=det, set="fit", n=len(s_fit), n_nonfinite=int((~np.isfinite(s_fit)).sum())))
                rows.append(dict(step=step, variant=var, detector=det, set="test", n=len(s_te), n_nonfinite=int((~np.isfinite(s_te)).sum())))
                for name, so in cood.items():
                    s_ood = _np(so[det]); nf = int((~np.isfinite(s_ood)).sum()) + int((~np.isfinite(s_te)).sum())
                    a = auroc(s_te, s_ood) if nf == 0 else np.nan
                    rec = recorded[(recorded.step == step) & (recorded.ood_set == name) & (recorded.detector == det) & (recorded.variant == var)].auroc_allid if len(recorded) else pd.Series(dtype=float)
                    rows.append(dict(step=step, variant=var, detector=det, set=name, n=len(s_ood), n_nonfinite=nf, auroc_recomputed=a,
                                     auroc_recorded=float(rec.iloc[0]) if len(rec) else np.nan, inverted=bool(a < 0.5) if nf == 0 else False))
    df = pd.DataFrame(rows); df.insert(0, "run", run.name); df.to_csv(run / "score_validity.csv", index=False)
    nf = df[df.n_nonfinite > 0]; both = df.dropna(subset=["auroc_recomputed", "auroc_recorded"]) if "auroc_recorded" in df else df.iloc[0:0]
    mism = (both.auroc_recomputed - both.auroc_recorded).abs()
    inv = df[df.get("inverted", pd.Series(False, index=df.index)) == True] if "inverted" in df else df.iloc[0:0]
    print(f"\n## {run.name}: steps {steps}; rows {len(df)}")
    print(f"nonfinite features/scores: {len(nf)} rows" + (":\n" + nf[["step", "variant", "detector", "set", "n_nonfinite"]].to_string(index=False) if len(nf) else ""))
    print(f"AUROC recomputed vs recorded: {len(both)} pairs, max abs diff {mism.max() if len(mism) else float('nan'):.2e}, above tol {int((mism > tol).sum()) if len(mism) else 0}")
    if len(inv): print("inverted detectors (recomputed AUROC < 0.5):\n" + inv.groupby(["variant", "detector"]).agg(n_cells=("step", "size"), min_auroc=("auroc_recomputed", "min"), sets=("set", lambda s: ",".join(sorted(set(s))))).to_string())
    return len(nf) == 0 and (len(mism) == 0 or (mism <= tol).all())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--device", default="cpu"); ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args(); ok = True
    for r in a.runs:
        r = pathlib.Path(r)
        if not (r / "features_step0.npz").exists(): print(f"skip {r}: no features"); continue
        ok = audit_run(r, a.device, a.tol) and ok
    sys.exit(0 if ok else 1)

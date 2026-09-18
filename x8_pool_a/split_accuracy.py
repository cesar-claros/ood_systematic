"""Fit, validation and test accuracy per checkpoint from the saved features and checkpoints of finished runs (CPU, seconds per run).
Writes <run>/accuracy_splits.csv (step, fit_acc, val_acc, test_acc, test_acc_recorded). The validation accuracy is the permitted
ID-side quantity for the corrected ladder readout (ladder_readout_v2.py picks the file up when present).

  python x8_pool_a/split_accuracy.py $EXPERIMENT_ROOT_DIR/adaptation_ladder/*_seed0 $EXPERIMENT_ROOT_DIR/adaptation_pilot/*_seed*
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

CODE_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR)); sys.path.insert(0, str(CODE_DIR / "x8_pool_a"))
import pool_a_csfs as csf  # noqa: E402


def main(runs):
    for run in map(pathlib.Path, runs):
        if not (run / "features_step0.npz").exists(): print("skip", run); continue
        cfg = json.load(open(run / "ledger.json"))["config"]; m = pd.read_csv(run / "measurements.csv").set_index("step")
        z0 = np.load(run / "features_step0.npz"); n_cls = int(z0["fit_y"].max()) + 1; to_t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).float()
        probe = csf.train_probe(to_t(z0["fit_h"]), torch.from_numpy(z0["fit_y"]).long(), n_cls, seed=cfg["seed"]); mu, sd = probe["mu"].detach(), probe["sd"].detach()
        rows = []
        for step in sorted(int(p.stem.split("step")[1]) for p in run.glob("features_step*.npz")):
            z = np.load(run / f"features_step{step}.npz")
            if step == 0: W, b = probe["W"].detach(), probe["b"].detach()
            else: ck = torch.load(run / f"ckpt_step{step}.pt", map_location="cpu"); W, b = ck["head.weight"].float(), ck["head.bias"].float()
            acc = {k: float(((((to_t(z[f"{k}_h"]) - mu) / sd) @ W.T + b).argmax(1) == torch.from_numpy(z[f"{k}_y"]).long()).float().mean()) for k in ("fit", "val", "test")}
            rows.append(dict(step=step, fit_acc=acc["fit"], val_acc=acc["val"], test_acc=acc["test"], test_acc_recorded=float(m.id_test_acc.loc[step])))
        df = pd.DataFrame(rows); df.to_csv(run / "accuracy_splits.csv", index=False)
        print(run.name, "max |test - recorded| = %.2e" % (df.test_acc - df.test_acc_recorded).abs().max(), "| val acc:", " ".join(f"{int(s)}:{v:.3f}" for s, v in zip(df.step, df.val_acc)))


if __name__ == "__main__":
    main(sys.argv[1:])

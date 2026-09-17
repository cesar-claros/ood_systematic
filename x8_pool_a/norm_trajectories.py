"""H2 diagnostic (zero GPU): ID and OOD feature-norm trajectories from a run's saved features_step*.npz files.

For each checkpoint: mean and median norm of ID test features, of each OOD set, the OOD/ID mean-norm ratio, the fraction of OOD
features whose norm is below the ID median, and the mean squared distance to the ID fit-set global mean (a norm-free typicality proxy).
The norm-collapse hypothesis predicts an OOD/ID ratio crossing below 1 under full fine-tuning on weakly collapsed references.
Run on the cluster, from <repo_path>/code:  python x8_pool_a/norm_trajectories.py $EXPERIMENT_ROOT_DIR/adaptation_pilot/A_full_seed0 [more run dirs]
Writes norm_trajectories.csv into each run directory and prints a summary table.
"""
import pathlib
import sys

import numpy as np
import pandas as pd


def summarize(run: pathlib.Path) -> pd.DataFrame:
    rows = []
    for f in sorted(run.glob("features_step*.npz"), key=lambda p: int(p.stem.split("step")[1])):
        step = int(f.stem.split("step")[1]); z = np.load(f)
        fit, test = z["fit_h"].astype(np.float64), z["test_h"].astype(np.float64); mu = fit.mean(0)
        id_norm = np.linalg.norm(test, axis=1); id_med = np.median(id_norm); id_d2 = ((test - mu) ** 2).sum(1).mean()
        rows.append({"step": step, "set": "ID_test", "n": len(test), "norm_mean": id_norm.mean(), "norm_median": id_med, "norm_cv": id_norm.std() / id_norm.mean(),
                     "ratio_to_id_mean": 1.0, "frac_below_id_median": 0.5, "dist2_to_id_mean": id_d2, "dist2_ratio": 1.0})
        for k in [k for k in z.files if k.startswith("ood_")]:
            o = z[k].astype(np.float64); n = np.linalg.norm(o, axis=1); d2 = ((o - mu) ** 2).sum(1).mean()
            rows.append({"step": step, "set": k[4:], "n": len(o), "norm_mean": n.mean(), "norm_median": np.median(n), "norm_cv": n.std() / n.mean(),
                         "ratio_to_id_mean": n.mean() / id_norm.mean(), "frac_below_id_median": float((n < id_med).mean()), "dist2_to_id_mean": d2, "dist2_ratio": d2 / id_d2})
    df = pd.DataFrame(rows); df.to_csv(run / "norm_trajectories.csv", index=False); return df


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for arg in sys.argv[1:]:
        run = pathlib.Path(arg); df = summarize(run)
        print(f"\n=== {run.name} ===")
        print(df[df.set != "ID_test"].pivot(index="step", columns="set", values="ratio_to_id_mean").round(3).rename_axis(columns="OOD/ID mean-norm ratio").to_string())
        print(df[df.set != "ID_test"].pivot(index="step", columns="set", values="dist2_ratio").round(3).rename_axis(columns="OOD/ID squared-distance-to-mean ratio").to_string())
        print("ID norm CV by step:", df[df.set == "ID_test"].set_index("step").norm_cv.round(4).to_dict())

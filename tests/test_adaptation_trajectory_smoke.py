"""CPU smoke test for x8_pool_a/adaptation_trajectory.py (toy backbone, synthetic data). Checks mechanics only, never outcomes."""
import pathlib
import subprocess
import sys

import pandas as pd

CODE = pathlib.Path(__file__).resolve().parents[1]


def test_trajectory_driver_runs(tmp_path):
    out = tmp_path / "traj"
    cmd = [sys.executable, str(CODE / "x8_pool_a" / "adaptation_trajectory.py"), "--backbone", "toy", "--data", "synthetic", "--method", "lora",
           "--steps", "6", "--checkpoints", "0,3,6", "--n-fit", "120", "--n-val", "80", "--n-test", "60", "--n-ood", "60", "--out", str(out)]
    subprocess.run(cmd, check=True, cwd=CODE, capture_output=True, timeout=600)
    rows = pd.read_csv(out / "outcomes.csv"); meas = pd.read_csv(out / "measurements.csv")
    assert rows.detector.nunique() == 20 and set(rows.variant) == {"adapted", "reference", "combined"}
    assert set(rows.step) == {0, 3, 6} and rows.groupby(["step", "variant"]).size().min() == 40
    assert {"residue_energy_projector", "nc_var_collapse", "paired_cka_fit", "total_drift_fit", "id_test_acc"} <= set(meas.columns)
    assert (out / "ledger.json").exists() and (out / "ckpt_step6.pt").exists() and (out / "features_step0.npz").exists()
    assert rows.auroc_allid.between(0, 1).all()

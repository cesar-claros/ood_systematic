"""Retarget protocol P-2: generate long_harmonized_v2, the ICML-cycle
table of record (frozen 2026-08-31 with the protocol).

RULE (frozen): v1 is never edited. v2 = v1 plus two columns:
- `var_collapse_corrected`: the stated-definition NC1. For the 280
  VGG13 pool checkpoints it comes from the audited pool_coords pilot0
  papyan panel (numpy convention, float64, rcond 1e-6 hermitian pinv);
  for ResNet18 and ViT rows it is NaN until the D-R4 re-measurement
  with the fixed pipeline fills it (that fill produces v2.1 with a new
  hash; this script then takes --panels <dir> inputs).
- `nc1_convention`: 'pilot0_corrected' where filled, 'pending_remeasure'
  otherwise.
Every ICML-cycle analysis reads v2 and the corrected column; the legacy
`var_collapse` column stays untouched for the audit trail.

Usage (from code/): python make_harmonized_v2.py
Output: nc_csf_predictivity/outputs/track1/dataset/long_harmonized_v2.parquet
        (sha256 printed; commit the script and record the hash)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from heldout_theory_validation import load_coords

V1 = Path("nc_csf_predictivity/outputs/track1/dataset/"
          "long_harmonized.parquet")
V2 = V1.with_name("long_harmonized_v2.parquet")


def main() -> None:
    df = pd.read_parquet(V1)
    coords, problems = load_coords(Path("pilot0/pool_coords"))
    assert not problems, problems
    cell = (df.paradigm.astype(str) + "|" + df.source.astype(str) + "|"
            + df["run"].astype(str) + "|" + df.reward.astype(str) + "|"
            + df.dropout.astype(str))
    vc_map = {c: coords[c]["papyan"]["var_collapse"] for c in coords}
    corrected = np.where(df.architecture.values == "VGG13",
                         cell.map(vc_map).values, np.nan)
    df["var_collapse_corrected"] = corrected.astype(float)
    df["nc1_convention"] = np.where(
        np.isfinite(df.var_collapse_corrected.values),
        "pilot0_corrected", "pending_remeasure")
    n_vgg = int((df.architecture == "VGG13").sum())
    n_fill = int(np.isfinite(df.var_collapse_corrected.values).sum())
    assert n_fill == n_vgg, (n_fill, n_vgg)
    df.to_parquet(V2)
    h = hashlib.sha256(V2.read_bytes()).hexdigest()
    print(f"v2 written: {len(df)} rows; corrected filled {n_fill} "
          f"(all VGG13); pending {len(df) - n_fill} (ResNet18+ViT)")
    print(f"sha256 {h}")


if __name__ == "__main__":
    main()

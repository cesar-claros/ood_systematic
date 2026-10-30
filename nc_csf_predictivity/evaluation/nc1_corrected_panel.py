"""Corrected NC1 for every model pool of the NeurIPS paper.

The paper's NC1 ("var_collapse") divides the within-class covariance by
N*C instead of N, so it equals Papyan's NC1 divided by C (Appendix B). The
fixed estimator (`src/neural_collapse.py` since 77d87b2) divides by N and
pins the pseudoinverse of Sigma_B (float64, hermitian, rcond 1e-6). This
script collects the corrected value per model:

  VGG13 (280)        pilot0 Papyan panel (`pilot0/pool_coords`), the source of
                     `var_collapse_corrected` in long_harmonized_v2.
  ResNet18 (56)      aug-view Papyan panel of the D-R4 re-measurement
                     (`rn18_handoff_replication/outputs/phase1_nc1`): the
                     datamodule's training set with training augmentations,
                     the view of the paper's NC metrics.
  SSL probes (40),   legacy x C, exact: their estimators already use the
  ImageNet (47)      fixed pseudoinverse (numpy float64, rcond 1e-6).
  ViT (40)           legacy x C, approximate: no re-measurement exists, and
                     the legacy ViT run used torch's default pseudoinverse.

Gates (abort before the table is written):
  GP  the remeasured panels come from the paper's view: for VGG13 and
      ResNet18, max_equiangular_wc (a weight-only metric, so identical for
      the same checkpoint) matches to rel 1e-5 and the class-mean angle
      metrics (equiangular_uc, max_equiangular_uc) to median rel 1e-2.
  GN  every pool has its full model count and no missing value.

Outputs: outputs/nc1_corrected/nc1_corrected.csv, outputs/54_nc1_corrected.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent
CODE_DIR = EVAL_DIR.parents[1]
for path in (EVAL_DIR.parent / "ablations", EVAL_DIR, CODE_DIR):
    sys.path.insert(0, str(path))
from calibration_features_clique import DEFAULT_OUT_ROOT, add_model_id  # noqa: E402
from heldout_theory_validation import load_coords  # noqa: E402
from regimefree_wilcoxon import load_parquet  # noqa: E402

N_CLASSES = {"cifar10": 10, "cifar100": 100, "supercifar100": 20, "tinyimagenet": 200,
             "imagenet": 1000}
POOL_SIZE = {"VGG13": 280, "ResNet18": 56, "ViT": 40, "SSL probes": 40, "ImageNet": 47}
RN18_DIR = CODE_DIR / "rn18_handoff_replication" / "outputs" / "phase1_nc1"
RN18_SOURCES = {"supercifar": "supercifar100", "tiny-imagenet-200": "tinyimagenet"}
WEIGHT_TOL, ANGLE_TOL = 1e-5, 1e-2
ANGLES = ["equiangular_uc", "max_equiangular_uc"]


def gate_view(pool: str, paper: pd.DataFrame, panel: pd.DataFrame) -> None:
    """GP: a remeasured panel comes from the same checkpoints and data view.

    Raises:
        RuntimeError: If a weight metric or the class-mean angles differ.
    """
    both = paper.merge(panel, on="model_id", suffixes=("", "_panel"))
    rel = lambda k: ((both[k] - both[f"{k}_panel"]).abs() / both[k].abs())  # noqa: E731
    weight, angle = rel("max_equiangular_wc").max(), max(rel(k).median() for k in ANGLES)
    if len(both) != POOL_SIZE[pool] or weight > WEIGHT_TOL or angle > ANGLE_TOL:
        raise RuntimeError(f"GP failed for {pool}: {len(both)} models, weight rel "
                           f"{weight:.2e}, angle median rel {angle:.2e}")
    print(f"GP: {pool} panel matches the paper's view (weight rel {weight:.1e}, "
          f"angle median rel {angle:.1e})")


def vgg_panel(models: pd.DataFrame) -> pd.DataFrame:
    """Papyan panel of the 280 VGG13 checkpoints from pilot0, keyed by model id."""
    coords, problems = load_coords(CODE_DIR / "pilot0" / "pool_coords")
    if problems:
        raise RuntimeError(f"pilot0 coords: {problems}")
    vgg = models[models["architecture"] == "VGG13"]
    cell = (vgg["paradigm"].astype(str) + "|" + vgg["source"].astype(str) + "|"
            + vgg["run"].astype(str) + "|" + vgg["reward"].astype(str) + "|"
            + vgg["dropout"].astype(str))
    panel = pd.DataFrame([coords[c]["papyan"] for c in cell])
    return panel.assign(model_id=vgg["model_id"].values)


def rn18_panel() -> pd.DataFrame:
    """Aug-view Papyan panel of the paper's 56 ResNet18 checkpoints."""
    recs = []
    for path in sorted(RN18_DIR.glob("*.json")):
        if path.name.startswith("FAILED_"):
            continue
        rec = json.loads(path.read_text())
        if rec["paradigm"] not in ("confidnet", "devries", "dg"):
            continue
        source = RN18_SOURCES.get(rec["source"], rec["source"])
        recs.append({"model_id": f"ResNet18|{rec['paradigm']}|{source}|{rec['run_label']}"
                                 f"|{int(rec['dropout'])}|{float(rec['reward']):g}",
                     **rec["aug_view"]["papyan"]})
    return pd.DataFrame(recs)


def rescaled(pool: str, frame: pd.DataFrame, exact: bool) -> pd.DataFrame:
    """Legacy NC1 times the number of classes."""
    c = frame["source"].map(N_CLASSES)
    return pd.DataFrame({"pool": pool, "model_id": frame["model_id"].values,
                         "source": frame["source"].values, "n_classes": c.values,
                         "nc1_legacy": frame["var_collapse"].values,
                         "nc1_corrected": (frame["var_collapse"] * c).values,
                         "convention": "legacy x C (exact)" if exact
                         else "legacy x C (approximate, legacy pinv)"})


def main() -> None:
    """Collect, gate, and report the corrected NC1 of every pool."""
    out_root = Path(DEFAULT_OUT_ROOT)
    long_df = add_model_id(load_parquet(out_root / "track1" / "dataset"
                                        / "long_harmonized.parquet"))
    models = long_df.drop_duplicates("model_id")
    parts = []
    for pool, panel, convention in (("VGG13", vgg_panel(models), "pilot0 remeasure"),
                                    ("ResNet18", rn18_panel(), "D-R4 aug-view remeasure")):
        paper = models[models["architecture"] == pool]
        gate_view(pool, paper, panel)
        both = paper.merge(panel[["model_id", "var_collapse"]], on="model_id",
                           suffixes=("", "_panel"))
        parts.append(pd.DataFrame({
            "pool": pool, "model_id": both["model_id"], "source": both["source"],
            "n_classes": both["source"].map(N_CLASSES), "nc1_legacy": both["var_collapse"],
            "nc1_corrected": both["var_collapse_panel"], "convention": convention}))
    parts.append(rescaled("ViT", models[models["architecture"] == "ViT"], exact=False))
    ssl = load_parquet(out_root / "pool_a" / "models_pool_a_harmonized.parquet")
    ssl = ssl.assign(model_id=ssl["paradigm"] + "|" + ssl["source"] + "|"
                     + ssl["run"].astype(str))
    parts.append(rescaled("SSL probes", ssl, exact=True))
    imagenet = pd.concat(map(pd.read_parquet, sorted((CODE_DIR / "x9_outputs")
                                                     .glob("*_model.parquet"))))
    parts.append(rescaled("ImageNet", imagenet.assign(model_id=imagenet["tag"],
                                                      source="imagenet"), exact=True))
    table = pd.concat(parts, ignore_index=True)
    counts = table.groupby("pool")["model_id"].nunique()
    if (counts.reindex(list(POOL_SIZE)) != pd.Series(POOL_SIZE)).any() \
            or table[["nc1_legacy", "nc1_corrected"]].isna().any().any():
        raise RuntimeError(f"GN failed: {counts.to_dict()}")
    print(f"GN: {counts.to_dict()}")

    table["ratio_to_c"] = table["nc1_corrected"] / (table["nc1_legacy"]
                                                    * table["n_classes"])
    summary = (table.groupby(["pool", "source"], sort=False)
               .agg(models=("model_id", "nunique"), C=("n_classes", "first"),
                    legacy_median=("nc1_legacy", "median"),
                    corrected_median=("nc1_corrected", "median"),
                    corrected_min=("nc1_corrected", "min"),
                    corrected_max=("nc1_corrected", "max"),
                    ratio_to_c_median=("ratio_to_c", "median"),
                    ratio_to_c_max=("ratio_to_c", "max"),
                    convention=("convention", "first"))
               .reset_index())
    csv_dir = out_root / "nc1_corrected"
    csv_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_dir / "nc1_corrected.csv", index=False)
    lines = ["# Corrected NC1 for every pool\n\n",
             "**Source:** `nc_csf_predictivity/evaluation/nc1_corrected_panel.py`. "
             "Corrected NC1 = Tr(Sigma_W Sigma_B^+)/C with Sigma_W averaged over N "
             "(Papyan); legacy = the paper's value (corrected / C under the legacy "
             "pseudoinverse). ratio_to_c = corrected / (legacy x C): 1 when only the "
             "normalization differs.\n\n", "Gates GP and GN passed.\n\n```\n",
             summary.round(4).to_string(index=False), "\n```\n"]
    (out_root / "54_nc1_corrected.md").write_text("".join(lines))
    print(f"wrote {out_root / '54_nc1_corrected.md'}")


if __name__ == "__main__":
    main()

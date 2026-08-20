"""A2 row-span / nullspace geometry (evaluation doc section 4.2, order-of-
work step 3). HPC-side: loads each Pilot 1 checkpoint, forwards the train
split, and measures how much of the centered class-mean matrix M lives
outside the row span of the classifier W.

Motivation: a rank-(C-1) fixed ETF head at C = 100, D = 512 leaves a
nullspace of dimension >= 413 that cross-entropy cannot constrain, so
class means can acquire large class-dependent nullspace components without
changing logits or accuracy. If A2's self-duality is poor in full space
but good after projecting M into the row span of W, the A2 finding is a
nullspace-leakage result rather than an alignment failure inside the span.

Per model, reports:
  eta_perp              ||M (I - P_W)||_F^2 / ||M||_F^2 (nullspace
                        fraction of the centered class means)
  self_duality_full     Papyan metric in full space (cross-check against
                        the manipulation-stage geometry JSONs)
  self_duality_proj     the same metric with M replaced by M P_W
  rank_w                numerical rank of W
  per-class nullspace fractions (median / max)

Baselines and A1 arms are measured too: their learned heads have rank at
most C, so eta_perp is well defined everywhere and gives the reference
level against which A2 is judged.

Usage (from code/, inside the frozen container, on the HPC):
    python nc_csf_predictivity/interventions/a2_nullspace_extract.py \
        [--group cifar100_intervention] [--experiments name ...] \
        [--out_dir nc_csf_predictivity/interventions/nullspace]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# Pure math (numpy only; unit-tested locally).
# ---------------------------------------------------------------------------

def rowspan_basis(w: np.ndarray) -> np.ndarray:
    """Orthonormal basis (rows) of the row span of W, numerical rank cut."""
    _, s, vt = np.linalg.svd(w.astype(np.float64), full_matrices=False)
    tol = s[0] * max(w.shape) * np.finfo(np.float64).eps
    rank = int((s > tol).sum())
    return vt[:rank]


def nullspace_fractions(m: np.ndarray, basis: np.ndarray
                        ) -> tuple[float, np.ndarray]:
    """(pooled eta_perp, per-class fractions) for centered class means M."""
    m64 = m.astype(np.float64)
    m_in = (m64 @ basis.T) @ basis
    m_out = m64 - m_in
    per_class = (m_out ** 2).sum(1) / np.maximum((m64 ** 2).sum(1), 1e-30)
    eta = float((m_out ** 2).sum() / (m64 ** 2).sum())
    return eta, per_class


def self_duality(w: np.ndarray, m: np.ndarray) -> float:
    """Papyan self-duality metric ||W/||W||_F - M/||M||_F||_F^2."""
    w64, m64 = w.astype(np.float64), m.astype(np.float64)
    return float(np.sum((w64 / np.linalg.norm(w64)
                         - m64 / np.linalg.norm(m64)) ** 2))


def nullspace_record(w: np.ndarray, m: np.ndarray) -> dict:
    basis = rowspan_basis(w)
    eta, per_class = nullspace_fractions(m, basis)
    m_proj = (m.astype(np.float64) @ basis.T) @ basis
    return {
        "rank_w": len(basis),
        "eta_perp": eta,
        "per_class_median": float(np.median(per_class)),
        "per_class_max": float(per_class.max()),
        "self_duality_full": self_duality(w, m),
        "self_duality_proj": self_duality(w, m_proj),
    }


# ---------------------------------------------------------------------------
# HPC driver (fd-shifts imports deferred so the math is testable locally).
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="A2 nullspace geometry extraction")
    parser.add_argument("--group", type=str, default="cifar100_intervention")
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--out_dir", type=str,
                        default="nc_csf_predictivity/interventions/nullspace")
    parser.add_argument("--use_cuda", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import os

    import torch
    from fd_shifts import logger
    from fd_shifts.loaders.data_loader import FDShiftsDataLoader
    from fd_shifts.models import get_model
    from fd_shifts.utils import exp_utils

    from nc_csf_predictivity.interventions.extract_manipulation import (
        NAME_RE,
        forward_split,
    )
    from pilot0.geometry import fit_feature_model
    from src import utils

    device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_root = Path(os.environ["EXPERIMENT_ROOT_DIR"])
    names = args.experiments or sorted(
        p.name for p in (exp_root / args.group).iterdir() if p.is_dir())

    records = []
    for name in names:
        meta = NAME_RE.match(name)
        if meta is None:
            logger.warning(f"Skipping unparseable experiment name: {name}")
            continue
        out_path = out_dir / f"{name}.json"
        if out_path.exists() and not args.overwrite:
            records.append(json.loads(out_path.read_text()))
            logger.info(f"skipping existing {out_path}")
            continue
        path = f"{args.group}/{name}"
        logger.info(f"=== {path} ===")
        cf = utils.get_conf(path, "intervention")
        datamodule = FDShiftsDataLoader(cf)
        datamodule.setup()
        module = get_model(cf.model.name)(cf)
        module.load_only_state_dict(exp_utils._get_path_to_best_ckpt(
            cf.exp.dir, "last", cf.test.selection_mode), device="cpu")
        module.model.encoder.disable_dropout()
        module.to(device)
        module.eval()

        h_tr, y_tr, _ = forward_split(module, datamodule.train_dataloader(),
                                      device)
        model = fit_feature_model(h_tr, y_tr, int(cf.data.num_classes))
        w, _ = utils.get_model_and_last_layer(module, "intervention",
                                              return_model=False)
        record = nullspace_record(w.detach().cpu().numpy(),
                                  model.class_means)
        record.update({"experiment": path, "kind": meta["kind"],
                       "run": int(meta["run"]), "lam": meta["lam"]})
        out_path.write_text(json.dumps(record, indent=1))
        records.append(record)
        logger.info(f"  eta_perp={record['eta_perp']:.4f} "
                    f"sd_full={record['self_duality_full']:.4f} "
                    f"sd_proj={record['self_duality_proj']:.4f}")

    lines = ["# A2 nullspace geometry (evaluation doc 4.2)", "",
             ("| model | lam | rank(W) | eta_perp | per-class max "
             "| self-duality full | self-duality projected |"),
             "|---|---|---|---|---|---|---|"]
    for r in sorted(records, key=lambda r: (r["lam"], r["run"])):
        lines.append(
            f"| run{r['run']} | {r['lam']} | {r['rank_w']} "
            f"| {r['eta_perp']:.4f} | {r['per_class_max']:.4f} "
            f"| {r['self_duality_full']:.4f} "
            f"| {r['self_duality_proj']:.4f} |")
    lines.append("")
    lines.append("Reading: if A2's projected self-duality is small while "
                 "its full-space value is large, the A2 refutation is a "
                 "nullspace-leakage result (features escape through "
                 "ker W, which cross-entropy cannot constrain), and "
                 "fixed-classifier training needs explicit span or "
                 "class-mean control.")
    report = out_dir / "a2_nullspace_report.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"wrote {report} ({len(records)} models)")


if __name__ == "__main__":
    main()

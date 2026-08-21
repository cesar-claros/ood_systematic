"""Maha amplitude-operator repair: per-model extraction (HPC side).

B-axis protocol section 7 prerequisite. For every Pilot 1 model this
computes, per OOD set, the FROZEN closed-form Maha prediction, the
min-statistic repaired prediction (`pilot0.theory.predicted_maha_auroc_min`),
and the audit section-5.4 empirical measurables from actual features:
nearest-prototype switching rates, score means/variances, and the
correct-filtered empirical Maha AUROC. Pilot 1 is design data (audit
section 7): blinding is over for these models, so the ID test split is
forwarded here (unlike stage 2b).

Usage (from code/, inside the container, on the HPC):
    python nc_csf_predictivity/interventions/maha_repair_extract.py \
        [--group cifar100_intervention] [--experiments name ...] \
        [--out_dir nc_csf_predictivity/interventions/maha_repair] \
        [--n_samples 4000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# The repair validation is defined on the Pilot 1 pool ONLY. B-pilot
# models (varreg/ctrreg) are geometry-only until dose selection
# (B_axis_pilot_protocol.md: no OOD forward, no detector score), so they
# are skipped even if explicitly listed or present in the group dir.
PILOT1_KINDS = ("etfreg", "etfhard")


def d2_and_scores(h: np.ndarray, means: np.ndarray,
                  precision: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample distance matrix argmin and negated-min scores (GEMM)."""
    h64 = h.astype(np.float64)
    m_quad = np.einsum("cd,dk,ck->c", means, precision, means,
                       optimize=True)
    h_prec = h64 @ precision
    d2 = ((h_prec * h64).sum(1)[:, None]
          - 2.0 * h_prec @ means.T + m_quad[None, :])
    return d2.argmin(1), -d2.min(1)


def rank_auroc(s_id: np.ndarray, s_ood: np.ndarray) -> float:
    from scipy.stats import rankdata
    ranks = rankdata(np.concatenate([s_id, s_ood]))
    n_id, n_ood = len(s_id), len(s_ood)
    return float((ranks[:n_id].sum() - n_id * (n_id + 1) / 2.0)
                 / (n_id * n_ood))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Maha repair extraction (old vs min-statistic)")
    parser.add_argument("--group", type=str, default="cifar100_intervention")
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument(
        "--out_dir", type=str,
        default="nc_csf_predictivity/interventions/maha_repair")
    parser.add_argument("--n_samples", type=int, default=4000)
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
    from nc_csf_predictivity.interventions.stage2b_extract_predict import (
        OOD_TEST_SETS,
        _ForwardAdapter,
    )
    from pilot0.geometry import fit_feature_model
    from pilot0.scores import MahalanobisScorer
    from pilot0.theory import predicted_maha_auroc, predicted_maha_auroc_min
    from src import utils

    class _LabeledAdapter(_ForwardAdapter):
        """stage2b adapter + labels (needed here for the correct-filter
        and the empirical ID switching rate; stage2b never used them)."""

        @torch.no_grad()
        def __call__(self, batch, idx):
            out = _ForwardAdapter.__call__(self, batch, idx)
            out["labels"] = batch[1].cpu()
            return out

    device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_root = Path(os.environ["EXPERIMENT_ROOT_DIR"])
    names = args.experiments or sorted(
        p.name for p in (exp_root / args.group).iterdir() if p.is_dir())

    for name in names:
        meta = NAME_RE.match(name)
        if meta is None:
            logger.warning(f"Skipping unparseable experiment name: {name}")
            continue
        if meta["kind"] not in PILOT1_KINDS:
            logger.info(f"skipping {name}: B-pilot models are geometry-only "
                        f"until dose selection (protocol rule)")
            continue
        out_path = out_dir / f"{name}.json"
        if out_path.exists() and not args.overwrite:
            logger.info(f"skipping existing {out_path}")
            continue
        path = f"{args.group}/{name}"
        logger.info(f"=== {path} ===")
        try:
            cf = utils.get_conf(path, "intervention")
            datamodule = FDShiftsDataLoader(cf)
            datamodule.setup()
            module = get_model(cf.model.name)(cf)
            module.load_only_state_dict(exp_utils._get_path_to_best_ckpt(
                cf.exp.dir, "last", cf.test.selection_mode), device="cpu")
            module.model.encoder.disable_dropout()
            module.to(device)
            module.eval()

            h_tr, y_tr, _ = forward_split(
                module, datamodule.train_dataloader(), device)
            n_classes = int(cf.data.num_classes)
            model = fit_feature_model(h_tr, y_tr, n_classes)
            maha = MahalanobisScorer(h_tr, y_tr, n_classes)
            means, precision = maha.means, maha.precision
            w, b = utils.get_model_and_last_layer(module, "intervention",
                                                  return_model=False)
            w_np = w.detach().cpu().numpy().astype(np.float64)
            b_np = b.detach().cpu().numpy().astype(np.float64)
            adapter = _LabeledAdapter(module, cf, device)

            ev = utils.compute_model_evaluations(adapter, datamodule,
                                                 "test_1")
            h_id = ev["encoded"].numpy().astype(np.float64)
            y_id = ev["labels"].numpy()
            correct = (h_id @ w_np.T + b_np).argmax(1) == y_id
            argmin_id, s_id_all = d2_and_scores(h_id, means, precision)
            s_id = s_id_all[correct]

            # Repair iteration 2 (validation round 1 finding): the
            # train-fit population covariance predicts ZERO prototype
            # switching while real test features switch ~25%, so the
            # scored-ID population is modeled here from VALIDATION
            # residuals against the TRAIN-fit means (deployment-honest:
            # val is CSF design data; the detector itself stays
            # train-fit).
            ev = utils.compute_model_evaluations(adapter, datamodule,
                                                 "val")
            h_val = ev["encoded"].numpy().astype(np.float64)
            y_val = ev["labels"].numpy()
            resid_val = h_val - means[y_val]
            cov_val = resid_val.T @ resid_val / len(resid_val)
            argmin_val, s_val = d2_and_scores(h_val, means, precision)

            record = {"experiment": path, "kind": meta["kind"],
                      "run": int(meta["run"]), "lam": meta["lam"],
                      "n_samples": args.n_samples,
                      "id_test": {
                          "n": len(h_id),
                          "n_correct": int(correct.sum()),
                          "emp_switch_rate": float(
                              (argmin_id != y_id).mean()),
                          # The scored ID population is correct-filtered;
                          # all-sample switching tracks the error rate, so
                          # this is the rate the population model should
                          # be compared against (round-2 finding).
                          "emp_switch_rate_correct": float(
                              (argmin_id[correct] != y_id[correct]).mean()),
                          "emp_score_mean": float(s_id.mean()),
                          "emp_score_var": float(s_id.var(ddof=1))},
                      "val": {
                          "n": len(h_val),
                          "emp_switch_rate": float(
                              (argmin_val != y_val).mean()),
                          "emp_score_mean": float(s_val.mean()),
                          "emp_score_var": float(s_val.var(ddof=1))},
                      "sets": {}}
            for mode, test_set in OOD_TEST_SETS.items():
                ev = utils.compute_model_evaluations(adapter, datamodule,
                                                     test_set)
                h_ood = ev["encoded"].numpy().astype(np.float64)
                m_o = h_ood.mean(0)
                resid = h_ood - m_o
                cov_o = resid.T @ resid / len(resid)
                pred_old = predicted_maha_auroc(
                    means, precision, model.sigma_w, m_o, cov_o)
                pred_min, diag = predicted_maha_auroc_min(
                    means, model.class_freq, precision, model.sigma_w,
                    m_o, cov_o, n_samples=args.n_samples, seed=0,
                    diagnostics=True)
                pred_old_val = predicted_maha_auroc(
                    means, precision, cov_val, m_o, cov_o)
                pred_min_val, diag_val = predicted_maha_auroc_min(
                    means, model.class_freq, precision, cov_val,
                    m_o, cov_o, n_samples=args.n_samples, seed=0,
                    diagnostics=True)
                argmin_o, s_ood = d2_and_scores(h_ood, means, precision)
                diffs = m_o - means
                c_star = int(np.argmin(
                    ((diffs @ precision) * diffs).sum(1)))
                record["sets"][mode] = {
                    "n_ood": len(h_ood),
                    "pred_old": float(pred_old),
                    "pred_min": float(pred_min),
                    "pred_old_val": float(pred_old_val),
                    "pred_min_val": float(pred_min_val),
                    "mc_diag": diag,
                    "mc_diag_val": diag_val,
                    "emp_auroc": rank_auroc(s_id, s_ood),
                    "emp_ood_nearest_share": float(
                        (argmin_o == c_star).mean()),
                    "emp_ood_score_mean": float(s_ood.mean()),
                    "emp_ood_score_var": float(s_ood.var(ddof=1)),
                }
                logger.info(
                    f"  {mode}: old={pred_old:.3f} min={pred_min:.3f} "
                    f"old_val={pred_old_val:.3f} min_val={pred_min_val:.3f} "
                    f"emp={record['sets'][mode]['emp_auroc']:.3f}")
        except (np.linalg.LinAlgError, ValueError, RuntimeError,
                FileNotFoundError) as err:
            logger.error(f"maha repair extraction failed for {path}: {err}")
            (out_dir / f"{name}.FAILED.json").write_text(json.dumps(
                {"experiment": path, "error": str(err)}, indent=1))
            continue
        out_path.write_text(json.dumps(record, indent=1))
        logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()

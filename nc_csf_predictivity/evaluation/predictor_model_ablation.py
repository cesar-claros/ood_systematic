"""Predictor-model ablation: tuned LR vs tuned RF, with threshold tuning.

Side experiment under one grouped validation protocol (GroupKFold over the
five VGG-13 runs; a held-out run's models never appear in training). Each
(hyperparameter, tau) candidate is fit once per fold and its held-out
probabilities are swept over tau in {0.15..0.65}. Per model family, FOUR
operating points are selected:

  tau05  : best hyperparameters at fixed tau=0.5 by held-out set-regret
           (reproduces the first version of this experiment; gated).
  budget : best (hp, tau) by held-out set-regret SUBJECT TO mean held-out
           shortlist size <= 6.5 (the paper's upper headline size). Regret
           alone is weakly monotone decreasing in tau (tau -> 0 admits the
           full roster and zeroes regret by construction), so a size
           constraint is required to make regret-based tau tuning
           well-posed.
  f1     : best (hp, tau) by mean set-F1 between the predicted shortlist
           and the top clique of the held-out model's cell. This is the
           classification-native, parameter-free criterion: an empty
           shortlist scores recall 0, the full roster collapses precision,
           so both failure modes are penalized without a budget.

STRICT LABELS: all validation-loop labels (training merges AND F1
targets) are recomputed PER FOLD from the four in-fold runs only
(compute_track1_cliques on run != held), so a held-out run's scores never
shape its own labels. Final refits legitimately use the published 5-run
consensus cliques. Remaining caveat, deliberately measured rather than
hidden: training rows from a held-out model's CELL are still present with
their (4-run) labels and near-identical NC features, so a cell-memorizing
model can still score high validation F1; if the strict-label RF F1 stays
near its consensus-label value, the dominant leakage channel is
feature-side cell duplication and the next strictness level is
cell-disjoint folds.
  free   : best (hp, tau) by unconstrained held-out regret, kept ONLY to
           demonstrate the degeneracy (it inflates shortlists toward the
           full roster).

Hyperparameter grids: LogisticRegression (L2, balanced) with C in
logspace(-2,2,7); RandomForest (300 trees, balanced_subsample,
random_state=0) over max_depth {None, 6} x min_samples_leaf {1, 4} x
max_features {sqrt, 0.5}. Reference arm: the paper's per-CSF
LogisticRegressionCV at tau=0.5, gated against the published transfer
numbers. Final: all arms refit on the full VGG pool and evaluated on the
three transfers under the first-reply conventions (ResNet-18 / zero-ViT
cross-family ViT / SSL probes with the VGG-only pool; joint side; best
fixed CSF = Always-of-6), with abstention statistics per arm and a
labeled hindsight-tau diagnostic.

Run from `code/`:
  ./.venv/bin/python nc_csf_predictivity/evaluation/predictor_model_ablation.py
Output: nc_csf_predictivity/outputs/35_predictor_model_ablation.md
"""
from __future__ import annotations

import itertools
import pathlib
import sys

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CODE_DIR = pathlib.Path(__file__).resolve().parents[2]
for p in (CODE_DIR, CODE_DIR / "x8_pool_a",
          CODE_DIR / "nc_csf_predictivity" / "data",
          CODE_DIR / "nc_csf_predictivity" / "ablations",
          CODE_DIR / "nc_csf_predictivity" / "evaluation"):
    sys.path.insert(0, str(p))

from pool_a_analysis import OUT_ROOT, pool_cliques_for  # noqa: E402
from cliques_track1 import compute_track1_cliques  # noqa: E402
from calibration_features_clique import (  # noqa: E402
    NC_PRIMARY,
    add_model_id,
    build_pipeline,
)
from input_ablation_grid import REGIMES, evaluate  # noqa: E402

FEATS = NC_PRIMARY + ["source", "regime"]
TAUS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
SIZE_BUDGET = 6.5   # the paper's upper headline mean shortlist size
CELL_KEYS = ["paradigm", "source", "dropout", "reward", "regime"]

EXPECTED_REF = {          # paper LR, tau=0.5
    "ResNet18": (1.02, 1.18, 0.39),
    "ViT": (1.90, 3.66, 2.07),
    "SSL": (5.99, 3.22, 1.44),
}
EXPECTED_TAU05 = {        # first-version arms at tau=0.5 (regression gates)
    ("LR", "ResNet18"): (0.52, 1.30, 0.48),
    ("LR", "ViT"): (1.92, 2.32, 2.07),
    ("LR", "SSL"): (5.85, 3.39, 3.81),
    ("RF", "ResNet18"): (2.96, 7.98, 2.33),
    ("RF", "ViT"): (9.35, 25.57, 23.64),
    ("RF", "SSL"): (8.42, 56.67, 16.36),
}


def make_pipe(clf) -> Pipeline:
    pre = ColumnTransformer([
        ("nc", "passthrough", NC_PRIMARY),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                              sparse_output=False), ["source", "regime"]),
    ])
    return Pipeline([("preprocess", pre), ("clf", clf)])


def fit_probas(clf_factory, tr: pd.DataFrame, te: pd.DataFrame,
               csf_cols: list[str], paper_cv: bool = False) -> pd.DataFrame:
    """Fit one head per CSF; return per-(model, regime) probabilities."""
    out = te[["model_id", "regime"]].copy()
    for name in csf_cols:
        y = tr[name].astype(int).values
        if y.min() == y.max() or min(np.bincount(y)) < 5:
            continue
        pipe = build_pipeline("source") if paper_cv else \
            make_pipe(clf_factory())
        pipe.fit(tr[FEATS], y)
        out[name] = pipe.predict_proba(te[FEATS])[:, 1]
    return out


def proba_matrix_for(index_pairs: list[tuple], columns: list[str],
                     probas: pd.DataFrame) -> np.ndarray:
    pm = probas.set_index(["model_id", "regime"])
    P = np.zeros((len(index_pairs), len(columns)))
    for j, csf in enumerate(columns):
        if csf in pm.columns:
            P[:, j] = pm[csf].reindex(index_pairs).fillna(0.0) \
                .to_numpy(float)
    return P


class RegretEval:
    """Vectorized threshold-sweep regret over one target's eval rows."""

    def __init__(self, rows: pd.DataFrame):
        rows = rows[rows["regime"].isin(REGIMES)]
        wide = rows.pivot_table(
            index=["model_id", "eval_dataset", "regime"], columns="csf",
            values="augrc", aggfunc="first")
        self.csfs = list(wide.columns)
        self.A = wide.to_numpy(float)
        with np.errstate(invalid="ignore"):
            self.oracle = np.nanmin(self.A, axis=1)
            self.worst = np.nanmax(self.A, axis=1)
        self.index = wide.index

    def subset(self, model_ids) -> "RegretEval":
        mask = self.index.get_level_values("model_id").isin(model_ids)
        sub = RegretEval.__new__(RegretEval)
        sub.csfs, sub.A = self.csfs, self.A[mask]
        sub.oracle, sub.worst = self.oracle[mask], self.worst[mask]
        sub.index = self.index[mask]
        return sub

    def proba_matrix(self, probas: pd.DataFrame) -> np.ndarray:
        keys = [(m, r) for m, _, r in self.index]
        return proba_matrix_for(keys, self.csfs, probas)

    def regret(self, P: np.ndarray, tau: float) -> tuple[float, float, float]:
        member = (P >= tau) & ~np.isnan(self.A)
        A_m = np.where(member, self.A, np.inf)
        best = A_m.min(axis=1)
        empty = ~np.isfinite(best)
        best = np.where(empty, self.worst, best)
        reg = np.clip(best - self.oracle, 0, None)
        return (float(reg.mean()), float(empty.mean()),
                float(member.sum(axis=1).mean()))


def main() -> None:
    lh = pd.read_parquet(
        OUT_ROOT / "track1" / "dataset" / "long_harmonized.parquet")
    lh = add_model_id(lh)
    cliques = pool_cliques_for(("VGG13",), lh)
    lh_std = lh.copy()
    for arch, sub in lh_std.groupby("architecture"):
        for c in NC_PRIMARY:
            lh_std.loc[sub.index, c] = (
                (sub[c] - sub[c].mean()) / (sub[c].std() + 1e-12))
    def label_pivot(cl: pd.DataFrame, columns=None) -> pd.DataFrame:
        wide = (cl.assign(label=cl["in_top_clique"].astype(int))
                .pivot_table(index=CELL_KEYS, columns="csf",
                             values="label", aggfunc="first")
                .reset_index().fillna(0))
        if columns is not None:
            for c in columns:
                if c not in wide.columns:
                    wide[c] = 0
            wide = wide[CELL_KEYS + list(columns)]
        return wide

    label_wide = label_pivot(cliques)
    csf_cols = [c for c in label_wide.columns if c not in CELL_KEYS]
    vgg_raw = lh[lh["architecture"] == "VGG13"]

    models = (lh_std[["model_id", "architecture", "paradigm", "source",
                      "run", "dropout", "reward"] + NC_PRIMARY]
              .drop_duplicates("model_id"))
    vgg = models[models["architecture"] == "VGG13"]

    def train_frame(mods: pd.DataFrame, labels=None) -> pd.DataFrame:
        rows = [{**m.to_dict(), "regime": r}
                for _, m in mods.iterrows()
                for r in ["near", "mid", "far", "all"]]
        return pd.DataFrame(rows).merge(
            label_wide if labels is None else labels, on=CELL_KEYS,
            how="inner")

    def test_frame(mods: pd.DataFrame) -> pd.DataFrame:
        rows = [{**m.to_dict(), "regime": r}
                for _, m in mods.iterrows() for r in REGIMES]
        return pd.DataFrame(rows)

    vgg_eval = RegretEval(lh[lh["architecture"] == "VGG13"][
        ["model_id", "eval_dataset", "regime", "csf", "augrc"]])

    lr_grid = [("LR", (("C", c),)) for c in np.logspace(-2, 2, 7)]
    rf_grid = [("RF", (("max_depth", d), ("max_features", f),
                       ("min_samples_leaf", l)))
               for d, l, f in itertools.product([None, 6], [1, 4],
                                                ["sqrt", 0.5])]

    def factory(kind: str, hp: dict):
        if kind == "LR":
            return lambda: LogisticRegression(
                penalty="l2", C=hp["C"], class_weight="balanced",
                max_iter=5000)
        return lambda: RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            random_state=0, n_jobs=-1, **hp)

    runs = sorted(vgg["run"].unique())
    fold_labels: dict[int, pd.DataFrame] = {}
    for held in runs:
        fc, _ = compute_track1_cliques(vgg_raw[vgg_raw["run"] != held])
        flw = label_pivot(fc, columns=csf_cols)
        merged_chk = flw.merge(label_wide, on=CELL_KEYS,
                               suffixes=("_f", "_c"))
        agree = np.mean([
            (merged_chk[f"{c}_f"] == merged_chk[f"{c}_c"]).mean()
            for c in csf_cols])
        fold_labels[held] = flw
        logger.info(f"fold {held}: per-fold VGG cliques computed "
                    f"({flw.shape[0]} cells; label agreement with 5-run "
                    f"consensus {agree:.1%})")

    val: dict[tuple, dict[float, list[tuple]]] = {}
    for kind, hpt in lr_grid + rf_grid:
        hp = dict(hpt)
        for held in runs:
            tr = train_frame(vgg[vgg["run"] != held], fold_labels[held])
            te_mods = vgg[vgg["run"] == held]
            te = test_frame(te_mods)
            probas = fit_probas(factory(kind, hp), tr, te, csf_cols)
            sub_eval = vgg_eval.subset(te_mods["model_id"])
            P = sub_eval.proba_matrix(probas)
            # F1 targets: the held-out models' cells under PER-FOLD labels
            lab = te.merge(fold_labels[held], on=CELL_KEYS, how="inner")
            pairs = list(zip(lab["model_id"], lab["regime"]))
            L = lab[csf_cols].to_numpy(float) > 0
            Pf = proba_matrix_for(pairs, csf_cols, probas)
            for tau in TAUS:
                r, _, sz = sub_eval.regret(P, tau)
                S = Pf >= tau
                inter = (S & L).sum(axis=1)
                denom = S.sum(axis=1) + L.sum(axis=1)
                f1 = float(np.mean(np.where(
                    denom > 0, 2 * inter / np.maximum(denom, 1), 1.0)))
                val.setdefault((kind, hpt), {}).setdefault(
                    tau, []).append((r, sz, f1))
        m05 = np.mean([v[0] for v in val[(kind, hpt)][0.50]])
        logger.info(f"val {kind} {hp}: tau=0.5 regret {m05:.3f}")

    best = {}
    for kind in ["LR", "RF"]:
        flat = {(hpt, tau): tuple(
                    float(np.mean([v[i] for v in vals])) for i in range(3))
                for (k, hpt), taus in val.items() if k == kind
                for tau, vals in taus.items()}
        hpt05 = min({k: v for k, v in flat.items() if k[1] == 0.50},
                    key=lambda k: flat[k][0])
        hptj = min(flat, key=lambda k: flat[k][0])
        within = {k: v for k, v in flat.items() if v[1] <= SIZE_BUDGET}
        hptb = min(within, key=lambda k: within[k][0]) if within else hpt05
        hptf = max(flat, key=lambda k: flat[k][2])
        best[kind] = {}
        for tag, key in [("tau05", hpt05), ("budget", hptb),
                         ("f1", hptf), ("free", hptj)]:
            r_, s_, f_ = flat[key]
            best[kind][tag] = (dict(key[0]), key[1], r_, s_, f_)
            logger.info(f"BEST {kind} {tag}: {dict(key[0])} tau={key[1]} "
                        f"(val regret {r_:.3f}, size {s_:.1f}, "
                        f"F1 {f_:.3f})")

    # final refits and transfer evaluation
    tr_full = train_frame(vgg)
    targets: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for arch in ["ResNet18", "ViT"]:
        rows = lh[lh["architecture"] == arch][
            ["model_id", "eval_dataset", "source", "regime", "csf",
             "augrc"]]
        targets[arch] = (test_frame(models[models["architecture"] == arch]),
                         rows)
    pool_m = pd.read_parquet(OUT_ROOT / "pool_a" / "models_pool_a.parquet")
    for enc, sub in pool_m.groupby("paradigm"):
        for c in NC_PRIMARY:
            pool_m.loc[sub.index, c] = (
                (sub[c] - sub[c].mean()) / (sub[c].std() + 1e-12))
    pool_m["model_id"] = (pool_m["paradigm"] + "|" + pool_m["source"] + "|"
                          + pool_m["run"].astype(str))
    ldf = pd.read_parquet(OUT_ROOT / "pool_a" / "long_pool_a.parquet")
    ldf["model_id"] = (ldf["paradigm"] + "|" + ldf["source"] + "|"
                       + ldf["run"].astype(str))
    targets["SSL"] = (test_frame(pool_m),
                      ldf[["model_id", "eval_dataset", "source", "regime",
                           "csf", "augrc"]])

    proba_cache: dict[tuple, dict[str, pd.DataFrame]] = {}

    def target_probas(kind: str, hp: dict, paper_cv=False):
        key = (kind, tuple(sorted(hp.items())), paper_cv)
        if key not in proba_cache:
            proba_cache[key] = {
                tgt: fit_probas(None if paper_cv else factory(kind, hp),
                                tr_full, te, csf_cols, paper_cv=paper_cv)
                for tgt, (te, _) in targets.items()}
        return proba_cache[key]

    V1_HP = {"LR": {"C": 1.0},
             "RF": {"max_depth": None, "max_features": 0.5,
                    "min_samples_leaf": 4}}
    arms = [("paper LR", ("LR", {}, 0.50, True)),
            ("LR v1 tau=.5", ("LR", V1_HP["LR"], 0.50, False)),
            ("RF v1 tau=.5", ("RF", V1_HP["RF"], 0.50, False))]
    for kind in ["LR", "RF"]:
        for tag in ["tau05", "budget", "f1", "free"]:
            hp, tau = best[kind][tag][0], best[kind][tag][1]
            if tag == "tau05" and hp == V1_HP[kind]:
                continue  # identical to the gate arm
            label = f"{kind} {'tau=.5*' if tag == 'tau05' else tag + '*'}"
            arms.append((label, (kind, hp, tau, False)))

    results, mech = {}, {}
    tgt_evals = {tgt: RegretEval(rows[["model_id", "eval_dataset",
                                       "regime", "csf", "augrc"]])
                 for tgt, (_, rows) in targets.items()}
    for arm_label, (kind, hp, tau, paper_cv) in arms:
        probas = target_probas(kind, hp, paper_cv)
        for tgt, (te, rows) in targets.items():
            pr = probas[tgt]
            csf_prob_cols = [c for c in pr.columns
                             if c not in ("model_id", "regime")]
            melted = pr.melt(id_vars=["model_id", "regime"],
                             value_vars=csf_prob_cols, var_name="csf",
                             value_name="p")
            sl = melted[melted["p"] >= tau][["model_id", "regime", "csf"]]
            results[(arm_label, tgt)] = evaluate(rows, sl)
            P = tgt_evals[tgt].proba_matrix(pr)
            _, empty_rate, size = tgt_evals[tgt].regret(P, tau)
            mech[(arm_label, tgt)] = (empty_rate, size)
            pooled = " ".join(
                f"{r}={results[(arm_label, tgt)][('all', r)]['predictor']}"
                for r in REGIMES)
            logger.info(f"{arm_label} -> {tgt}: {pooled} "
                        f"(empty {empty_rate:.1%}, size {size:.1f})")

    bad = []
    for tgt, exp in EXPECTED_REF.items():
        got = tuple(results[("paper LR", tgt)][("all", r)]["predictor"]
                    for r in REGIMES)
        if any(abs(g - e) > 0.1 for g, e in zip(got, exp)):
            bad.append(("ref", tgt, exp, got))
    for (kind, tgt), exp in EXPECTED_TAU05.items():
        got = tuple(results[(f"{kind} v1 tau=.5", tgt)][("all", r)]
                    ["predictor"] for r in REGIMES)
        if any(abs(g - e) > 0.05 for g, e in zip(got, exp)):
            bad.append((kind, tgt, exp, got))
    if bad:
        for b in bad:
            logger.error(f"gate mismatch: {b}")
        raise SystemExit("Gates FAILED; report not written.")
    logger.info("Gates PASSED (reference + first-version arms)")

    # hindsight-best tau per target (diagnostic only; regret is weakly
    # monotone decreasing in tau, so the unconstrained optimum is the grid
    # edge; the budget and F1 arms are the meaningful operating points)
    diag_lines = []
    for kind in ["LR", "RF"]:
        hp = best[kind]["f1"][0]
        probas = target_probas(kind, hp)
        for tgt in targets:
            P = tgt_evals[tgt].proba_matrix(probas[tgt])
            curve = {tau: tgt_evals[tgt].regret(P, tau)[0] for tau in TAUS}
            tstar = min(curve, key=curve.get)
            diag_lines.append(
                f"- DIAGNOSTIC hindsight tau ({kind} {hp} on {tgt}): "
                f"tau*={tstar} regret {curve[tstar]:.2f} (deployed F1 "
                f"tau={best[kind]['f1'][1]}: "
                f"{curve[best[kind]['f1'][1]]:.2f})")

    legend = "; ".join(
        f"{kind} {tag} = {best[kind][tag][0]} tau={best[kind][tag][1]} "
        f"(val regret {best[kind][tag][2]:.3f}, size "
        f"{best[kind][tag][3]:.1f}, F1 {best[kind][tag][4]:.3f})"
        for kind in ["LR", "RF"]
        for tag in ["tau05", "budget", "f1", "free"])
    lines = [
        "# Predictor-model ablation: tuned LR vs tuned RF, with threshold "
        "tuning\n",
        "\n**Source:** `nc_csf_predictivity/evaluation/"
        "predictor_model_ablation.py`. Grouped tuning (GroupKFold over the "
        "five VGG runs). Selection criteria per family: tau=0.5 regret "
        "(gated first-version arms); size-budgeted regret (mean held-out "
        f"size <= {SIZE_BUDGET}, the paper's operating point; regret alone "
        "is weakly monotone decreasing in tau, so unconstrained regret "
        "tuning is ill-posed); set-F1 against the held-out cells' top "
        "cliques (classification-native and parameter-free: empty lists "
        "score recall 0, the full roster collapses precision; caveat: "
        "cell labels are 5-run consensus, so held-out runs contribute to "
        "their own labels); and unconstrained regret ('free'), kept only "
        "to demonstrate the degeneracy. Transfers under the first-reply "
        "conventions. STRICT LABELS: validation-loop training labels "
        "and F1 targets recomputed per fold from the four in-fold runs "
        "only; final refits use the published 5-run cliques; starred "
        "arms are strict-label selections, v1 arms pin the first-version "
        "hyperparameters as gates.\n\n"
        f"Selected: {legend}.\n\n",
        "| Target | Regime | Best fixed CSF | "
        + " | ".join(a for a, _ in arms) + " |\n|"
        + "---|" * (3 + len(arms)) + "\n"]
    for tgt in ["ResNet18", "ViT", "SSL"]:
        for regime in REGIMES:
            bf = results[(arms[0][0], tgt)][("all", regime)]
            cells = " | ".join(
                f"{results[(a, tgt)][('all', regime)]['predictor']:.2f}"
                for a, _ in arms)
            lines.append(f"| {tgt} | {regime} | {bf['best_fixed']:.2f} "
                         f"({bf['best_fixed_name']}) | {cells} |\n")
    lines.append("\nAbstention statistics per arm (empty-row rate / mean "
                 "shortlist size):\n\n")
    for arm_label, _ in arms:
        stats = "; ".join(
            f"{tgt} {mech[(arm_label, tgt)][0]:.1%}/"
            f"{mech[(arm_label, tgt)][1]:.1f}" for tgt in targets)
        lines.append(f"- {arm_label}: {stats}\n")
    lines.append("\n" + "\n".join(diag_lines) + "\n")
    out = OUT_ROOT / "35_predictor_model_ablation.md"
    out.write_text("".join(lines))
    logger.info(f"Wrote {out}")
    print("".join(lines))


if __name__ == "__main__":
    main()

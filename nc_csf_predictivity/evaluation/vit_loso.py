"""Leave-one-ViT-source-out test of the coverage explanation (camera-ready item 10.2).

App K finds that the VGG13-trained NC-only heads do not target on the 40
fine-tuned ViTs (a size-matched random shortlist has lower mean regret). The
proposed explanation: the heads can only recommend CSFs that the VGG13
cliques reward. This test trains heads that see ViT labels and scores them on
a ViT source they have not seen (4 folds, 10 test models each).

Arms (regime-free, marginal labels, 22-CSF roster; ViT roster 21 CSFs):
  A  VGG13 heads (the App K predictor, gate GK).
  B  ViT-only NC heads, trained on the 30 ViT models of the other sources.
  C  VGG13 + ViT NC heads, pooled training, no family indicator.
  D  ViT intercept-only: a fixed shortlist of the CSFs whose training-ViT
     label rate is >= 0.5 (the family base rates, without NC).
  D* post hoc, see below.
  B' sensitivity of B: ViT NC z-scored with the training-fold ViT models only
     (the other arms z-score each architecture on its full pool).
A CSF without a head (constant labels or fewer than 5 of a class) is
shortlisted iff its training label rate is >= 0.5. Every arm gets its own
size-matched random shortlist.

Decision rule (fixed before the run), pooled cell, one-sided model-clustered
sign-flip mean test, Holm over the two primary comparisons:
  primary: B < D and B < random(B).
  Outcome 1 (NC targets within ViT): both primary comparisons significant.
  Outcome 2 (family base rates, not NC): B and D both below A (mean), and
            B not significantly below D.
  Outcome 3 (nothing fixes it): neither B nor D below A, or neither below
            its random set.
Power caveat: each fold trains on 6 (source, dropout) ViT configurations.

Post-hoc addition (after the first run): arm D was degenerate, because no
CSF reaches a ViT label rate of 0.5 (the ViT top cliques are small), so its
shortlist was empty in about half the folds. Arm D* is the fair base-rate
control: for each test model, the k CSFs with the highest training-ViT label
rate, k = that model's arm-B shortlist size. Per-source cells are added for
the same reason. Neither changes the pre-stated outcome, which B and C
against A decide.

Gates (abort before any LOSO number is computed):
  GH  the VGG13 head fit reproduces the stored 22-CSF ResNet18 predictions.
  GL  the family-generic per-config clique code reproduces the stored VGG13
      labels (perconfig_cliques_bo.parquet) exactly.
  GV  ViT AUGRC rows complete and equal to the harmonized table.
  GK  arm A reproduces the App K pooled mean regret.

Outputs: outputs/vit_loso/{labels_vit,cells,tests}.csv, outputs/53_vit_loso.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

EVAL_DIR = Path(__file__).resolve().parent
CODE_DIR = EVAL_DIR.parents[1]
for path in (EVAL_DIR.parent / "ablations", EVAL_DIR, CODE_DIR / "x9_imagenet"):
    sys.path.insert(0, str(path))
from benchmark_newcsf_cliques import (  # noqa: E402
    DISPLAY,
    FEATURE_SIDE,
    HEAD_SIDE,
    NEW_CSFS,
    load_base,
    perconfig_cliques,
)
from calibration_features_clique import (  # noqa: E402
    DEFAULT_OUT_ROOT,
    NC_PRIMARY,
    add_model_id,
    standardize_per_architecture,
)
from calibration_regime_free import (  # noqa: E402
    build_test_nr,
    build_train_nr,
    fit_heads,
)
from newcsf_heads import AGGREGATE, LABEL_CSF_NAMES, SOURCE_NAMES, labels_22  # noqa: E402
from phase3_transfer import CONFIG, VARIANT, fitted_heads, per_row, test_pair  # noqa: E402
from regimefree_wilcoxon import N_FLIPS, SEED, load_parquet  # noqa: E402
from vit_transfer_22 import load_vit  # noqa: E402

LABEL_KEYS = ["paradigm", "source", "dropout", "reward", "regime", "csf"]
REGIMES = ["near", "mid", "far"]
VIT_ROSTER: list[str] = []
APP_K_POOLED = 3.0127
TOL = 1e-3


def family_labels(base: pd.DataFrame, family: str, roster: list[str]) -> pd.DataFrame:
    """Per-config top-clique labels of one family, named like the harmonized table."""
    p = perconfig_cliques(base, roster, family=family)
    if (p["status"] != "ok").any():
        raise RuntimeError(f"{family}: non-ok clique cells {p['status'].unique()}")
    p = p.assign(paradigm=p["paradigm"].replace({"vit": "modelvit"}),
                 source=p["source"].replace(SOURCE_NAMES),
                 csf=p["csf"].replace(LABEL_CSF_NAMES),
                 label=p["in_top_clique"].astype(int))
    return p[LABEL_KEYS + ["label"]]


def gate_labels(base: pd.DataFrame, roster: list[str], out_root: Path) -> None:
    """GL: the family-generic clique code reproduces the stored VGG13 labels.

    Raises:
        RuntimeError: On any label difference.
    """
    rebuilt = family_labels(base, "VGG13", roster)
    stored = labels_22(out_root)
    both = stored.merge(rebuilt, on=LABEL_KEYS, how="outer", suffixes=("", "_new"))
    bad = int((both["label"] != both["label_new"]).sum())
    if bad:
        raise RuntimeError(f"GL failed: {bad} of {len(both)} labels differ")
    print(f"GL: {len(both):,} VGG13 labels reproduced")


def heads_to_probs(heads: dict, y_train: pd.DataFrame, x_test: pd.DataFrame,
                   test_ids: pd.Series) -> pd.DataFrame:
    """Probabilities per (test model, CSF); headless CSFs get their training rate."""
    rates = y_train.mean()
    cols = [pd.DataFrame({"model_id": test_ids.values, "csf": csf,
                          "p": heads[csf].predict_proba(x_test)[:, 1] if csf in heads
                          else float(rates[csf])})
            for csf in y_train.columns]
    return pd.concat(cols, ignore_index=True)


def zscore_with(long_df: pd.DataFrame, ref_ids: list[str]) -> pd.DataFrame:
    """ViT NC z-scored with the mean and std of the ViT models in ref_ids."""
    out = standardize_per_architecture(long_df, NC_PRIMARY)
    vit = long_df["architecture"] == "ViT"
    ref = long_df[vit & long_df["model_id"].isin(ref_ids)].drop_duplicates("model_id")
    out.loc[vit, NC_PRIMARY] = ((long_df.loc[vit, NC_PRIMARY] - ref[NC_PRIMARY].mean())
                                / ref[NC_PRIMARY].std())
    return out


def fold_probs(long_raw: pd.DataFrame, long_z: pd.DataFrame, labels: pd.DataFrame,
               vgg_ids: list[str], train_vit: list[str], test_vit: list[str]
               ) -> dict[str, pd.DataFrame]:
    """Probabilities of arms B, C, D, and B' for one held-out ViT source."""
    out = {}
    x_te, meta = build_test_nr(long_z, test_vit, CONFIG)
    for arm, ids, archs in (("B", train_vit, ("ViT",)),
                            ("C", vgg_ids + train_vit, ("VGG13", "ViT"))):
        x_tr, y_tr, cols = build_train_nr(long_z, labels, ids, CONFIG, VARIANT,
                                          architectures=archs)
        out[arm] = heads_to_probs(fit_heads(x_tr, y_tr, cols, CONFIG), y_tr, x_te,
                                  meta["model_id"])
        if arm == "B":
            rates = y_tr.mean()
            out["D"] = pd.DataFrame([{"model_id": m, "csf": c, "p": float(rates[c])}
                                     for m in meta["model_id"] for c in cols])
            size = out["B"].query("csf in @VIT_ROSTER and p >= 0.5").groupby(
                "model_id").size().reindex(meta["model_id"], fill_value=0)
            ranked = rates[rates.index.isin(VIT_ROSTER)].sort_values(
                ascending=False, kind="stable").index.tolist()
            out["D*"] = pd.DataFrame([{"model_id": m, "csf": c,
                                       "p": float(c in ranked[:size[m]])}
                                      for m in meta["model_id"] for c in cols])
    long_fold = zscore_with(long_raw, train_vit)
    x_tr, y_tr, cols = build_train_nr(long_fold, labels, train_vit, CONFIG, VARIANT,
                                      architectures=("ViT",))
    x_te_f, meta_f = build_test_nr(long_fold, test_vit, CONFIG)
    out["B'"] = heads_to_probs(fit_heads(x_tr, y_tr, cols, CONFIG), y_tr, x_te_f,
                               meta_f["model_id"])
    return out


def summarize(pr: pd.DataFrame, roster_size: int) -> pd.DataFrame:
    """Mean regret, set size, oracle-in-set, and random set per arm and cell."""
    recs = []
    for (arm, cell), g in pd.concat(
            [pr.assign(cell="pooled"), pr.assign(cell=pr["regime"]),
             pr.assign(cell=pr["family"] + " (held out)")]).groupby(
                ["arm", "cell"], sort=False):
        recs.append({"arm": arm, "cell": cell, "rows": len(g),
                     "regret": g["regret"].mean(), "set_size": g["set_size"].mean(),
                     "random_set": g["random"].mean(),
                     "oracle_in_set": (g["regret"] == 0).mean(),
                     "random_expected": (g["set_size"] / roster_size).mean()})
    return pd.DataFrame(recs)


def compare(pr: pd.DataFrame, pairs: list[tuple[str, str]], flips: np.ndarray
            ) -> pd.DataFrame:
    """One-sided tests (first arm better) per comparison and cell."""
    wide = pr.pivot_table(index=["model_id", "eval_dataset", "regime"], columns="arm",
                          values="regret").reset_index()
    rand = pr.pivot_table(index=["model_id", "eval_dataset", "regime"], columns="arm",
                          values="random").add_prefix("random_").reset_index()
    wide = wide.merge(rand, on=["model_id", "eval_dataset", "regime"])
    recs = []
    for arm, ref in pairs:
        for cell in ["pooled"] + REGIMES:
            sub = wide if cell == "pooled" else wide[wide["regime"] == cell]
            p_wil, p_mean = test_pair(sub.assign(regret=sub[arm]), ref, flips)
            recs.append({"comparison": f"{arm} < {ref}", "cell": cell,
                         "diff": (sub[arm] - sub[ref]).mean(),
                         "p_wilcoxon": p_wil, "p_mean": p_mean})
    return pd.DataFrame(recs)


def main() -> None:
    """Gates, ViT labels, four LOSO folds, tests, and the report."""
    out_root = Path(DEFAULT_OUT_ROOT)
    heads_a = fitted_heads(out_root)
    base, _ = load_base(AGGREGATE, CODE_DIR / "clip_scores")
    roster22 = [DISPLAY.get(c, c) for c in HEAD_SIDE + FEATURE_SIDE] + NEW_CSFS
    gate_labels(base, roster22, out_root)
    labels_vit = family_labels(base, "ViT", [c for c in roster22 if c != "Confidence"])
    labels = pd.concat([labels_22(out_root), labels_vit], ignore_index=True)

    long_raw = add_model_id(load_parquet(out_root / "track1" / "dataset"
                                         / "long_harmonized.parquet"))
    long_z = standardize_per_architecture(long_raw, NC_PRIMARY)
    rows = load_vit(long_raw)
    roster = sorted(rows["csf"].unique())
    VIT_ROSTER[:] = roster
    regime = (long_raw.loc[long_raw["architecture"] == "ViT",
                           ["source", "eval_dataset", "regime"]].drop_duplicates()
              .rename(columns={"source": "family"}))
    models = long_raw.drop_duplicates("model_id")
    vgg_ids = models.loc[models["architecture"] == "VGG13", "model_id"].tolist()
    vit = models[models["architecture"] == "ViT"]

    x_all, meta_all = build_test_nr(long_z, vit["model_id"].tolist(), CONFIG)
    probs = {"A": pd.concat([pd.DataFrame({"model_id": meta_all["model_id"].values,
                                           "csf": csf, "p": pipe.predict_proba(x_all)[:, 1]})
                             for csf, pipe in heads_a.items()], ignore_index=True)}
    fold_parts: dict[str, list[pd.DataFrame]] = {}
    for source in sorted(vit["source"].unique()):
        test_vit = vit.loc[vit["source"] == source, "model_id"].tolist()
        train_vit = vit.loc[vit["source"] != source, "model_id"].tolist()
        for arm, p in fold_probs(long_raw, long_z, labels, vgg_ids, train_vit,
                                 test_vit).items():
            fold_parts.setdefault(arm, []).append(p)
    probs.update({arm: pd.concat(parts, ignore_index=True)
                  for arm, parts in fold_parts.items()})

    pr = pd.concat([per_row(rows, p, roster).assign(arm=arm) for arm, p in probs.items()],
                   ignore_index=True).merge(regime, on=["family", "eval_dataset"])
    pooled_a = pr.loc[pr["arm"] == "A", "regret"].mean()
    if abs(pooled_a - APP_K_POOLED) > TOL:
        raise RuntimeError(f"GK failed: arm A pooled {pooled_a:.4f} vs {APP_K_POOLED}")
    print(f"GK: arm A pooled regret {pooled_a:.4f} reproduces App K")

    flips = np.random.default_rng(SEED).choice(np.array([-1.0, 1.0]),
                                               size=(N_FLIPS, vit["model_id"].nunique()))
    primary = compare(pr, [("B", "D"), ("B", "random_B")], flips)
    pooled = primary["cell"] == "pooled"
    primary["p_mean_holm"] = np.nan
    primary.loc[pooled, "p_mean_holm"] = multipletests(
        primary.loc[pooled, "p_mean"], method="holm")[1]
    secondary = compare(pr, [("B", "A"), ("D", "A"), ("C", "A"), ("D", "random_D"),
                             ("B", "D*"), ("D*", "A"), ("D*", "random_D*"),
                             ("C", "random_C"), ("B'", "random_B'"), ("B'", "D")], flips)
    cells = summarize(pr, len(roster))
    shortlist = pd.concat(
        [p.assign(arm=arm, inc=p["p"] >= 0.5) for arm, p in probs.items()]
    ).query("csf in @roster").groupby(["arm", "csf"])["inc"].mean().unstack(0).round(2)
    vit_rate = (labels_vit[labels_vit["regime"].isin(REGIMES)].groupby("csf")["label"]
                .mean().round(2).sort_values(ascending=False))

    csv_dir = out_root / "vit_loso"
    csv_dir.mkdir(parents=True, exist_ok=True)
    labels_vit.to_csv(csv_dir / "labels_vit.csv", index=False)
    cells.to_csv(csv_dir / "cells.csv", index=False)
    pd.concat([primary.assign(kind="primary"), secondary.assign(kind="secondary")]
              ).to_csv(csv_dir / "tests.csv", index=False)
    lines = ["# Leave-one-ViT-source-out test of the coverage explanation\n\n",
             "**Source:** `nc_csf_predictivity/evaluation/vit_loso.py` (decision rule "
             "in the docstring, fixed before the run). Mean imputed set-regret "
             "(AUGRC x1000) over the 40 ViTs, 4 folds; tests one-sided, model-"
             "clustered sign-flip (mean) and Wilcoxon.\n\n",
             "Gates GH, GL, GV, GK passed.\n\n",
             "## Cells\n\n```\n", cells.round(4).to_string(index=False), "\n```\n\n",
             "## Primary tests (Holm over the two pooled comparisons)\n\n```\n",
             primary.round(4).to_string(index=False), "\n```\n\n",
             "## Secondary tests (raw p)\n\n```\n",
             secondary.round(4).to_string(index=False), "\n```\n\n",
             "## ViT label rate per CSF (near/mid/far cells)\n\n```\n",
             vit_rate.to_string(), "\n```\n\n",
             "## Shortlist inclusion rate per arm\n\n```\n", shortlist.to_string(),
             "\n```\n"]
    (out_root / "53_vit_loso.md").write_text("".join(lines))
    print(f"wrote {out_root / '53_vit_loso.md'}")


if __name__ == "__main__":
    main()

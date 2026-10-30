"""Phase 3 of documentation/imagenet_scale_plan.md: ImageNet transfer of the NC-only predictor.

The headline predictor of the paper (regime-free NC-only heads, 22-CSF
per-config labels, trained on the 280 VGG13 models) is applied without
re-training to the 47 timm ImageNet-1k checkpoints of the x9 sweep. NC metrics
are z-scored per model family on the target side (the per-architecture
convention). Evaluation per (model, OOD set) row over the 21-CSF ImageNet
roster:

  - mean imputed set-regret against the per-row oracle;
  - the best fixed CSF of each cell, chosen with hindsight on the same rows
    (the SSL-table convention: no prior ImageNet clique knowledge exists),
    and Always-CTM, the strongest fixed CSF on the CIFAR benchmark;
  - the size-matched random-set control of regimefree_wilcoxon.py;
  - one-sided Wilcoxon and model-clustered sign-flip mean tests, Holm over
    the five main cells (pooled, near and far under each grouping).

Cells: pooled over the 8 OOD sets; near/far from the paper's CLIP protocol
(ViT-B/32 distances, k selected by the Appendix D rule = 2 on ImageNet,
clustered by clip_clustering_all_backbones.cluster_one); near/far from
OpenOOD v1.5 (5 of the 8 sets); per family (exploratory, raw p-values).
Secondary readouts: KPCA-free sensitivity (KPCA RecError is the one
fit-draw-unstable CSF, gate G3), oracle and shortlist frequencies, and the NC
descriptor table across pools.

Gates (abort before any ImageNet number is computed):
  GH  the head fit reproduces the stored 22-CSF ResNet18 predictions of
      evaluation/newcsf_heads.py exactly.
  GD  every model has all 21 CSFs on all 8 OOD sets.
  GR  the k = 2 grouping has no singleton cluster (the Appendix D rule).

Inputs: code/x9_outputs/*_{rows,model}.parquet, code/clip_scores_imagenet/.
Outputs: nc_csf_predictivity/outputs/x9_phase3/*.csv and
         nc_csf_predictivity/outputs/50_imagenet_transfer.md
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

CODE_DIR = pathlib.Path(__file__).resolve().parents[1]
for path in (CODE_DIR, CODE_DIR / "nc_csf_predictivity" / "ablations",
             CODE_DIR / "nc_csf_predictivity" / "evaluation"):
    sys.path.insert(0, str(path))
from calibration_features_clique import (  # noqa: E402
    DEFAULT_OUT_ROOT,
    NC_PRIMARY,
    add_model_id,
    standardize_per_architecture,
)
from calibration_regime_free import (  # noqa: E402
    build_test_nr,
    build_train_nr,
    feature_columns_nr,
    fit_heads,
)
from clip_clustering_all_backbones import cluster_one  # noqa: E402
from newcsf_heads import labels_22  # noqa: E402
from regimefree_wilcoxon import (  # noqa: E402
    N_FLIPS,
    SEED,
    load_parquet,
    one_sided_p,
    random_set_regret,
    signflip_mean_p,
)

CONFIG, VARIANT = "none_nr", "marginal"
X9_DIR = CODE_DIR / "x9_outputs"
CLIP_JSON = CODE_DIR / "clip_scores_imagenet" / "clip_proximity_imagenet_ViT-B-32.json"
OPENOOD = {"ssb hard": "near", "ninco": "near", "inaturalist": "far",
           "texture": "far", "openimage o": "far"}
KPCA = "KPCA RecError global"
TOL = 1e-12


def load_imagenet() -> tuple[pd.DataFrame, pd.DataFrame]:
    """OOD AUGRC rows and per-model NC metrics of the x9 sweep."""
    rows = pd.concat(map(pd.read_parquet, sorted(X9_DIR.glob("*_rows.parquet"))))
    rows = rows[rows["eval_dataset"] != "iid_test"]
    rows = rows.assign(eval_dataset=rows["eval_dataset"].str.replace("_", " "),
                       model_id=rows["tag"]).rename(columns={"augrc": "raw_augrc"})
    models = pd.concat(map(pd.read_parquet, sorted(X9_DIR.glob("*_model.parquet"))))
    return rows[["model_id", "family", "eval_dataset", "csf", "raw_augrc"]], models


def clip_regimes() -> dict[str, str]:
    """Near/far per OOD set from the paper's clustering code at k = 2.

    Raises:
        RuntimeError: If a cluster holds a single set (gate GR).
    """
    distances, _ = cluster_one(str(CLIP_JSON), "imagenet", n_clusters=2)
    group = distances["group"]
    group = group.iloc[:, 0] if isinstance(group, pd.DataFrame) else group
    group = group.drop("test")
    if group.value_counts().min() < 2:
        raise RuntimeError(f"GR failed: singleton cluster in {group.to_dict()}")
    return {name: {"1": "near", "2": "far"}[str(g)] for name, g in group.items()}


def fitted_heads(out_root: pathlib.Path) -> dict:
    """NC-only heads on the VGG13 pool, checked against the stored ResNet18 preds.

    Raises:
        RuntimeError: If the predictions differ from newcsf_heads.py (gate GH).
    """
    long_df = standardize_per_architecture(add_model_id(load_parquet(
        out_root / "track1" / "dataset" / "long_harmonized.parquet")), NC_PRIMARY)
    split = load_parquet(out_root / "splits" / "xarch.parquet")
    train = split.loc[split["role"] == "train", "model_id"].tolist()
    test = split.loc[split["role"] == "test", "model_id"].tolist()
    heads = fit_heads(*build_train_nr(long_df, labels_22(out_root), train,
                                      CONFIG, VARIANT), CONFIG)
    x_te, meta = build_test_nr(long_df, test, CONFIG)
    rebuilt = pd.concat([pd.DataFrame({"model_id": meta["model_id"].values, "csf": csf,
                                       "p": pipe.predict_proba(x_te)[:, 1]})
                         for csf, pipe in heads.items()])
    stored = load_parquet(out_root / "newcsf_heads" / "track1" / "xarch"
                          / f"{CONFIG}_{VARIANT}" / "preds.parquet")
    stored = stored[stored["regime"] == "mid"]
    both = stored.merge(rebuilt, on=["model_id", "csf"], how="outer", indicator=True)
    dev = float((both["p_competitive"] - both["p"]).abs().max())
    if (both["_merge"] != "both").any() or dev > TOL:
        raise RuntimeError(f"GH failed: unmatched {int((both['_merge'] != 'both').sum())}, "
                           f"max |dp| {dev:.3g}")
    print(f"GH: {len(heads)} heads, {len(both)} ResNet18 predictions reproduced, "
          f"max |dp| {dev:.1e}")
    return heads


def shortlists(heads: dict, models: pd.DataFrame) -> pd.DataFrame:
    """Predicted probabilities per (model, CSF), NC z-scored within each family."""
    z = models.copy()
    z[NC_PRIMARY] = models.groupby("family")[NC_PRIMARY].transform(
        lambda c: (c - c.mean()) / c.std())
    x = z[feature_columns_nr(CONFIG)]
    return pd.concat([pd.DataFrame({"model_id": z["tag"].values, "csf": csf,
                                    "p": pipe.predict_proba(x)[:, 1]})
                      for csf, pipe in heads.items()], ignore_index=True)


def per_row(rows: pd.DataFrame, probs: pd.DataFrame, roster: list[str]) -> pd.DataFrame:
    """Per (model, OOD set): oracle, predictor regret, fixed-CSF regrets, random set."""
    rows = rows[rows["csf"].isin(roster)]
    wide = rows.pivot_table(index=["model_id", "family", "eval_dataset"],
                            columns="csf", values="raw_augrc")
    members = probs[(probs["p"] >= 0.5) & probs["csf"].isin(roster)]
    members = members.groupby("model_id")["csf"].apply(list)
    oracle, worst = wide.min(axis=1), wide.max(axis=1)
    picked = [wide.loc[idx, members.get(idx[0], [])].min() if members.get(idx[0])
              else worst[idx] for idx in wide.index]
    out = pd.DataFrame({"o_augrc": oracle, "regret": np.array(picked) - oracle,
                        "set_size": [len(members.get(i[0], [])) for i in wide.index]})
    out = pd.concat([out, wide.sub(oracle, axis=0).add_prefix("fixed::")], axis=1)
    out = out.reset_index().assign(side="all")
    out["random"] = random_set_regret(out, rows).to_numpy()
    return out


def test_pair(cell: pd.DataFrame, baseline: str, flips: np.ndarray) -> tuple[float, float]:
    """Predictor better than `baseline`: Wilcoxon and clustered sign-flip mean p."""
    paired = cell.assign(diff=cell["regret"] - cell[baseline])
    return (one_sided_p(paired["diff"].to_numpy(), "less"),
            signflip_mean_p(paired, flips)[0])


def cell_table(rows: pd.DataFrame, cells: dict[str, pd.Series], flips: np.ndarray
               ) -> pd.DataFrame:
    """Means, hindsight best fixed CSF, Always-CTM, random set, and tests per cell."""
    fixed = [c for c in rows.columns if c.startswith("fixed::")]
    recs = []
    for name, mask in cells.items():
        cell = rows[mask]
        means = cell[fixed].mean()
        best = means.idxmin()
        p_best = test_pair(cell, best, flips)
        p_rand = test_pair(cell, "random", flips)
        recs.append({"cell": name, "rows": len(cell), "models": cell["model_id"].nunique(),
                     "nc_only": cell["regret"].mean(), "set_size": cell["set_size"].mean(),
                     "best_fixed": best.removeprefix("fixed::"),
                     "best_fixed_regret": means[best],
                     "always_ctm": cell.get("fixed::CTM", pd.Series(np.nan)).mean(),
                     "random_set": cell["random"].mean(),
                     "p_wil_best": p_best[0], "p_mean_best": p_best[1],
                     "p_wil_random": p_rand[0], "p_mean_random": p_rand[1]})
    return pd.DataFrame(recs)


def holm_main(table: pd.DataFrame, n_main: int) -> pd.DataFrame:
    """Holm over the first n_main cells for every p-value column."""
    table = table.copy()
    for col in [c for c in table.columns if c.startswith("p_")]:
        table[f"{col}_holm"] = np.nan
        table.loc[:n_main - 1, f"{col}_holm"] = multipletests(
            table.loc[:n_main - 1, col], method="holm")[1]
    return table


def nc_ranges(out_root: pathlib.Path, models: pd.DataFrame) -> pd.DataFrame:
    """Median [min, max] of scale-free NC metrics per pool and ImageNet family."""
    metrics = ["self_duality", "max_equiangular_wc", "equiangular_wc", "equinorm_wc"]
    long_df = add_model_id(load_parquet(out_root / "track1" / "dataset"
                                        / "long_harmonized.parquet"))
    pools = [(arch, long_df[long_df["architecture"] == arch].drop_duplicates("model_id"))
             for arch in ("VGG13", "ResNet18", "ViT")]
    pools.append(("SSL probes", load_parquet(out_root / "pool_a"
                                             / "models_pool_a_harmonized.parquet")))
    pools += [(f"ImageNet {fam}", sub) for fam, sub in models.groupby("family")]
    fmt = lambda s: f"{s.median():.3f} [{s.min():.3f}, {s.max():.3f}]"  # noqa: E731
    return pd.DataFrame([{"pool": name, "models": len(df),
                          **{m: fmt(df[m]) for m in metrics}} for name, df in pools])


def main() -> None:
    """Gates, ImageNet shortlists, per-cell evaluation, and the report."""
    out_root = pathlib.Path(DEFAULT_OUT_ROOT)
    heads = fitted_heads(out_root)
    rows, models = load_imagenet()
    roster = sorted(rows["csf"].unique())
    counts = rows.groupby("model_id").size()
    if models["tag"].nunique() != 47 or len(roster) != 21 or (counts != 21 * 8).any():
        raise RuntimeError(f"GD failed: {models['tag'].nunique()} models, "
                           f"{len(roster)} CSFs, rows per model {counts.unique()}")
    print(f"GD: 47 models x 21 CSFs x 8 OOD sets complete")
    regimes = clip_regimes()
    print(f"GR: k = 2 regimes {regimes}")

    probs = shortlists(heads, models)
    flips = np.random.default_rng(SEED).choice(np.array([-1.0, 1.0]),
                                               size=(N_FLIPS, models["tag"].nunique()))
    tables = {}
    for label, csfs in (("21 CSFs", roster), ("20 CSFs, no KPCA", [c for c in roster if c != KPCA])):
        pr = per_row(rows, probs, csfs)
        clip_reg, oood_reg = pr["eval_dataset"].map(regimes), pr["eval_dataset"].map(OPENOOD)
        cells = {"pooled (8 sets)": pd.Series(True, index=pr.index),
                 "CLIP near": clip_reg == "near", "CLIP far": clip_reg == "far",
                 "OpenOOD near": oood_reg == "near", "OpenOOD far": oood_reg == "far"}
        n_main = len(cells)
        for fam in sorted(pr["family"].unique()):
            cells[f"{fam} pooled"] = pr["family"] == fam
            for reg in ("near", "far"):
                cells[f"{fam} CLIP {reg}"] = (pr["family"] == fam) & (clip_reg == reg)
        tables[label] = holm_main(cell_table(pr, cells, flips), n_main)
        if label == "21 CSFs":
            per_row_main = pr

    oracle_freq = (per_row_main.set_index(["model_id", "eval_dataset"])
                   .filter(like="fixed::").idxmin(axis=1).str.removeprefix("fixed::")
                   .value_counts(normalize=True).round(3))
    inclusion = (probs[probs["csf"].isin(roster)].assign(inc=lambda d: d["p"] >= 0.5)
                 .groupby("csf")["inc"].mean().round(2).sort_values(ascending=False))
    ranges = nc_ranges(out_root, models)

    csv_dir = out_root / "x9_phase3"
    csv_dir.mkdir(parents=True, exist_ok=True)
    per_row_main.to_csv(csv_dir / "per_row.csv", index=False)
    probs.to_csv(csv_dir / "probabilities.csv", index=False)
    for label, table in tables.items():
        table.to_csv(csv_dir / f"cells_{label.split(',')[0].replace(' ', '')}.csv", index=False)
    ranges.to_csv(csv_dir / "nc_ranges.csv", index=False)

    show = ["cell", "rows", "models", "nc_only", "set_size", "best_fixed", "best_fixed_regret",
            "always_ctm", "random_set", "p_wil_best_holm", "p_mean_best_holm",
            "p_wil_random_holm", "p_mean_random_holm"]
    raw = ["cell", "rows", "nc_only", "best_fixed", "best_fixed_regret", "always_ctm",
           "random_set", "p_wil_best", "p_mean_best", "p_wil_random", "p_mean_random"]
    lines = ["# ImageNet-scale transfer of the NC-only predictor (Phase 3)\n\n",
             "**Source:** `x9_imagenet/phase3_transfer.py`. VGG13-trained regime-free "
             "NC-only heads (22-CSF labels) applied without re-training to 47 timm "
             "ImageNet-1k models; NC z-scored per family; mean imputed set-regret "
             "(AUGRC x1000) over the 21-CSF ImageNet roster; best fixed CSF chosen with "
             "hindsight per cell; tests one-sided (NC only better), Holm over the five "
             "main cells; family cells are exploratory (raw p).\n\n",
             "Gates: GH heads reproduce the stored ResNet18 predictions; GD 47 x 21 x 8 "
             "complete; GR k = 2 grouping without singleton "
             f"(near {sorted(k for k, v in regimes.items() if v == 'near')}). "
             "G1 (August): ResNet-50 accuracy 76.15 vs 76.18; MSR and Energy AUROC "
             "within tolerance of OpenOOD v1.5.\n\n"]
    for label, table in tables.items():
        n_main = 5
        lines.append(f"## {label}: main cells\n\n```\n"
                     + table.iloc[:n_main][show].round(4).to_string(index=False) + "\n```\n\n")
        if label == "21 CSFs":
            lines.append("## 21 CSFs: per family (exploratory)\n\n```\n"
                         + table.iloc[n_main:][raw].round(4).to_string(index=False)
                         + "\n```\n\n")
    lines.append("## Oracle CSF frequency (rows, 21 CSFs)\n\n```\n"
                 + oracle_freq.to_string() + "\n```\n\n")
    lines.append("## Shortlist inclusion rate per CSF (47 models)\n\n```\n"
                 + inclusion.to_string() + "\n```\n\n")
    lines.append("## NC metrics across pools (median [min, max], scale-free metrics)\n\n```\n"
                 + ranges.to_string(index=False) + "\n```\n")
    (out_root / "50_imagenet_transfer.md").write_text("".join(lines))
    print(f"wrote {out_root / '50_imagenet_transfer.md'}")


if __name__ == "__main__":
    main()

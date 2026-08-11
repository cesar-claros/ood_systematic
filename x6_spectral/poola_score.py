"""Pool A prediction scoring (X6 pristine tier; the final verdict script).

Joins the locked prediction JSONs (outputs/poola/*.json) with the
post-lock outcome table (outcomes.csv) and scores the r8 deployed-stack
trial's sign predictions against this pool's first-ever projection-variant
AUGRC outcomes. The pool is pristine by construction: no projection-variant
table existed for these encoder/probe cells before the lock, so every
prediction is genuinely ex ante.

Endpoints (pass-5.1, pre-registered for the l14 confirmatory rerun per the
pass-5 re-review 5.5; the main pool is rescored with the same code but
labeled exploratory):
- PRIMARY: cell-clustered selector regret. Per row, the trial selects the
  variant arm when its predicted delta is positive, else the base arm;
  regret = AUGRC(selected) - min(AUGRC(base), AUGRC(variant)), x1000 units.
  Rows average within cell, cells average macro; compared against the
  always-base and always-variant policies; uncertainty from a two-level
  bootstrap resampling cells and OOD sets within cells (B = 1000, seed 0),
  95 percent percentile intervals.
- SECONDARY: material-row sign accuracy (|true delta| > 1; zero true
  deltas excluded, zero predictions count as misses) with the same
  clustered bootstrap, plus the legacy nulls (always+1 / always-1 /
  majority / per-(family, variant) majority).
- PROSPECTIVE NORM-CHANNEL TEST (B5/B6, only for pools whose measurement
  JSONs carry the pass-5.1 diagnostics): per (cell, OOD set), the
  pre-registered prediction is sign(GradNorm-global outcome delta) =
  sign(pure-l1-channel adaptation-batch AUGRC delta), gated on the head
  residual r_{W,inf} <= 0.1 (Proposition B5's measured hypothesis); the
  mechanism sub-claim scores anti-aligned channels (raw l1 AUC < 0.5)
  against positive outcome deltas.
- Tier-A: deterministic one-sided claims are RETIRED (Theorem B8); where
  present in older JSONs they are reported descriptively only.

Usage (HPC, from code/, after poola_outcomes.py):
    python x6_spectral/poola_score.py --pool l14
    python x6_spectral/poola_score.py --synthetic   # local end-to-end test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "x8_pool_a"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from spectra_campaign_harness import DEPLOYED_TRIAL_KEYS

MATERIAL_DELTA = 1.0
#: outcome-row variant -> tier_a claim entry it is scored against
TIER_A_VARIANT = {"global": "global", "class pred": "class pred",
                  "class": "class pred"}


def load_predictions(poola_dir: Path) -> tuple[dict, dict, dict, dict]:
    """Locked predictions plus pass-5.1 diagnostics.

    Returns (preds, claims, headspan, normchan): trial deltas x1000 keyed
    by cell + (ood, key); tier_a blocks per cell (descriptive only);
    head-span residuals per cell; norm-channel blocks per cell + (ood,)
    (empty dicts when the JSONs predate pass 5.1)."""
    preds: dict[tuple, float] = {}
    claims: dict[tuple, dict] = {}
    headspan: dict[tuple, dict] = {}
    normchan: dict[tuple, dict] = {}
    for path in sorted(poola_dir.glob("*__*.json")):
        rec = json.load(open(path))
        c = rec["cell"]
        cell = (c["encoder"], c["source"], int(c["n_per_class"]),
                int(c["seed"]))
        claims[cell] = rec["tier_a"]
        if "head_span" in rec:
            headspan[cell] = rec["head_span"]
        for ood, block in rec["datasets"].items():
            if "summary" not in block:
                continue
            if "norm_channel" in block:
                normchan[cell + (ood,)] = block["norm_channel"]
            tm = block["summary"]["trial_deployed_mean"]
            for key in DEPLOYED_TRIAL_KEYS:
                dv = tm.get(f"{key}_augrc_delta")
                if dv is not None:
                    preds[cell + (ood, key)] = 1000.0 * float(dv)
    return preds, claims, headspan, normchan


def load_outcomes(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            rows.append({**r, "n_per_class": int(r["n_per_class"]),
                         "seed": int(r["seed"]),
                         "augrc_base": float(r["augrc_base"]),
                         "augrc_var": float(r["augrc_var"]),
                         "delta_augrc": float(r["delta_augrc"])})
    return rows


def join_rows(outcomes: list[dict], preds: dict,
              claims: dict) -> tuple[list[dict], int]:
    joined, unmatched = [], 0
    for r in outcomes:
        cell = (r["encoder"], r["source"], r["n_per_class"], r["seed"])
        pd = preds.get(cell + (r["ood"], r["trial_key"]))
        if pd is None:
            unmatched += 1
            continue
        entry = claims.get(cell, {}).get(TIER_A_VARIANT[r["variant"]], {})
        augrc_sel = r["augrc_var"] if pd > 0 else r["augrc_base"]
        best = min(r["augrc_base"], r["augrc_var"])
        joined.append({**r, "pred_delta": round(pd, 4),
                       "sign_true": int(np.sign(r["delta_augrc"])),
                       "trial_pred": int(np.sign(pd)),
                       "material": abs(r["delta_augrc"]) > MATERIAL_DELTA,
                       "regret": round(augrc_sel - best, 4),
                       "regret_base": round(r["augrc_base"] - best, 4),
                       "regret_var": round(r["augrc_var"] - best, 4),
                       "tier_a_claim": entry.get("prediction", "n/a")})
    return joined, unmatched


CELL_KEYS = ("encoder", "source", "n_per_class", "seed")


def _by_cell_ood(rows: list[dict]) -> dict[tuple, dict[str, list[dict]]]:
    cells: dict[tuple, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list))
    for r in rows:
        cells[tuple(r[k] for k in CELL_KEYS)][r["ood"]].append(r)
    return {c: dict(g) for c, g in cells.items()}


def _macro(rows: list[dict], field: str) -> float | None:
    """Cell-macro mean of a row field (rows average within cell first)."""
    cells = _by_cell_ood(rows)
    if not cells:
        return None
    return float(np.mean([np.mean([r[field] for g in groups.values()
                                   for r in g])
                          for groups in cells.values()]))


def _material_acc(rows: list[dict]) -> float | None:
    scored = [r for r in rows if r["material"] and r["sign_true"] != 0]
    if not scored:
        return None
    return float(np.mean([r["trial_pred"] == r["sign_true"]
                          for r in scored]))


def two_level_bootstrap(rows: list[dict], stat, n_boot: int = 1000,
                        seed: int = 0) -> tuple[float, float] | None:
    """95 percent percentile CI resampling cells, then OOD sets per cell."""
    cells = _by_cell_ood(rows)
    keys = list(cells)
    if not keys:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sample: list[dict] = []
        for ci in rng.integers(0, len(keys), len(keys)):
            groups = cells[keys[ci]]
            oods = list(groups)
            for oi in rng.integers(0, len(oods), len(oods)):
                sample.extend(groups[oods[oi]])
        v = stat(sample)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return (float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))


def _acc(rows: list[dict]) -> tuple[int, float | None]:
    scored = [r for r in rows if r["sign_true"] != 0]
    if not scored:
        return 0, None
    return len(scored), float(np.mean([r["trial_pred"] == r["sign_true"]
                                       for r in scored]))


def _fam_majority(rows: list[dict]) -> float | None:
    scored = [r for r in rows if r["sign_true"] != 0]
    if not scored:
        return None
    groups = defaultdict(list)
    for r in scored:
        groups[(r["family"], r["variant"])].append(r["sign_true"])
    maj = {g: (int(np.sign(sum(v))) or 1) for g, v in groups.items()}
    return float(np.mean([maj[(r["family"], r["variant"])] == r["sign_true"]
                          for r in scored]))


def _null_line(rows: list[dict]) -> str:
    scored = [r for r in rows if r["sign_true"] != 0]
    if not scored:
        return "no scoreable rows"
    pos = float(np.mean([r["sign_true"] > 0 for r in scored]))
    return (f"always+1 {pos:.3f}, always-1 {1 - pos:.3f}, majority "
            f"{max(pos, 1 - pos):.3f}, family-majority "
            f"{_fam_majority(scored):.3f}")


def _ci_s(ci: tuple[float, float] | None, fmt: str = ".3f") -> str:
    return f"[{ci[0]:{fmt}}, {ci[1]:{fmt}}]" if ci else "[n/a]"


def build_report(joined: list[dict], claims: dict, headspan: dict,
                 normchan: dict, unmatched: int,
                 n_boot: int = 1000) -> str:
    cells = _by_cell_ood(joined)
    lines = ["# X6 Pool A scorecard (pass-5.1 endpoints)", "",
             "Generated by poola_score.py from the locked predictions and "
             "the post-lock outcome table. Cells with predictions: "
             f"{len(claims)}; cells joined: {len(cells)}; outcome rows "
             f"joined: {len(joined)} (unmatched: {unmatched}).", "",
             "## Primary endpoint: cell-clustered selector regret "
             "(AUGRC x1000, lower is better)", ""]
    for label, field in (("trial-selected", "regret"),
                         ("always-base", "regret_base"),
                         ("always-variant", "regret_var")):
        m = _macro(joined, field)
        ci = two_level_bootstrap(joined, lambda rows, f=field: _macro(rows, f),
                                 n_boot=n_boot)
        lines.append(f"- {label} policy: macro regret {m:.3f} "
                     f"(95% cell-clustered CI {_ci_s(ci)})")
    lines += ["", "### Regret by family and variant", "",
              "| family | variant | trial | always-base | always-variant |",
              "|---|---|---|---|---|"]
    fam_groups = defaultdict(list)
    for r in joined:
        fam_groups[(r["family"], r["variant"])].append(r)
    for g in sorted(fam_groups):
        rows = fam_groups[g]
        lines.append(f"| {g[0]} | {g[1]} | {_macro(rows, 'regret'):.3f} | "
                     f"{_macro(rows, 'regret_base'):.3f} | "
                     f"{_macro(rows, 'regret_var'):.3f} |")

    lines += ["", "## Secondary endpoint: sign accuracy", ""]
    mat_rows = [r for r in joined if r["material"]]
    ci_acc = two_level_bootstrap(joined, _material_acc, n_boot=n_boot)
    for label, rows in (("all rows", joined),
                        (f"material rows, |delta| > {MATERIAL_DELTA:g}",
                         mat_rows)):
        n, acc = _acc(rows)
        acc_s = f"{acc:.3f}" if acc is not None else "n/a"
        ci_part = f" (95% cell-clustered CI {_ci_s(ci_acc)})" \
            if rows is mat_rows else ""
        lines.append(f"- {label} (n={n}): trial accuracy {acc_s}{ci_part}; "
                     f"nulls: {_null_line(rows)}")
    lines += ["", "### By family and variant", "",
              "| family | variant | n | n mat | acc | acc mat | mean true "
              "delta | mean pred delta |",
              "|---|---|---|---|---|---|---|---|"]

    def cut_table(group_of) -> dict:
        groups = defaultdict(list)
        for r in joined:
            groups[group_of(r)].append(r)
        return groups

    for g, rows in sorted(cut_table(
            lambda r: (r["family"], r["variant"])).items()):
        mat = [r for r in rows if r["material"]]
        n, acc = _acc(rows)
        nm, accm = _acc(mat)
        acc_s = f"{acc:.3f}" if acc is not None else "n/a"
        accm_s = f"{accm:.3f}" if accm is not None else "n/a"
        lines.append(f"| {g[0]} | {g[1]} | {n} | {nm} | {acc_s} | {accm_s} "
                     f"| {np.mean([r['delta_augrc'] for r in rows]):+.2f} "
                     f"| {np.mean([r['pred_delta'] for r in rows]):+.2f} |")
    for title, key in (("By encoder", "encoder"),
                       ("By probe-train size", "n_per_class"),
                       ("By ID source", "source")):
        lines += ["", f"### {title}", "",
                  f"| {key} | n | acc | acc mat | nulls |", "|---|---|---|---|---|"]
        for g, rows in sorted(cut_table(lambda r: r[key]).items(),
                              key=lambda kv: str(kv[0])):
            mat = [r for r in rows if r["material"]]
            n, acc = _acc(rows)
            nm, accm = _acc(mat)
            acc_s = f"{acc:.3f}" if acc is not None else "n/a"
            accm_s = f"{accm:.3f} (n={nm})" if accm is not None else "n/a"
            lines.append(f"| {g} | {n} | {acc_s} | {accm_s} | "
                         f"{_null_line(rows)} |")

    if normchan:
        lines += ["", "## Prospective norm-channel test (B5/B6, "
                  "pre-registered)", ""]
        gated, ungated = [], []
        for r in joined:
            if r["trial_key"] != "gradnorm":
                continue
            cell = tuple(r[k] for k in CELL_KEYS)
            nc = normchan.get(cell + (r["ood"],))
            hs = headspan.get(cell)
            if nc is None or hs is None:
                continue
            row = {**r, "nc": nc, "r_inf": hs["global"]["r_inf"]}
            (gated if row["r_inf"] <= 0.1 else ungated).append(row)

        def nc_acc(rows: list[dict]) -> tuple[int, float | None]:
            scored = [r for r in rows if r["sign_true"] != 0]
            if not scored:
                return 0, None
            return len(scored), float(np.mean(
                [int(np.sign(r["nc"]["l1_augrc_delta"])) == r["sign_true"]
                 for r in scored]))

        n_g, acc_g = nc_acc(gated)
        n_u, acc_u = nc_acc(ungated)
        fmt = lambda a: f"{a:.3f}" if a is not None else "n/a"
        lines.append("- prediction sign(pure-l1-channel batch delta) vs "
                     "GradNorm-global outcome sign: gated "
                     f"(r_W,inf <= 0.1) accuracy {fmt(acc_g)} on n={n_g}; "
                     f"ungated {fmt(acc_u)} on n={n_u}")
        anti = [r for r in gated if r["nc"].get("anti_aligned_l1")]
        if anti:
            pos = float(np.mean([r["delta_augrc"] > 0 for r in anti]))
            mat = float(np.mean([r["delta_augrc"] > MATERIAL_DELTA
                                 for r in anti]))
            lines.append("- mechanism sub-claim (anti-aligned channel "
                         f"implies projection helps): {pos:.3f} positive, "
                         f"{mat:.3f} material-positive, n={len(anti)}")
        else:
            lines.append("- mechanism sub-claim: no gated anti-aligned "
                         "rows")
        r_infs = [hs["global"]["r_inf"] for hs in headspan.values()]
        margin_ds = [nc.get("margin_max_abs_delta", 0.0)
                     for nc in normchan.values()]
        lines.append(f"- head-span census: r_W,inf global median "
                     f"{np.median(r_infs):.4f}, max {max(r_infs):.4f}; "
                     f"margin-factor max |delta| median "
                     f"{np.median(margin_ds):.2e}")

    lines += ["", "## Tier-A (descriptive only; deterministic one-sided "
              "claims retired by Theorem B8)", ""]
    census = defaultdict(int)
    for cell, tier in claims.items():
        for var in ("global", "class pred"):
            if tier.get(var, {}).get("prediction") == "no-benefit":
                census[(var, cell[2])] += 1
    if census:
        parts = [f"{var} at n{npc}: {cnt}" for (var, npc), cnt
                 in sorted(census.items())]
        lines.append(f"Claim census (cells): {'; '.join(parts)}.")
    else:
        lines.append("Claim census: no no-benefit claims in this pool.")
    for var_label, variants in (("global", {"global"}),
                                ("class pred", {"class pred", "class"})):
        rows = [r for r in joined if r["variant"] in variants
                and r["tier_a_claim"] == "no-benefit"]
        if not rows:
            lines.append(f"- {var_label} claims: no claimed rows.")
            continue
        falsified = [r for r in rows if r["delta_augrc"] > MATERIAL_DELTA]
        cells_claimed = {(r["encoder"], r["source"], r["n_per_class"],
                          r["seed"]) for r in rows}
        cells_falsified = {(r["encoder"], r["source"], r["n_per_class"],
                            r["seed"]) for r in falsified}
        lines.append(
            f"- {var_label} claims: {len(rows)} rows over "
            f"{len(cells_claimed)} cells; held on "
            f"{1 - len(falsified) / len(rows):.3f} of rows "
            f"({len(falsified)} material-positive); cell-level: "
            f"{len(cells_falsified)}/{len(cells_claimed)} claims falsified "
            f"by at least one row.")
    lines.append("- class avg: registered scope, no class-avg trial keys "
                 "exist in this pool; not scored.")
    return "\n".join(lines) + "\n"


def write_outputs(joined: list[dict], report: str, out_dir: Path) -> None:
    csv_path = out_dir / "poola_scoring.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(joined[0].keys()))
        writer.writeheader()
        writer.writerows(joined)
    md_path = out_dir / "poola_report.md"
    md_path.write_text(report)
    print(f"{len(joined)} scored rows -> {csv_path}\nreport -> {md_path}")


def run_synthetic() -> None:
    """End-to-end self-test of the pass-5.1 protocol: measure -> lock
    manifest (hash-only verification plus a tamper test) -> disjoint
    outcomes -> score, on fabricated features. Asserts structural
    integrity (join coverage, row counts, finiteness, diagnostic
    presence, report sections), never outcome values."""
    import tempfile

    import projection_filtering_analysis as pfa
    from poola_lock import sha256, verify_lock
    from poola_measure import make_synthetic, measure_cell
    from poola_outcomes import ADAPT_ROWS, outcome_cell

    tmp = Path(tempfile.mkdtemp(prefix="poola_e2e_"))
    make_synthetic(tmp)
    out = tmp / "poola"
    out.mkdir()
    cells = [("cifar10", 25, 0), ("cifar100", 0, 1)]
    for source, n_pc, seed in cells:
        status = measure_cell(tmp, out, "synthetic", source, n_pc, seed)
        assert status == "measured", f"measure: {status}"

    manifest = {"pool": "synthetic", "rule_version": "r8+pass5.1",
                "predictions": [{"file": p.name, "sha256": sha256(p)}
                                for p in sorted(out.glob("*__*.json"))],
                "features": [{"file": f.name, "sha256": sha256(f)}
                             for f in sorted(tmp.glob("*.npz"))]}
    mpath = out / "manifest.json"
    with open(mpath, "w") as fh:
        json.dump(manifest, fh)
    ok = verify_lock("l14", tmp, None, git_checks=False, pred_dir=out,
                     manifest_path=mpath)
    assert not ok, f"clean lock failed verification: {ok}"
    first = sorted(out.glob("*__*.json"))[0]
    orig = first.read_text()
    first.write_text(orig + " ")
    tampered = verify_lock("l14", tmp, None, git_checks=False, pred_dir=out,
                           manifest_path=mpath)
    assert tampered, "tampered prediction not detected"
    first.write_text(orig)

    outcome_rows: list[dict] = []
    for source, n_pc, seed in cells:
        rows = outcome_cell(tmp, "synthetic", source, n_pc, seed,
                            skip_rows=ADAPT_ROWS)
        assert rows, f"no outcome rows for {source}"
        outcome_rows.extend(rows)
    n_expect = sum(len(pfa.OOD_DATASETS[s]) for s, _, _ in cells) \
        * len(DEPLOYED_TRIAL_KEYS)
    assert len(outcome_rows) == n_expect, (len(outcome_rows), n_expect)
    preds, claims, headspan, normchan = load_predictions(out)
    assert len(headspan) == len(cells), "head_span diagnostics missing"
    assert len(normchan) == sum(len(pfa.OOD_DATASETS[s])
                                for s, _, _ in cells), \
        "norm_channel diagnostics missing"
    joined, unmatched = join_rows(outcome_rows, preds, claims)
    assert unmatched == 0, f"{unmatched} unmatched rows"
    assert len(joined) == n_expect, (len(joined), n_expect)
    assert all(np.isfinite(r["delta_augrc"]) and np.isfinite(r["pred_delta"])
               and np.isfinite(r["regret"]) and r["regret"] >= 0
               for r in joined)
    assert any(r["trial_pred"] != 0 for r in joined), "all predictions zero"
    report = build_report(joined, claims, headspan, normchan, unmatched,
                          n_boot=200)
    for section in ("Primary endpoint", "Secondary endpoint",
                    "Prospective norm-channel"):
        assert section in report, f"report missing section: {section}"
    write_outputs(joined, report, tmp)
    n, acc = _acc(joined)
    print(f"synthetic e2e OK: {len(joined)} rows joined "
          f"(disjoint eval, {ADAPT_ROWS} adaptation rows excluded), lock "
          f"verify + tamper test passed, macro regret "
          f"{_macro(joined, 'regret'):.3f} vs always-base "
          f"{_macro(joined, 'regret_base'):.3f}, sign agreement {acc:.2f} "
          f"on {n} rows (informational), artifacts in {tmp}")


def main() -> None:
    from poola_lock import pred_dir_for
    parser = argparse.ArgumentParser(
        description="X6 Pool A prediction scoring (post-outcomes)")
    parser.add_argument("--pool", choices=["main", "l14"], default="l14",
                        help="l14 = confirmatory rerun (preregistered "
                             "endpoints); main = historical, exploratory")
    parser.add_argument("--poola-dir", type=str, default=None,
                        help="override the per-pool prediction directory")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.synthetic:
        run_synthetic()
        return
    poola_dir = Path(args.poola_dir) if args.poola_dir \
        else pred_dir_for(args.pool)
    outcomes_csv = poola_dir / "outcomes.csv"
    if not outcomes_csv.exists():
        sys.exit(f"missing {outcomes_csv}: run poola_outcomes.py first")
    preds, claims, headspan, normchan = load_predictions(poola_dir)
    if not preds:
        sys.exit(f"no prediction JSONs found in {poola_dir}")
    joined, unmatched = join_rows(load_outcomes(outcomes_csv), preds, claims)
    if not joined:
        sys.exit("no rows joined: outcome table and predictions disagree "
                 "on cells")
    report = build_report(joined, claims, headspan, normchan, unmatched)
    if args.pool == "main":
        report = report.replace(
            "# X6 Pool A scorecard (pass-5.1 endpoints)",
            "# X6 Pool A scorecard (pass-5.1 endpoints; EXPLORATORY: "
            "historical pool, non-auditable lock and non-disjoint "
            "samples per the pass-5 re-review)")
    write_outputs(joined, report, poola_dir)
    print()
    print(report)


if __name__ == "__main__":
    main()

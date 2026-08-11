"""Pool A prediction scoring (X6 pristine tier; the final verdict script).

Joins the locked prediction JSONs (outputs/poola/*.json) with the
post-lock outcome table (outcomes.csv) and scores the r8 deployed-stack
trial's sign predictions against this pool's first-ever projection-variant
AUGRC outcomes. The pool is pristine by construction: no projection-variant
table existed for these encoder/probe cells before the lock, so every
prediction is genuinely ex ante.

Scored claims, frozen with this script before any outcome exists:
- Trial arm (primary): per (cell, trial key, OOD set), predicted sign =
  sign of the r8 batch-trial mean AUGRC delta; true sign = sign of the
  full-test outcome delta. Conventions mirror score_tier_b: rows with a
  zero true delta are excluded, a zero predicted sign counts as a miss,
  nulls are always+1 / always-1 / overall majority / per-(family, variant)
  majority, and the material cut keeps |delta| > 1 (AUGRC x1000).
- Tier-A arm (one-sided): "no-benefit" claims from ID-only diagnostics
  are falsified by a material positive outcome (delta > +1) and hold
  otherwise; "undetermined" cells make no claim and are not scored.
  Variant mapping: global variants score against tier_a["global"];
  class-pred variants against tier_a["class pred"]; the RecError class
  variant also scores against tier_a["class pred"] (the same per-class
  recoverability gate, with no routing involved). Class-avg claims are
  unscoreable in this pool (no class-avg trial keys exist): registered
  scope, reported but not scored.

Usage (HPC, from code/, after poola_outcomes.py --locked):
    python x6_spectral/poola_score.py
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


def load_predictions(poola_dir: Path) -> tuple[dict, dict]:
    """Locked predictions: (cell + (ood, key)) -> delta x1000; cell -> tier_a."""
    preds: dict[tuple, float] = {}
    claims: dict[tuple, dict] = {}
    for path in sorted(poola_dir.glob("*__*.json")):
        rec = json.load(open(path))
        c = rec["cell"]
        cell = (c["encoder"], c["source"], int(c["n_per_class"]),
                int(c["seed"]))
        claims[cell] = rec["tier_a"]
        for ood, block in rec["datasets"].items():
            if "summary" not in block:
                continue
            tm = block["summary"]["trial_deployed_mean"]
            for key in DEPLOYED_TRIAL_KEYS:
                dv = tm.get(f"{key}_augrc_delta")
                if dv is not None:
                    preds[cell + (ood, key)] = 1000.0 * float(dv)
    return preds, claims


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
        joined.append({**r, "pred_delta": round(pd, 4),
                       "sign_true": int(np.sign(r["delta_augrc"])),
                       "trial_pred": int(np.sign(pd)),
                       "material": abs(r["delta_augrc"]) > MATERIAL_DELTA,
                       "tier_a_claim": entry.get("prediction", "n/a")})
    return joined, unmatched


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


def build_report(joined: list[dict], claims: dict, unmatched: int) -> str:
    lines = ["# X6 Pool A pristine-tier scorecard", "",
             "Generated by poola_score.py from the locked predictions "
             "(outputs/poola/*.json) and the post-lock outcome table "
             "(outcomes.csv). Cells with predictions: "
             f"{len(claims)}; outcome rows joined: {len(joined)} "
             f"(unmatched: {unmatched}).", "", "## Trial arm (primary)", ""]
    for label, rows in (("all rows", joined),
                        (f"material rows, |delta| > {MATERIAL_DELTA:g}",
                         [r for r in joined if r["material"]])):
        n, acc = _acc(rows)
        acc_s = f"{acc:.3f}" if acc is not None else "n/a"
        lines.append(f"- {label} (n={n}): trial accuracy {acc_s}; nulls: "
                     f"{_null_line(rows)}")
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

    lines += ["", "## Tier-A arm (one-sided no-benefit claims)", ""]
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
    """End-to-end self-test: measure -> outcomes -> score on fabricated
    features, asserting structural integrity (join coverage, row counts,
    finiteness), never outcome values."""
    import tempfile

    import projection_filtering_analysis as pfa
    from poola_measure import make_synthetic, measure_cell
    from poola_outcomes import outcome_cell

    tmp = Path(tempfile.mkdtemp(prefix="poola_e2e_"))
    make_synthetic(tmp)
    out = tmp / "poola"
    out.mkdir()
    cells = [("cifar10", 25, 0), ("cifar100", 0, 1)]
    outcome_rows: list[dict] = []
    for source, n_pc, seed in cells:
        status = measure_cell(tmp, out, "synthetic", source, n_pc, seed)
        assert status == "measured", f"measure: {status}"
        rows = outcome_cell(tmp, "synthetic", source, n_pc, seed)
        assert rows, f"no outcome rows for {source}"
        outcome_rows.extend(rows)
    n_expect = sum(len(pfa.OOD_DATASETS[s]) for s, _, _ in cells) \
        * len(DEPLOYED_TRIAL_KEYS)
    assert len(outcome_rows) == n_expect, (len(outcome_rows), n_expect)
    preds, claims = load_predictions(out)
    joined, unmatched = join_rows(outcome_rows, preds, claims)
    assert unmatched == 0, f"{unmatched} unmatched rows"
    assert len(joined) == n_expect, (len(joined), n_expect)
    assert all(np.isfinite(r["delta_augrc"]) and np.isfinite(r["pred_delta"])
               for r in joined)
    assert any(r["trial_pred"] != 0 for r in joined), "all predictions zero"
    report = build_report(joined, claims, unmatched)
    write_outputs(joined, report, tmp)
    n, acc = _acc(joined)
    print(f"synthetic e2e OK: {len(joined)} rows joined, sign agreement "
          f"{acc:.2f} on {n} scoreable rows (informational), artifacts in "
          f"{tmp}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="X6 Pool A prediction scoring (post-outcomes)")
    parser.add_argument("--poola-dir", type=str,
                        default="x6_spectral/outputs/poola")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.synthetic:
        run_synthetic()
        return
    poola_dir = Path(args.poola_dir)
    outcomes_csv = poola_dir / "outcomes.csv"
    if not outcomes_csv.exists():
        sys.exit(f"missing {outcomes_csv}: run poola_outcomes.py --locked "
                 "first")
    preds, claims = load_predictions(poola_dir)
    if not preds:
        sys.exit(f"no prediction JSONs found in {poola_dir}")
    joined, unmatched = join_rows(load_outcomes(outcomes_csv), preds, claims)
    if not joined:
        sys.exit("no rows joined: outcome table and predictions disagree "
                 "on cells")
    report = build_report(joined, claims, unmatched)
    write_outputs(joined, report, poola_dir)
    print()
    print(report)


if __name__ == "__main__":
    main()

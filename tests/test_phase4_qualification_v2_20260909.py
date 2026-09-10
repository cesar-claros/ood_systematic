"""Corrected phase-4 qualification (repair item P1-B): the reviewer's F2 and
F11 defect tests, inverted against the version-2 simulator, plus the
declared-state contract with reader v2."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from rn18_handoff_replication import phase4_qualification_v2 as Q      # noqa: E402
from rn18_handoff_replication import rn18_analysis_v2 as R             # noqa: E402

MU = np.zeros((4, 4))
pa = lambda l: 1 - np.exp(l)


def _level_err(c):
    e00 = (np.abs(pa(c["lE00"]) - c["aurocE"]) + np.abs(pa(c["lC00"]) - c["aurocC"])) / 2
    e10 = (np.abs(pa(c["lE10"]) - c["aurocE"]) + np.abs(pa(c["lC10"]) - c["aurocC"])) / 2
    return e00 - e10


def test_level_errors_inherit_family_dependence_and_distribution():
    a = Q.sim_cells(dict(Q.BASE, corr=0.0), 5, np.random.default_rng(19), MU)
    b = Q.sim_cells(dict(Q.BASE, corr=0.9), 5, np.random.default_rng(19), MU)
    assert not np.array_equal(_level_err(a), _level_err(b))
    g = Q.sim_cells(dict(Q.BASE, dist="gauss"), 5, np.random.default_rng(19), MU)
    t3 = Q.sim_cells(dict(Q.BASE, dist="t3"), 5, np.random.default_rng(19), MU)
    assert not np.array_equal(_level_err(g), _level_err(t3))


def test_margin_is_derived_from_the_level_forecasts_and_constants_are_literal():
    c = Q.sim_cells(Q.BASE, 5, np.random.default_rng(31), MU)
    assert np.allclose(c["M"], c["lE00"] - c["lC00"])
    assert float(np.mean(np.sign(c["M"]) != np.sign(c["lE00"] - c["lC00"]))) == 0.0
    assert np.unique(c["preds"]["always_energy"]).tolist() == [-1.0] and np.unique(c["preds"]["always_ctm"]).tolist() == [1.0]


def test_pi_one_source_keeps_every_cell_and_zeroes_only_the_augrc_gap():
    c = Q.sim_cells(dict(Q.BASE, pi_one_source=True), 5, np.random.default_rng(31), MU)
    assert int(c["keep"].sum()) == 80 and np.isfinite(c["aurocE"]).all() and np.isfinite(c["aurocC"]).all()
    assert np.all(c["dG"][c["s_idx"] == 3] == 0.0) and np.any(c["dG"][c["s_idx"] != 3] != 0.0)


def test_declared_states_match_reader_v2():
    miss = Q.sim_cells(dict(Q.BASE, missing_cell=True), 10, np.random.default_rng(1), MU)
    assert Q.panel_state(miss) == "VALIDATION FAILED [key_set]"
    dup = Q.sim_cells(dict(Q.BASE, duplicate_family=True), 10, np.random.default_rng(1), MU)
    assert Q.panel_state(dup) == "VALIDATION FAILED [families]"
    nf = Q.sim_cells(dict(Q.BASE, nonfinite=True), 10, np.random.default_rng(1), MU)
    df = Q.to_frame(nf)
    assert R.sel_endpoint(df, Q.FixedComparators(), 1.0, "t")["verdict"].startswith("NOT ESTIMABLE")
    assert R.level_endpoint(df, 1.0, "t")["verdict"].startswith("NOT ESTIMABLE")
    deg = Q.sim_cells(dict(Q.BASE, comp_mode="degenerate_ctm", p00_bias=0.3), 10, np.random.default_rng(1), MU)
    rs = R.sel_endpoint(Q.to_frame(deg), Q.FixedComparators(), 1.0, "t")
    assert rs["P00_identical_to_always_ctm"] and rs["comparators"]["always_ctm"]["ci"] is None
    assert rs["comparators"][Q.REFERENCE]["ci"] is not None


def test_bounded_beta_family_null_is_exact_and_valid():
    c = Q.sim_cells(dict(Q.BASE, beta_family_null=True), 10, np.random.default_rng(2), MU)
    d = _level_err(c)
    fam_means = np.array([d[c["f_idx"] == f].mean() for f in range(10)])
    per_cell_spread = max(np.ptp(d[c["f_idx"] == f]) for f in range(10))
    assert per_cell_spread < 1e-12                                     # each family's delta is shared by its 16 cells
    assert np.all(0.2 - fam_means > 0) and np.all(0.2 - fam_means < 0.5)
    assert np.all((pa(c["lE10"]) > 0) & (pa(c["lC10"]) > 0) & (pa(c["lC00"]) < 1))


def test_scenario_streams_do_not_depend_on_python_hash_seed():
    script = "import sys; sys.path.insert(0, '.'); from rn18_handoff_replication.phase4_qualification_v2 import skey; print(skey('S0_base'))"
    keys = [int(subprocess.run([sys.executable, "-c", script], env={**os.environ, "PYTHONHASHSEED": h}, text=True,
                               capture_output=True, check=True, cwd=str(CODE)).stdout) for h in ("1", "2")]
    assert keys[0] == keys[1]


def test_fast_path_equals_reader_v2_numerically():
    c = Q.sim_cells(Q.BASE, 10, np.random.default_rng(77), MU)
    Q.check_reader(c, Q.fast_sel(c), Q.fast_level(c), "unit")           # asserts to 1e-10 on estimates, SEs, endpoints, verdicts


def test_per_example_ho_fixture_goes_through_frozen_augrc_and_gate():
    sub, truth = Q.sim_ho_source_examples(dict(Q.BASE, ho="planted"), np.random.default_rng(9))
    assert len(sub) == 96 and sub.cell.nunique() == 24
    from rn18_handoff_replication.rn18_analysis import ho_source
    assert ho_source(sub, "dK", b=0)["verdict"] == "HO-RETAINED"
    sub_r, _ = Q.sim_ho_source_examples(dict(Q.BASE, ho="reversed"), np.random.default_rng(9))
    assert ho_source(sub_r, "dK", b=0)["verdict"] != "HO-RETAINED"
    # the expected raw gap of the construction has the sign of the planted target
    for cell, pts in truth.items():
        y = [p[1] for p in pts]
        assert y == sorted(y)


def test_manifest_has_47_scenarios_and_persisted_keys():
    S = Q.scenarios()
    assert len(S) == 47 and sum(s["held_out"] for s in S) == 3
    names = {s["name"] for s in S}
    for must in ("null_level_bounded_beta_family", "dep_dropout_two_clusters", "pred_p00_degenerate_always_ctm", "dist_family_outlier"):
        assert must in names

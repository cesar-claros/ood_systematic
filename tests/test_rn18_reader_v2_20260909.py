"""Fail-closed behavior of rn18_analysis_v2 (repair item P1-C, review F9).

The validator is exercised on a scratch copy of the JSON artifacts with
the module paths redirected; each tamper must raise ValidationFailure
with the named check. The inference primitives are exercised directly.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from rn18_handoff_replication import rn18_analysis as v1          # noqa: E402
from rn18_handoff_replication import rn18_analysis_v2 as v2        # noqa: E402

ART = CODE / "rn18_handoff_replication"


def _have_artifacts() -> bool:
    return (ART / "FREEZE.json").exists() and (ART / "outputs/fourshift_rn18").exists() and \
        len(list((ART / "outputs/fourshift_rn18").glob("*.json"))) == 96


needs_artifacts = pytest.mark.skipif(not _have_artifacts(), reason="RN18 artifacts not present")


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """Copy the JSON records, FREEZE, manifests, simulations and phase-1 outputs
    (not the 2.4 GiB sidecars) to tmp_path and point the modules at it."""
    root = tmp_path / "rn18_handoff_replication"
    for sub in ("manifests", "simulations", "outputs/phase1_nc1"):
        shutil.copytree(ART / sub, root / sub)
    for d in ("fourshift_rn18", "fourshift_vgg_bridge"):
        (root / "outputs" / d).mkdir(parents=True)
        for p in (ART / "outputs" / d).glob("*.json"):
            shutil.copy(p, root / "outputs" / d / p.name)
    shutil.copy(ART / "FREEZE.json", root / "FREEZE.json")
    shutil.copy(ART / "rn18_analysis.py", root / "rn18_analysis.py")
    # freeze hashes are relative to code/: mirror the hashed code files and pilot0 csv
    fz = json.loads((ART / "FREEZE.json").read_text())
    for grp in ("code", "manifests"):
        for rel in fz["hashes"][grp]:
            src, dst = CODE / rel, tmp_path / rel
            if src.exists() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(src, dst)
    shutil.copy(ART / "outputs/rn18_report.json", root / "outputs/rn18_report.json")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(v2, "OUT", root / "outputs")
    monkeypatch.setattr(v1, "OUT", root / "outputs")
    monkeypatch.setattr(v1, "DIR_RN18", root / "outputs/fourshift_rn18")
    monkeypatch.setattr(v1, "DIR_VGG_DET", root / "outputs/fourshift_vgg_bridge")
    monkeypatch.setattr(v1, "DIR_P1", root / "outputs/phase1_nc1")
    monkeypatch.setattr(v1, "SEV", tmp_path / "pilot0/clip_severity_v2.csv")
    monkeypatch.setattr(v2, "FREEZE", root / "FREEZE.json")
    monkeypatch.setattr(v2, "PANEL", root / "manifests/expected_panel.json")
    monkeypatch.setattr(v2, "QUAL_V1", root / "simulations/qualification_report.json")
    monkeypatch.setattr(v2, "QUAL_V2", root / "simulations/qualification_report_v2.json")
    monkeypatch.setattr(v2, "AUDIT_V1", root / "simulations/audit_results.json")
    return root


def _load():
    return v2._load_all()


@needs_artifacts
def test_validator_passes_on_untampered_copy(scratch):
    axes, recs, vgg, p1 = _load()
    rep = v2.validate(recs, vgg, p1, axes)
    assert rep["verdict"] == "VALIDATION PASSED"
    assert rep["checks"]["key_set"] == {"n_expected": 384, "n_present": 384, "missing": 0, "extra": 0, "bad_fields": 0}
    assert rep["checks"]["denominators"]["rn18_records"] == 96 and rep["checks"]["families"]["n"] == 10
    assert "reader_of_record_discrepancy" in rep["checks"]            # the recorded c7d1b98 serialization fix
    assert rep["checks"]["license_v2"] is None or "ABSENT" in str(rep["checks"]["license_v2"])


@needs_artifacts
def test_validator_fails_on_changed_record(scratch):
    p = next((scratch / "outputs/fourshift_rn18").glob("*.json"))
    r = json.loads(p.read_text()); r["ood"]["mnist_new"]["gap_raw"] += 1e-9
    p.write_text(json.dumps(r))
    axes, recs, vgg, p1 = _load()
    with pytest.raises(v2.ValidationFailure) as e:
        v2.validate(recs, vgg, p1, axes)
    assert e.value.check == "inventory:fourshift_rn18" and e.value.detail["changed"] == [p.name]


@needs_artifacts
def test_validator_fails_on_unknown_file(scratch):
    (scratch / "outputs/fourshift_rn18/notes.txt").write_text("x")
    axes, recs, vgg, p1 = _load()
    with pytest.raises(v2.ValidationFailure) as e:
        v2.validate(recs, vgg, p1, axes)
    assert e.value.detail["other_files"] == ["notes.txt"]


@needs_artifacts
def test_validator_fails_on_missing_key_and_nonfinite_field(scratch):
    axes, recs, vgg, p1 = _load()
    recs2 = [dict(r) for r in recs]
    recs2[0] = dict(recs2[0], ood={k: v for k, v in recs2[0]["ood"].items() if k != "stl10_new"})
    with pytest.raises(v2.ValidationFailure) as e:
        v2.validate(recs2, vgg, p1, axes)
    assert e.value.check == "key_set" and e.value.detail["n_missing"] == 1
    recs3 = json.loads(json.dumps(recs)); recs3[5]["ood"]["kmnist_new"]["a"] = float("nan")
    with pytest.raises(v2.ValidationFailure) as e:
        v2.validate(recs3, vgg, p1, axes)
    assert e.value.check == "key_set" and e.value.detail["n_bad_fields"] == 1


@needs_artifacts
def test_validator_fails_on_wrong_seed_and_unlicensed_family(scratch):
    axes, recs, vgg, p1 = _load()
    recs2 = json.loads(json.dumps(recs))
    ce = next(r for r in recs2 if r["component"] == "standalone_ce"); ce["seed"] += 1
    with pytest.raises(v2.ValidationFailure) as e:
        v2.validate(recs2, vgg, p1, axes)
    assert e.value.check == "families"
    qp = scratch / "simulations/qualification_report.json"
    q = json.loads(qp.read_text()); q["licenses"]["LEVEL_Nf5"]["licensed"] = False
    qp.write_text(json.dumps(q))
    with pytest.raises(v2.ValidationFailure) as e:                    # the hash check fires first (correct order)
        v2.validate(recs, vgg, p1, axes)
    assert e.value.check == "freeze_hashes"
    fz = json.loads((scratch / "FREEZE.json").read_text())            # re-point the freeze hash to reach the license check
    fz["hashes"]["manifests"]["rn18_handoff_replication/simulations/qualification_report.json"] = v2.sha(qp)
    (scratch / "FREEZE.json").write_text(json.dumps(fz))
    with pytest.raises(v2.ValidationFailure) as e:
        v2.validate(recs, vgg, p1, axes)
    assert e.value.check == "licenses_v1" and "LEVEL_Nf5" in e.value.detail


@needs_artifacts
def test_readout_withheld_without_v2_license(scratch, capsys):
    v2.run(b=5, validate_only=False)
    out = capsys.readouterr().out
    assert "READOUT WITHHELD" in out and (scratch / "outputs/rn18_report_v2_validation.json").exists()
    assert not (scratch / "outputs/rn18_report_v2.json").exists()


def test_nan_prediction_is_unavailable_not_tie():
    p = v2.choice_prob(np.array([np.nan, 0.0, 1e-3, -1e-3]))
    assert np.isnan(p[0]) and p[1] == 0.5 and p[2] == 1.0 and p[3] == 0.0
    assert v1.choice_prob(np.array([np.nan]))[0] == 0.5                  # the version-1 defect, kept as the record


def test_zero_se_is_not_estimable():
    jk = v2.jackknife(pd.DataFrame({"family": list("abcde"), "x": [2.0] * 5}), lambda d: d.x.mean())
    assert jk["not_estimable"] and v2.interval(jk, 0.025, 1.1) is None
    assert v2.level_verdict(None)["verdict"] == "NOT ESTIMABLE"
    assert v2.sel_verdict({n: None for n in v2.NAMES}).startswith("NOT ESTIMABLE")


def test_decisions_use_full_precision():
    assert v2.level_verdict([-4e-6, 0.02])["verdict"] == "unresolved direction"
    assert v2.display([-4e-6, 0.02]) == [-0.0, 0.02] or v2.display([-4e-6, 0.02]) == [0.0, 0.02]
    assert v2.level_verdict([2e-6, 0.02])["verdict"] == "resolved improvement"
    ivs = {n: [0.0021, 0.01] for n in v2.NAMES}
    assert v2.sel_verdict(ivs).startswith("PRACTICALLY SUPERIOR")
    ivs[v2.REFERENCE] = [0.0019999, 0.01]
    assert v2.sel_verdict(ivs) == "UNRESOLVED"

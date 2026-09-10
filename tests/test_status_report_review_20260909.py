"""Independent counterexamples for the September 9 scientific status review."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import orjson
import polars as pl
import pytest
from numpy.typing import NDArray
from scipy.stats import beta, t

from rn18_handoff_replication import phase4_qualification as qualification
from rn18_handoff_replication import rn18_analysis as analysis
from rn18_handoff_replication.theory.gaussian_diagnostics import psd_sqrt

OUTPUT = Path("regression_outputs/status_report_review_20260909")


def _save(name: str, values: dict[str, object]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / f"{name}.json").write_bytes(
        orjson.dumps(values, option=orjson.OPT_INDENT_2)
    )


def _level_fixture() -> pl.DataFrame:
    rows = [
        {
            "family": f"family{family}",
            "source": f"source{source}",
            "cell": f"checkpoint{family}_{source}",
            "ood_set": f"shift{shift}",
            "aurocE": 0.5,
            "aurocC": 0.5,
            "l_E": float(np.log(0.25)),
            "l_C": float(np.log(0.25)),
            "l_E_p10": float(np.log(0.375)),
            "l_C_p10": float(np.log(0.375)),
        }
        for family in range(5)
        for source in range(2)
        for shift in range(2)
    ]
    return pl.DataFrame(rows)


def test_maxlogit_boundary_is_not_general_over_alignment_profiles() -> None:
    """Two tied OOD alignments violate the unrestricted chance-crossing claim."""
    rng = np.random.default_rng(20260909)
    n_draws = 400_000
    radius = 4.0
    id_noise = rng.standard_normal((n_draws, 3))
    ood_noise = rng.standard_normal((n_draws, 3))
    id_noise = (id_noise - id_noise.mean(axis=1, keepdims=True)) * np.sqrt(1.5)
    ood_noise = (ood_noise - ood_noise.mean(axis=1, keepdims=True)) * np.sqrt(1.5)
    id_scores = (id_noise + [radius, -radius / 2, -radius / 2]).max(axis=1)
    ood_scores = (ood_noise + [radius, radius, -2 * radius]).max(axis=1)
    estimate = float(np.mean(id_scores > ood_scores))
    standard_error = float(np.sqrt(estimate * (1 - estimate) / n_draws))
    high_snr_limit = float(0.25 + np.arcsin(0.25) / (2 * np.pi))
    _save(
        "maxlogit_alignment_counterexample",
        {
            "C": 3,
            "s": radius,
            "a": 0.5,
            "gamma": 2.0,
            "rho": 1.0,
            "paired_mc_draws": n_draws,
            "auroc": estimate,
            "mc_standard_error": standard_error,
            "fixed_index_claim": 0.5,
            "high_snr_limit": high_snr_limit,
        },
    )
    assert abs(estimate - high_snr_limit) < 5 * standard_error
    assert estimate < 0.31


def test_level_zero_variance_currently_produces_formal_claim() -> None:
    """Document the missing degeneracy branch rather than certify this behavior."""
    result = analysis.level_endpoint(_level_fixture().to_pandas(), mult=1.1)
    _save("level_zero_variance", result)
    assert result["ci"] == [0.125, 0.125]
    assert result["verdict"] == "resolved improvement"


def test_level_missing_shift_is_not_detected() -> None:
    """A structurally absent row escapes the check for nonfinite existing rows."""
    fixture = _level_fixture().slice(1)
    result = analysis.level_endpoint(fixture.to_pandas(), mult=1.1)
    _save("level_missing_shift", {"rows": fixture.height, **result})
    assert result["N_f"] == 5
    assert not result["verdict"].startswith("INELIGIBLE")


def test_nonfinite_comparator_prediction_is_silently_a_tie() -> None:
    """Missing predictions must not acquire the loss of an intentional abstention."""
    assert analysis.choice_prob(np.array([np.nan]))[0] == 0.5


def test_level_simulator_does_not_inherit_family_correlation() -> None:
    """Changing source dependence leaves all generated LEVEL errors identical."""
    base = dict(qualification.BASE, corr=0.0)
    dependent = dict(base, corr=0.9)
    first = qualification.sim_cells(
        base, 5, np.random.default_rng(19), np.zeros((4, 4))
    )
    second = qualification.sim_cells(
        dependent, 5, np.random.default_rng(19), np.zeros((4, 4))
    )
    assert not np.array_equal(first["dA"], second["dA"])
    np.testing.assert_array_equal(
        first["e00"] - first["e10"], second["e00"] - second["e10"]
    )


def test_simulator_uses_incompatible_margins_and_nonconstant_constants() -> None:
    """SEL margins and LEVEL forecasts do not describe the same frozen predictor."""
    cells = qualification.sim_cells(
        qualification.BASE, 5, np.random.default_rng(31), np.zeros((4, 4))
    )
    mismatch = float(
        np.mean(np.sign(cells["M"]) != np.sign(cells["lE00"] - cells["lC00"]))
    )
    _save("simulation_margin_mismatch", {"fraction_inconsistent": mismatch})
    assert mismatch > 0.2
    assert np.unique(np.sign(cells["preds"]["always_energy"])).size == 2
    assert np.unique(np.sign(cells["preds"]["always_ctm"])).size == 2


def test_failure_prevalence_boundary_drops_valid_pure_auroc_source() -> None:
    """The pi=1 fixture drops a source even though SEL/LEVEL target pure AUROC."""
    cells = qualification.sim_cells(
        dict(qualification.BASE, pi_one_source=True),
        5,
        np.random.default_rng(31),
        np.zeros((4, 4)),
    )
    assert int(cells["keep"].sum()) == 60
    assert np.isfinite(cells["aurocE"]).all()
    assert np.isfinite(cells["aurocC"]).all()


def test_bounded_skewed_family_null_exposes_uncovered_level_regime() -> None:
    """Valid bounded MAE differences can violate the licensed nominal size."""
    rng = np.random.default_rng(20260909)
    n_replications = 100_000
    findings: dict[str, object] = {}
    for n_families in (5, 10):
        differences: NDArray[np.float64] = 0.2 * (
            rng.beta(0.3, 3.0, size=(n_replications, n_families)) - 0.3 / 3.3
        )
        mean = differences.mean(axis=1)
        se = differences.std(axis=1, ddof=1) / np.sqrt(n_families)
        half_width = 1.1 * t.ppf(1 - 0.025 / 2, n_families - 1) * se
        lower = np.round(mean - half_width, 5)
        upper = np.round(mean + half_width, 5)
        false_claims = int(np.count_nonzero((lower > 0) | (upper < 0)))
        rate = false_claims / n_replications
        mc_lower = float(
            beta.ppf(0.025, false_claims, n_replications - false_claims + 1)
        )
        mc_upper = float(
            beta.ppf(0.975, false_claims + 1, n_replications - false_claims)
        )
        # e00=0.2 and e10=0.2-difference are valid MAEs for AUROC=0.5;
        # sharing each family's errors over all sources/shifts preserves this mean.
        assert np.all((0.2 - differences > 0) & (0.2 - differences < 0.5))
        assert mc_lower > 0.025
        findings[str(n_families)] = {
            "false_claim_rate": rate,
            "mc_95_interval": [mc_lower, mc_upper],
            "replications": n_replications,
            "nominal": 0.025,
            "multiplier": 1.1,
        }
    _save("bounded_family_skew", findings)


def test_scenario_stream_depends_on_unrecorded_python_hash_seed() -> None:
    """Master seeds alone do not specify the simulator's scenario streams."""
    script = "print(hash('S0_base') % (2 ** 31))"
    keys = []
    for hash_seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            text=True,
            capture_output=True,
            check=True,
        )
        keys.append(int(result.stdout))
    _save("python_hash_streams", {"scenario_keys": keys})
    assert keys[0] != keys[1]


def test_float32_covariance_serialization_can_violate_psd_contract() -> None:
    """Rounding an exact PSD matrix can exceed the diagnostic's 1e-10 threshold."""
    vector = np.random.default_rng(19).standard_normal(16)
    covariance = np.outer(vector, vector).astype(np.float32).astype(np.float64)
    eigenvalues = np.linalg.eigvalsh(covariance)
    _save(
        "covariance_rounding",
        {
            "min_eigenvalue": float(eigenvalues[0]),
            "max_eigenvalue": float(eigenvalues[-1]),
        },
    )
    with pytest.raises(ValueError, match="not PSD"):
        psd_sqrt(covariance)

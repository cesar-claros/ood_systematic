"""Selective CSF fitting/evaluation pipeline.

This is the active orchestrator for fitting and evaluating CSFs across the
ProjectionFiltering / Temperature / Mahalanobis / etc. families. It supports
two orthogonal filters:

  - `--csfs` / `--skip-csfs` (active set, families in ALL_FAMILIES).
  - `--projections` (modes: plain / global / class / class_pred).

Both are passed through via `active: set` and `projections: set` parameters
on the public functions. When either is None, the default is "everything".

Two internal backbones flow through the active set but are NOT user-facing
in ALL_FAMILIES (users would get an `Unknown CSF family` error from
normalize_family if they listed them):
  - `ProjectionFiltering`: produces the projections consumed by all
    `_global` / `_class` / `_class_pred` CSF variants.
  - `Temperature`: provides the raw softmax scaling.

The pre-split monolith (everything fit unconditionally, no family/projection
filtering, ~1700 lines) lives at `archived/utils_funcs.py` as a historical
reference; it is not imported anywhere in the active tree.

CSV writes in stats() merge with any existing file rather than overwriting,
keyed on the CSF identifier (index for stats; column for confids). This lets
re-runs with a different --csfs / --skip-csfs / --projections subset
preserve other families' results.
"""

import os
from typing import Iterable
import pandas as pd
from fd_shifts.analysis import metrics
from src.csfs import (
    ClassTypicalMatching,
    EntropyScores,
    fDBD,
    GradNorm,
    KernelPCA,
    MahalanobisDistance,
    MahalanobisPP,
    NCI,
    NeCo,
    NNGuide,
    pNML,
    ProjectionFiltering,
    ResidualScore,
    TemperatureScaling,
    ViMScore,
)
from src.neural_collapse import NeuralCollapseMetrics
from src import scores_funcs
from src.rc_stats import RiskCoverageStats
from torch.nn import functional as F
import torch
from fd_shifts import logger
from tqdm import tqdm

# Detect GPU if available
use_cuda = True if torch.cuda.is_available() else False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {DEVICE}...")


# ---------------------------------------------------------------------------
# Family inventory + aliases.
# ---------------------------------------------------------------------------

#: Canonical family names. Each is user-addressable via --csfs / --skip-csfs.
#: NOTE: ProjectionFiltering and Temperature are NOT here — they are internal
#: backbones, not CSFs. PCA_RecError is the actual reconstruction-error CSF
#: that consumes ProjectionFiltering's output (mirroring KPCA_RecError vs
#: the KernelPCA algorithm class).
ALL_FAMILIES: frozenset[str] = frozenset({
    # Parametric CSFs (have save_params files):
    "KPCA_RecError",
    "PCA_RecError",
    "CTM",
    "NNGuide",
    "fDBD",
    "MahalanobisDistance",
    "MahalanobisPP",
    "NCI",
    "pNML",
    "GEN",
    "REN",
    "ViM",
    "Residual",
    "NeCo",
    "NeuralCollapse",
    # Base detectors (no fit; eval only):
    "MSR",
    "MLS",
    "PE",
    "PCE",
    "GE",
    "Energy",
    "GradNorm",
    "MI",
    "Confidence",
})

#: Internal backbone names that flow through the active set but are NOT in
#: ALL_FAMILIES. Users cannot reference them via --csfs / --skip-csfs (they
#: would get an "Unknown CSF family" error); the pipeline manages them
#: automatically.
BACKBONES: frozenset[str] = frozenset({"ProjectionFiltering", "Temperature"})

#: Families whose `_global`/`_class*` variants consume ProjectionFiltering outputs.
#: PCA_RecError IS in this set: it IS the score produced by ProjectionFiltering's
#: own reconstruction error, so it depends on PF being set up.
PF_DEPENDENTS: frozenset[str] = frozenset({
    "CTM", "NNGuide", "fDBD", "MahalanobisDistance", "pNML",
    "GEN", "REN", "MSR", "MLS", "PE", "PCE", "GE", "Energy",
    "GradNorm", "NeuralCollapse",
    "PCA_RecError",
})

#: Lowercased alias → canonical name. Auto-generated for every name in
#: ALL_FAMILIES, plus a few manual conveniences.
_ALIASES: dict[str, str] = {name.lower(): name for name in ALL_FAMILIES}
_ALIASES.update({
    "kpca":                     "KPCA_RecError",
    "kpcarecerror":             "KPCA_RecError",
    "kernelpca":                "KPCA_RecError",
    "pcarecerror":              "PCA_RecError",
    "pca":                      "PCA_RecError",
    "classtypicalmatching":     "CTM",
    "maha":                     "MahalanobisDistance",
    "mahalanobis":              "MahalanobisDistance",
    "mahapp":                   "MahalanobisPP",
    "maha++":                   "MahalanobisPP",
    "mahalanobis++":            "MahalanobisPP",
    "nc":                       "NeuralCollapse",
    "neuralcollapsemetrics":    "NeuralCollapse",
    "vimscore":                 "ViM",
    "residualscore":            "Residual",
})

#: Canonical family → confids-key prefix (None means no confids keys of its
#: own — only the case for `NeuralCollapse`, whose parameter files are
#: consumed by the predictor sub-package, not by stats()).
_FAMILY_TO_PREFIX: dict[str, str | None] = {
    "KPCA_RecError":       "KPCA_RecError",
    "PCA_RecError":        "PCA_RecError",
    "CTM":                 "CTM",
    "NNGuide":             "NNGuide",
    "fDBD":                "fDBD",
    "MahalanobisDistance": "Maha",
    "MahalanobisPP":       "MahaPP",
    "NCI":                 "NCI",
    "pNML":                "pNML",
    "GEN":                 "GEN",
    "REN":                 "REN",
    "ViM":                 "ViM",
    "Residual":            "Residual",
    "NeCo":                "NeCo",
    "MSR":                 "MSR",
    "MLS":                 "MLS",
    "PE":                  "PE",
    "PCE":                 "PCE",
    "GE":                  "GE",
    "Energy":              "Energy",
    "GradNorm":            "GradNorm",
    "MI":                  "MI",
    "Confidence":          "Confidence",
    "NeuralCollapse":      None,
}


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def normalize_family(name: str) -> str:
    """Normalize an alias to the canonical family name (case-insensitive).

    Raises ValueError on an unknown name.
    """
    key = name.strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            f"Unknown CSF family: {name!r}. "
            f"Known families: {sorted(ALL_FAMILIES)}."
        )
    return _ALIASES[key]


def build_active(
    csfs: Iterable[str] | None = None,
    skip_csfs: Iterable[str] | None = None,
) -> set[str]:
    """Resolve --csfs / --skip-csfs into the active family set.

    - Both None: returns ALL_FAMILIES (everything runs).
    - csfs given: returns the listed families, auto-adding backbones
      (Temperature always; ProjectionFiltering if any dependent is active).
    - skip_csfs given: returns ALL_FAMILIES minus the listed ones, with an
      error if any listed family is a backbone (cannot be skipped).
    """
    if csfs is not None and skip_csfs is not None:
        raise ValueError("--csfs and --skip-csfs are mutually exclusive.")
    if csfs is not None:
        active = {normalize_family(c) for c in csfs}
        active.add("Temperature")
        if active & PF_DEPENDENTS:
            active.add("ProjectionFiltering")
        return active
    if skip_csfs is not None:
        skipped = {normalize_family(c) for c in skip_csfs}
        forbidden = skipped & BACKBONES
        if forbidden:
            raise ValueError(
                f"Cannot skip backbone families {sorted(forbidden)}. "
                f"They are required by ~30 other CSFs. "
                f"List the dependent families in --skip-csfs instead."
            )
        return set(ALL_FAMILIES) - skipped
    return set(ALL_FAMILIES)


def is_active(family: str, active: set[str]) -> bool:
    """True if the family is in the active set."""
    return family in active


def any_active(families: Iterable[str], active: set[str]) -> bool:
    """True if any of the given families is in the active set."""
    return any(f in active for f in families)


def _key_in_family_prefixes(key: str, prefixes: set[str]) -> bool:
    """True if `key` belongs to any of the family prefixes.

    Handles the "MCD-" prefix (for dropout-distribution variants) and the
    mode-suffix convention (e.g., "Maha_global" belongs to prefix "Maha").
    """
    base = key[4:] if key.startswith("MCD-") else key
    for prefix in prefixes:
        if base == prefix or base.startswith(prefix + "_"):
            return True
    return False


def _needs_pf(active: set[str]) -> bool:
    """True if any family that actually consumes ProjectionFiltering output
    is active. KernelPCA is NOT a PF consumer (it uses raw features); only
    `PF_DEPENDENTS` families plus an explicit request for the
    `ProjectionFiltering` family trigger PF setup.
    """
    return bool(active & PF_DEPENDENTS) or "ProjectionFiltering" in active


def filter_confids(confids: dict, active: set[str], projections: set[str] | None = None) -> dict:
    """Return a new dict containing only the confids whose keys belong to active families
    AND whose mode (plain/global/class/class_pred) is in `projections`.

    If projections is None, all modes are accepted.
    """
    active_prefixes = {p for f in active for p in (_FAMILY_TO_PREFIX.get(f),) if p is not None}
    proj = set(projections) if projections is not None else set(ALL_PROJECTIONS)
    return {
        k: v for k, v in confids.items()
        if _key_in_family_prefixes(k, active_prefixes) and _detect_mode(k) in proj
    }


class _MissingCSF:
    """Placeholder for a CSF not loaded (because its family was skipped).

    Any attribute access returns the same _MissingCSF instance, so chained
    accesses like `score_methods['kpca_global'].get_scores` don't raise.
    Calling the result returns ``None`` — confids entries that involve a
    missing CSF become ``None`` in the dict, and ``filter_confids`` then
    drops those keys (matched by family prefix) before the stats DataFrame
    is built.

    Known limit: ``scores_funcs.mcd_expected_function`` calls the wrapped
    function per MCD sample and stacks the outputs via ``torch.vstack`` —
    stacking ``None`` raises. So skipping a family on a do=1 (MCD) model
    will still crash on its ``MCD-E*`` confids entries; the fix for the
    MCD path would need per-family gating of the confids construction.
    """
    __slots__ = ()
    def __getattr__(self, name):
        return self
    def __call__(self, *args, **kwargs):
        return None


#: Module-level singleton used as the placeholder for skipped CSFs.
_MISSING_CSF = _MissingCSF()


def _merge_csv_rows(new_df: pd.DataFrame, path: str) -> pd.DataFrame:
    """Merge new_df (index = method name) into the existing CSV at `path`.

    Rows whose index is in new_df.index are REPLACED; others are RETAINED.
    """
    if os.path.exists(path):
        old = pd.read_csv(path, index_col=0)
        retained = old.loc[~old.index.isin(new_df.index)]
        return pd.concat([retained, new_df])
    return new_df


def _merge_csv_cols(new_df: pd.DataFrame, path: str) -> pd.DataFrame:
    """Merge new_df (columns = method names + 'residuals') into existing CSV at `path`.

    Columns whose name is in new_df.columns (or is 'residuals') are REPLACED.
    """
    if os.path.exists(path):
        old = pd.read_csv(path, index_col=0)
        new_cols = set(new_df.columns)
        retained = old.drop(
            columns=[c for c in old.columns if c in new_cols or c == "residuals"],
            errors="ignore",
        )
        return pd.concat([retained, new_df], axis=1)
    return new_df


# ---------------------------------------------------------------------------
# Model-health diagnostics.
# ---------------------------------------------------------------------------

def check_model_health(model_evaluations: dict, cf) -> list[str]:
    """Inspect the validation outputs and flag classic degeneracy modes.

    Returns a list of human-readable warning strings (empty if healthy).
    Detects:
      - Validation accuracy at or near chance level (model didn't learn).
      - Prediction collapse (>=95% of argmaxes go to a single class) — the
        Deep-Gamblers always-abstain failure mode when the reward is too low.
      - Near-constant penultimate-layer features (mean per-dim std below
        1e-4) — kernel-based CSFs (KernelPCA, NeCo, ...) will fail on these.
      - NaN / Inf in logits or features.

    The function is read-only and cheap (a few tensor reductions on the val
    split). Call it once at the top of csf_fit.py before any CSF fitting.
    """
    warnings: list[str] = []
    val = model_evaluations.get("val")
    if val is None:
        return warnings
    softmax = val.get("softmax")
    correct = val.get("correct")
    encoded = val.get("encoded")
    logits = val.get("logits")
    if softmax is None or correct is None:
        return warnings

    # 1. Validation accuracy at chance.
    num_classes = getattr(cf.data, "num_classes", softmax.shape[1])
    chance = 1.0 / num_classes
    val_acc = correct.float().mean().item()
    if val_acc < 2.0 * chance:
        warnings.append(
            f"Validation accuracy {val_acc:.3f} is at or below 2x chance "
            f"({chance:.3f}); the model probably didn't learn. Downstream "
            f"CSFs will be unreliable."
        )

    # 2. Prediction collapse: most argmaxes concentrate in one class.
    preds = softmax.max(dim=1).indices
    pred_counts = torch.bincount(preds, minlength=softmax.shape[1])
    n = preds.shape[0]
    top_class = pred_counts.argmax().item()
    top_frac = pred_counts.max().item() / max(n, 1)
    if top_frac >= 0.95:
        warnings.append(
            f"{top_frac:.1%} of validation predictions go to a single class "
            f"(class index {top_class}); the model has collapsed to a constant "
            f"output. For Deep Gamblers, this typically means the abstention "
            f"reward was too low and training converged to always-abstain."
        )

    # 3. Near-constant features.
    if encoded is not None:
        feature_std = encoded.float().std(dim=0).mean().item()
        if feature_std < 1e-4:
            warnings.append(
                f"Mean per-dim std of penultimate features on the validation "
                f"set is {feature_std:.2e} (near-zero). Kernel-based CSFs "
                f"(KernelPCA, NeCo, ...) will produce rank-deficient kernel "
                f"matrices and likely crash."
            )

    # 4. NaN / Inf in logits or features.
    for name, t in (("logits", logits), ("encoded features", encoded)):
        if t is None:
            continue
        if torch.isnan(t).any().item():
            warnings.append(f"NaNs detected in validation {name}.")
        if torch.isinf(t).any().item():
            warnings.append(f"Infs detected in validation {name}.")

    return warnings


# ---------------------------------------------------------------------------
# Projection / Temperature fitting helpers (csf_fit-side responsibility).
# ---------------------------------------------------------------------------

#: Projection modes addressable via csf_fit.py --projections.
#: 'plain' = CSFs that operate on raw (unprojected) features/logits;
#: 'global', 'class', 'class_pred' = CSFs that operate on ProjectionFiltering outputs.
ALL_PROJECTIONS: frozenset[str] = frozenset({"plain", "global", "class", "class_pred"})


def _detect_mode(key: str) -> str:
    """Determine the projection mode from a confids dict key.

    Strips any leading 'MCD-' marker. Match is longest-prefix:
    'class_pred' must be tested before 'class'.
    """
    base = key[4:] if key.startswith("MCD-") else key
    if "_class_pred" in base:
        return "class_pred"
    if "_class" in base:
        return "class"
    if "_global" in base:
        return "global"
    return "plain"


def _params_path(cf, filename: str) -> str:
    """Resolve the on-disk path for a save_params filename (without .pt)."""
    return f"{cf.exp.dir}/params/{filename}.pt"


def _load_or_fit_temperature(cf, model_opts, suffix, logits, labels):
    """Load Temperature_<suffix>_params if the file exists, else fit and save.

    Pass suffix=None for the raw Temperature (file: Temperature_params).
    """
    fname = f"Temperature_{suffix}_params" if suffix else "Temperature_params"
    temp = TemperatureScaling(cf)
    if os.path.exists(_params_path(cf, fname + model_opts)):
        temp.load_params(filename=fname + model_opts)
        logger.info(f"Loaded existing {fname}{model_opts}")
    else:
        temp.compute_temperature(logits, labels)
        temp.save_params(filename=fname + model_opts)
    return temp


def _load_or_fit_pf(cf, module, study_name, mode, model_opts, mcd,
                    encoded_train, encoded_val, residuals_val, labels_train):
    """Load ProjectionFiltering_<mode>[_distribution]_params if it exists, else fit and save."""
    suffix = "_distribution" if mcd else ""
    fname = f"ProjectionFiltering_{mode}{suffix}_params"
    pf = ProjectionFiltering(module, study_name, cf, mode=mode)
    if os.path.exists(_params_path(cf, fname + model_opts)):
        pf.load_params(filename=fname + model_opts)
        logger.info(f"Loaded existing {fname}{model_opts}")
    else:
        pf.tune_hyperparameters(
            encoded_train, encoded_val, residuals_val,
            labels_train=labels_train, only_correct=True,
        )
        pf.save_params(filename=fname + model_opts)
    return pf


def fit_projections(cf, module, study_name, model_evaluations, do_enabled,
                    model_opts, temp_scaled, projections, active=None):
    """Fit ProjectionFiltering instances and their per-projection Temperature scalings.

    Called from csf_fit.py BEFORE run_score_methods. Each PF / Temperature
    pair is load-if-exists, fit-if-not. PF_class is shared between 'class'
    and 'class_pred' projections; if either is requested, PF_class is
    fit/loaded once.

    Parameters
    ----------
    projections : Iterable[str]
        Subset of ALL_PROJECTIONS. Empty set = no projections (only the raw
        Temperature_params is needed; csf_fit.py handles that separately).
    """
    projections = set(projections)
    invalid = projections - ALL_PROJECTIONS
    if invalid:
        raise ValueError(
            f"Unknown projections {sorted(invalid)}. "
            f"Allowed: {sorted(ALL_PROJECTIONS)}."
        )
    # 'plain' is a no-op for fit_projections (no PF, raw Temperature handled in csf_fit).
    projections = projections - {"plain"}
    if not projections:
        return
    # Skip PF setup entirely if no active family consumes PF output (e.g.,
    # `--csfs KernelPCA --projections global` — KernelPCA uses raw features).
    if active is not None and not _needs_pf(set(active)):
        logger.info(
            "Skipping ProjectionFiltering fits: no active family consumes "
            "PF outputs (active=%s).", sorted(active),
        )
        return
    # If we reach here, _needs_pf was True (or no active set was given, in which
    # case the caller wants to fit everything). The pf_needed local supports
    # the same gate that we use in run_score_methods / load_score_methods / stats.
    pf_needed = True

    encoded_train = model_evaluations["train"]["encoded"]
    encoded_val = model_evaluations["val"]["encoded"]
    labels_train = model_evaluations["train"]["labels"]
    labels_val = model_evaluations["val"]["labels"]
    correct_val = model_evaluations["val"]["correct"]

    # ---- Deterministic branch ----
    if "global" in projections and pf_needed:
        pf_global = _load_or_fit_pf(
            cf, module, study_name, "global", model_opts, mcd=False,
            encoded_train=encoded_train, encoded_val=encoded_val,
            residuals_val=1 - correct_val, labels_train=labels_train,
        )
        logits_global_val = pf_global.get_logits(encoded_val)
        _load_or_fit_temperature(cf, model_opts, "global", logits_global_val, labels_val)

    if (projections & {"class", "class_pred"}) and pf_needed:
        pf_class = _load_or_fit_pf(
            cf, module, study_name, "class", model_opts, mcd=False,
            encoded_train=encoded_train, encoded_val=encoded_val,
            residuals_val=1 - correct_val, labels_train=labels_train,
        )
        if "class" in projections:
            logits_class_val = pf_class.get_logits(encoded_val)
            _load_or_fit_temperature(cf, model_opts, "class", logits_class_val, labels_val)
        if "class_pred" in projections:
            softmax_val = (
                model_evaluations["val"]["softmax_scaled"] if temp_scaled
                else model_evaluations["val"]["softmax"]
            )
            preds_val = softmax_val.max(dim=1).indices
            encoded_class_val = pf_class.get_backprojection(encoded_val)
            _, logits_class_pred_val = pf_class.get_combined_backprojection(
                encoded_class_val, combine="prediction", preds=preds_val,
            )
            _load_or_fit_temperature(
                cf, model_opts, "class_pred", logits_class_pred_val, labels_val,
            )

    # ---- MCD branch ----
    if not do_enabled:
        return

    encoded_dist_train_mean = model_evaluations["train"]["encoded_dist"].mean(dim=2)
    encoded_dist_val_mean = model_evaluations["val"]["encoded_dist"].mean(dim=2)
    correct_mcd_val = model_evaluations["val"]["correct_mcd"]

    if "global" in projections and pf_needed:
        pf_global_dist = _load_or_fit_pf(
            cf, module, study_name, "global", model_opts, mcd=True,
            encoded_train=encoded_dist_train_mean, encoded_val=encoded_dist_val_mean,
            residuals_val=1 - correct_mcd_val, labels_train=labels_train,
        )
        logits_global_dist_val = pf_global_dist.get_logits(encoded_dist_val_mean)
        _load_or_fit_temperature(
            cf, model_opts, "global_distribution", logits_global_dist_val, labels_val,
        )

    if (projections & {"class", "class_pred"}) and pf_needed:
        pf_class_dist = _load_or_fit_pf(
            cf, module, study_name, "class", model_opts, mcd=True,
            encoded_train=encoded_dist_train_mean, encoded_val=encoded_dist_val_mean,
            residuals_val=1 - correct_mcd_val, labels_train=labels_train,
        )
        if "class" in projections:
            logits_class_dist_val = pf_class_dist.get_logits(encoded_dist_val_mean)
            _load_or_fit_temperature(
                cf, model_opts, "class_distribution", logits_class_dist_val, labels_val,
            )
        if "class_pred" in projections:
            softmax_dist_val = (
                model_evaluations["val"]["softmax_scaled_dist"] if temp_scaled
                else model_evaluations["val"]["softmax_dist"]
            )
            preds_dist_val = softmax_dist_val.mean(dim=2).max(dim=1).indices
            encoded_class_dist_val = pf_class_dist.get_backprojection(encoded_dist_val_mean)
            _, logits_class_pred_dist_val = pf_class_dist.get_combined_backprojection(
                encoded_class_dist_val, combine="prediction", preds=preds_dist_val,
            )
            _load_or_fit_temperature(
                cf, model_opts, "class_pred_distribution",
                logits_class_pred_dist_val, labels_val,
            )


def run_score_methods(cf, module, study_name, model_evaluations, do_enabled:bool, model_opts:str='', temp_scaled:bool=False, active: set | None = None, projections: set | None = None):
    active = set(ALL_FAMILIES) if active is None else set(active)
    projections = set(ALL_PROJECTIONS) if projections is None else set(projections)
    pf_needed = _needs_pf(active)
    def gate(mode, family):
        return mode in projections and family in active
    # Temperature 
    temperature_scale = TemperatureScaling(cf)
    temperature_scale.load_params(filename='Temperature_params'+model_opts)
    #
    labels_train = model_evaluations['train']['labels']
    labels_val = model_evaluations['val']['labels']
    # Train evaluations
    encoded_train = model_evaluations['train']['encoded']
    logits_train = model_evaluations['train']['logits']
    encoded_val = model_evaluations['val']['encoded']
    correct_val = model_evaluations['val']['correct']
    # Neural Collapse Metrics
    if gate("plain", "NeuralCollapse"):
        neural_collapse = NeuralCollapseMetrics(module, study_name, cf)
        neural_collapse.compute_NeuralCollapse_params(encoded_train, labels_train)
        neural_collapse.save_params(filename='NeuralCollapse_params'+model_opts)
    # Global
    # Kernel PCA Global
    if gate("global", "KPCA_RecError"):
        kpca_global = KernelPCA(module, study_name, cf, mode='global')
        kpca_global.tune_hyperparameters(encoded_train, encoded_val, 1-correct_val, 
                                            labels_train=labels_train, only_correct=True, 
                                            temperature=temperature_scale.temperature, 
                                            center_on='all', kernel='rbf')
        kpca_global.save_params(filename='KernelPCA_global_params'+model_opts)
    # Projection Filtering Global (only loaded if any global-mode CSF is requested).
    if "global" in projections and pf_needed:
        projection_filtering_global = ProjectionFiltering(module, study_name, cf, mode='global')
        projection_filtering_global.load_params(filename='ProjectionFiltering_global_params'+model_opts)
        logits_global_train = projection_filtering_global.get_logits(encoded_train)
        # Backprojections for global
        encoded_global_train = projection_filtering_global.get_backprojection(encoded_train)
        encoded_global_val = projection_filtering_global.get_backprojection(encoded_val)
    # Neural Collapse Global Metrics
    if gate("global", "NeuralCollapse"):
        neural_collapse_global = NeuralCollapseMetrics(module, study_name, cf)
        neural_collapse_global.compute_NeuralCollapse_params(encoded_global_train, labels_train)
        neural_collapse_global.save_params(filename='NeuralCollapse_global_params'+model_opts)
    # Class Typical Matching Global
    if gate("global", "CTM"):
        ctm_global = ClassTypicalMatching(module, study_name, cf, mode='global')
        ctm_global.compute_CTM_params(encoded_global_train, labels_train)
        ctm_global.save_params(filename='CTM_global_params'+model_opts)
        del ctm_global
    # NNGuide PCA global
    if gate("global", "NNGuide"):
        nnguide_global = NNGuide(module,study_name,cf)
        nnguide_global.tune_hyperparameters(encoded_global_train, encoded_global_val, 1-correct_val,
                                            labels_train = labels_train, logits_train= logits_global_train,)
        nnguide_global.save_params(filename='NNGuide_global_params'+model_opts)
        del nnguide_global
    # fDBD PCA global
    if gate("global", "fDBD"):
        fDBD_global = fDBD(module,study_name,cf)
        fDBD_global.compute_fDBD_params(encoded_global_train)
        fDBD_global.save_params(filename='fDBD_global_params'+model_opts)
        del fDBD_global
    # Mahalanobis distance global
    if gate("global", "MahalanobisDistance"):
        maha_distance_global = MahalanobisDistance(cf) 
        maha_distance_global.compute_MahaDist_params(encoded_global_train, labels_train)
        maha_distance_global.save_params(filename='MahalanobisDistance_global_params'+model_opts)
        del maha_distance_global
    # pNML global
    if gate("global", "pNML"):
        pnml_global = pNML(module,study_name,cf)
        pnml_global.compute_pNML_params(encoded_global_train)
        pnml_global.save_params(filename='pNML_global_params'+model_opts)
        del pnml_global
    if "global" in projections and pf_needed:
        del encoded_global_train
        logits_global_val = projection_filtering_global.get_logits(encoded_val)
        del projection_filtering_global
        # Temperature Global
        temperature_global = TemperatureScaling(cf)
        temperature_global.load_params(filename='Temperature_global_params'+model_opts)
        # Softmax global
        softmax_global_val = temperature_global.get_scaled_softmax(logits_global_val) if temp_scaled else F.softmax(logits_global_val, dim=1, dtype=torch.float64)
        del logits_global_val
        del temperature_global
    # Generalized entropy Global
    if gate("global", "GEN"):
        generalized_entropy_global = EntropyScores(cf, 'generalized')
        generalized_entropy_global.compute_entropy_params(softmax_global_val, 1-correct_val)
        generalized_entropy_global.save_params(filename='GEN_global_params'+model_opts)
        del generalized_entropy_global
    # Renyi entropy Global
    if gate("global", "REN"):
        renyi_entropy_global = EntropyScores(cf, 'renyi')
        renyi_entropy_global.compute_entropy_params(softmax_global_val, 1-correct_val)
        renyi_entropy_global.save_params(filename='REN_global_params'+model_opts)
        del renyi_entropy_global
    if "global" in projections and pf_needed:
        del softmax_global_val
    # Kernel PCA Class
    if gate("class", "KPCA_RecError"):
        kpca_class = KernelPCA(module, study_name, cf, mode='class')
        kpca_class.tune_hyperparameters(encoded_train, encoded_val, 1-correct_val, 
                                            labels_train=labels_train, only_correct=True, 
                                            temperature=temperature_scale.temperature, 
                                            center_on='all', kernel='rbf')
        kpca_class.save_params(filename='KernelPCA_class_params'+model_opts)
    # Projection Filtering Class (only loaded if any class- or class_pred-mode CSF is requested).
    if ("class" in projections or "class_pred" in projections) and pf_needed:
        projection_filtering_class = ProjectionFiltering(module, study_name, cf, mode='class')
        projection_filtering_class.load_params(filename='ProjectionFiltering_class_params'+model_opts)
        logits_class_train = projection_filtering_class.get_logits(encoded_train)
        # Backprojections for class
        encoded_class_train = projection_filtering_class.get_backprojection(encoded_train)
        encoded_class_val = projection_filtering_class.get_backprojection(encoded_val)

        # Backprojections for class w/predictions
        softmax_train = model_evaluations['train']['softmax_scaled'] if temp_scaled else model_evaluations['train']['softmax']
        preds_train = softmax_train.max(dim=1).indices
        softmax_val = model_evaluations['val']['softmax_scaled'] if temp_scaled else model_evaluations['val']['softmax']
        preds_val = softmax_val.max(dim=1).indices
        del softmax_train
        encoded_class_pred_train, logits_class_pred_train = projection_filtering_class.get_combined_backprojection(encoded_class_train, combine='prediction', preds=preds_train)
        encoded_class_pred_val, logits_class_pred_val = projection_filtering_class.get_combined_backprojection(encoded_class_val, combine='prediction', preds=preds_val)
    # Neural Collapse Global Metrics
    if gate("class_pred", "NeuralCollapse"):
        neural_collapse_class_pred = NeuralCollapseMetrics(module, study_name, cf)
        neural_collapse_class_pred.compute_NeuralCollapse_params(encoded_class_pred_train, labels_train)
        neural_collapse_class_pred.save_params(filename='NeuralCollapse_class_pred_params'+model_opts)
    # Class Typical Matching Class
    if gate("class", "CTM"):
        ctm_class = ClassTypicalMatching(module, study_name, cf, mode='class')
        ctm_class.compute_CTM_params(encoded_class_train, labels_train)
        ctm_class.save_params(filename='CTM_class_params'+model_opts)
        del ctm_class
    if "class_pred" in projections:
        # Temperature Class Pred
        temperature_class_pred = TemperatureScaling(cf)
        temperature_class_pred.load_params(filename='Temperature_class_pred_params'+model_opts)
        # Softmax from filtered logits
        softmax_class_pred_val = temperature_class_pred.get_scaled_softmax(logits_class_pred_val) if temp_scaled else F.softmax(logits_class_pred_val, dim=1, dtype=torch.float64)
        del logits_class_pred_val
        del temperature_class_pred
    # Generalized entropy Class Pred
    if gate("class_pred", "GEN"):
        generalized_entropy_class_pred = EntropyScores(cf, 'generalized')
        generalized_entropy_class_pred.compute_entropy_params(softmax_class_pred_val, 1-correct_val)
        generalized_entropy_class_pred.save_params(filename='GEN_class_pred_params'+model_opts)
        del generalized_entropy_class_pred
    # Renyi entropy Class Pred
    if gate("class_pred", "REN"):
        renyi_entropy_class_pred = EntropyScores(cf, 'renyi')
        renyi_entropy_class_pred.compute_entropy_params(softmax_class_pred_val, 1-correct_val)
        renyi_entropy_class_pred.save_params(filename='REN_class_pred_params'+model_opts)
        del renyi_entropy_class_pred
    if "class_pred" in projections:
        del softmax_class_pred_val
    #
    # Class Typical Matching Class w/predictions
    if gate("class_pred", "CTM"):
        ctm_class_pred = ClassTypicalMatching(module, study_name, cf, mode='global')
        ctm_class_pred.compute_CTM_params(encoded_class_pred_train, labels_train)
        ctm_class_pred.save_params(filename='CTM_class_pred_params'+model_opts)
        del ctm_class_pred
    if ("class" in projections or "class_pred" in projections) and pf_needed:
        del encoded_class_train
    #
    #
    # NNGuide PCA class w/predictions
    if gate("class_pred", "NNGuide"):
        nnguide_class_pred = NNGuide(module,study_name,cf)
        nnguide_class_pred.tune_hyperparameters(encoded_class_pred_train, encoded_class_pred_val, 1-correct_val,
                                            labels_train = labels_train, logits_train= logits_class_train,)
        nnguide_class_pred.save_params(filename='NNGuide_class_pred_params'+model_opts)
        del nnguide_class_pred
    # fDBD PCA class w/predictions
    if gate("class_pred", "fDBD"):
        fDBD_class_pred = fDBD(module,study_name,cf)
        fDBD_class_pred.compute_fDBD_params(encoded_class_pred_train)
        fDBD_class_pred.save_params(filename='fDBD_class_pred_params'+model_opts)
        del fDBD_class_pred
    # Mahalanobis distance class w/predictions
    if gate("class_pred", "MahalanobisDistance"):
        maha_distance_class_pred = MahalanobisDistance(cf) 
        maha_distance_class_pred.compute_MahaDist_params(encoded_class_pred_train, labels_train)
        maha_distance_class_pred.save_params(filename='MahalanobisDistance_class_pred_params'+model_opts)
        del maha_distance_class_pred
    # pNML class w/predictions
    if gate("class_pred", "pNML"):
        pnml_class_pred = pNML(module,study_name,cf)
        pnml_class_pred.compute_pNML_params(encoded_class_pred_train)
        pnml_class_pred.save_params(filename='pNML_class_pred_params'+model_opts)
        del pnml_class_pred
    if "class_pred" in projections:
        del encoded_class_pred_train
    if "class" in projections:
        # Validation set logits from backprojections
        logits_class_val = projection_filtering_class.get_logits(encoded_val)
        # Temperature Class
        temperature_class = TemperatureScaling(cf)
        temperature_class.load_params(filename='Temperature_class_params'+model_opts)
        # Softmax from filtered logits
        softmax_class_val = temperature_class.get_scaled_softmax(logits_class_val) if temp_scaled else F.softmax(logits_class_val, dim=1, dtype=torch.float64)
        del logits_class_val
        del temperature_class
    if ("class" in projections or "class_pred" in projections) and pf_needed:
        del projection_filtering_class
    # Generalized entropy Class 
    if gate("class", "GEN"):
        generalized_entropy_class = EntropyScores(cf, 'generalized')
        generalized_entropy_class.compute_entropy_params(softmax_class_val, 1-correct_val)
        generalized_entropy_class.save_params(filename='GEN_class_params'+model_opts)
        del generalized_entropy_class
    # Renyi entropy Class
    if gate("class", "REN"):
        renyi_entropy_class = EntropyScores(cf, 'renyi')
        renyi_entropy_class.compute_entropy_params(softmax_class_val, 1-correct_val)
        renyi_entropy_class.save_params(filename='REN_class_params'+model_opts)
        del renyi_entropy_class
    if "class" in projections:
        del softmax_class_val
    if ("class" in projections or "class_pred" in projections) and pf_needed:
        del logits_class_train
    # Validation evaluations
    softmax_val = model_evaluations['val']['softmax_scaled'] if temp_scaled else model_evaluations['val']['softmax'] 
    # Class Typical Matching
    if gate("plain", "CTM"):
        ctm = ClassTypicalMatching(module, study_name, cf, mode='global')
        ctm.compute_CTM_params(encoded_train, labels_train)
        ctm.save_params(filename='CTM_params'+model_opts)
        del ctm
    # Class Typical Matching for correct only
    if gate("plain", "CTM"):
        ctm_oc = ClassTypicalMatching(module, study_name, cf, mode='global')
        ctm_oc.compute_CTM_params(encoded_train, labels_train, only_correct=True)
        ctm_oc.save_params(filename='CTM_oc_params'+model_opts)
        del ctm_oc
    # Generalized entropy
    if gate("plain", "GEN"):
        generalized_entropy = EntropyScores(cf, 'generalized')
        generalized_entropy.compute_entropy_params(softmax_val, 1-correct_val)
        generalized_entropy.save_params(filename='GEN_params'+model_opts)
        del generalized_entropy
    # Renyi entropy
    if gate("plain", "REN"):
        renyi_entropy = EntropyScores(cf, 'renyi')
        renyi_entropy.compute_entropy_params(softmax_val, 1-correct_val)
        renyi_entropy.save_params(filename='REN_params'+model_opts)
        del renyi_entropy
    # NNGuide
    if gate("plain", "NNGuide"):
        nnguide = NNGuide(module,study_name,cf)
        nnguide.tune_hyperparameters(encoded_train, encoded_val, 1-correct_val,
                                            labels_train = labels_train, logits_train= logits_train,)
        nnguide.save_params(filename='NNGuide_params'+model_opts)
        del nnguide
    # fDBD
    if gate("plain", "fDBD"):
        fdbd_inst = fDBD(module,study_name,cf)
        fdbd_inst.compute_fDBD_params(encoded_train)
        fdbd_inst.save_params(filename='fDBD_params'+model_opts)
        del fdbd_inst
    # Mahalanobis distance
    if gate("plain", "MahalanobisDistance"):
        maha_distance = MahalanobisDistance(cf)
        maha_distance.compute_MahaDist_params(encoded_train, labels_train)
        maha_distance.save_params(filename='MahalanobisDistance_params'+model_opts)
        del maha_distance
    # Mahalanobis++ (L2-normalized features)
    if gate("plain", "MahalanobisPP"):
        maha_pp = MahalanobisPP(cf)
        maha_pp.compute_MahaDist_params(encoded_train, labels_train)
        maha_pp.save_params(filename='MahalanobisPP_params'+model_opts)
        del maha_pp
    # NCI (neural-collapse-inspired; alpha selected on validation)
    if gate("plain", "NCI"):
        nci_score = NCI(module, study_name, cf)
        nci_score.compute_NCI_params(
            encoded_train,
            activations_val=encoded_val,
            logits_val=model_evaluations['val']['logits'],
            correct_val=correct_val,
        )
        nci_score.save_params(filename='NCI_params'+model_opts)
        del nci_score
    # pNML 
    if gate("plain", "pNML"):
        pnml = pNML(module,study_name,cf)
        pnml.compute_pNML_params(encoded_train)
        pnml.save_params(filename='pNML_params'+model_opts)
        del pnml
    # ViM Score 
    if gate("plain", "ViM"):
        vim = ViMScore(module,study_name,cf)
        vim.compute_ViM_params(encoded_train)
        vim.save_params(filename='ViM_params'+model_opts)
        del vim
    # Residual Score 
    if gate("plain", "Residual"):
        residual = ResidualScore(module,study_name,cf)
        residual.compute_Residual_params(encoded_train)
        residual.save_params(filename='Residual_params'+model_opts)
        del residual
    # NeCo Score 
    if gate("plain", "NeCo"):
        neco = NeCo(module,study_name,cf)
        neco.compute_NeCo_params(encoded_train)
        neco.save_params(filename='NeCo_params'+model_opts)
        del neco
    del encoded_train

    if do_enabled:
        # Temperature distribution 
        temperature_scale_dist = TemperatureScaling(cf)
        temperature_scale_dist.load_params(filename='Temperature_distribution_params'+model_opts)
        #
        encoded_dist_train = model_evaluations['train']['encoded_dist']
        logits_dist_train = model_evaluations['train']['logits_dist']
        encoded_dist_val = model_evaluations['val']['encoded_dist']
        correct_mcd_val = model_evaluations['val']['correct_mcd']
        # Neural Collapse Metrics for Distribution
        if gate("plain", "NeuralCollapse"):
            neural_collapse_dist = NeuralCollapseMetrics(module, study_name, cf)
            neural_collapse_dist.compute_NeuralCollapse_params(encoded_dist_train.mean(dim=2), labels_train)
            neural_collapse_dist.save_params(filename='NeuralCollapse_distribution_params'+model_opts)
        # Global
        # Kernel PCA Global
        if gate("global", "KPCA_RecError"):
            kpca_global_dist = KernelPCA(module, study_name, cf, mode='global')
            kpca_global_dist.tune_hyperparameters(encoded_dist_train.mean(dim=2), encoded_dist_val.mean(dim=2), 1-correct_mcd_val, 
                                                labels_train=labels_train, only_correct=True, 
                                                temperature=temperature_scale_dist.temperature, 
                                                center_on='all', kernel='rbf')
            kpca_global_dist.save_params(filename='KernelPCA_global_distribution_params'+model_opts)
        # Projection Filtering Global for distribution (only loaded if any global-mode CSF is requested).
        if "global" in projections and pf_needed:
            projection_filtering_global_dist = ProjectionFiltering(module, study_name, cf, mode='global')
            projection_filtering_global_dist.load_params(filename='ProjectionFiltering_global_distribution_params'+model_opts)
            logits_global_dist_train = projection_filtering_global_dist.get_logits(encoded_dist_train.mean(dim=2))
            # Backprojections global for distribution
            encoded_global_dist_train = projection_filtering_global_dist.get_backprojection(encoded_dist_train.mean(dim=2))
            encoded_global_dist_val = projection_filtering_global_dist.get_backprojection(encoded_dist_val.mean(dim=2))
        # Neural Collapse Global Metrics for Distribution
        if gate("global", "NeuralCollapse"):
            neural_collapse_global_dist = NeuralCollapseMetrics(module, study_name, cf)
            neural_collapse_global_dist.compute_NeuralCollapse_params(encoded_global_dist_train, labels_train)
            neural_collapse_global_dist.save_params(filename='NeuralCollapse_global_distribution_params'+model_opts)
        # Class Typical Matching Global for distribution
        if gate("global", "CTM"):
            ctm_global_dist = ClassTypicalMatching(module, study_name, cf, mode='global')
            ctm_global_dist.compute_CTM_params(encoded_global_dist_train, labels_train)
            ctm_global_dist.save_params(filename='CTM_global_distribution_params'+model_opts)
            del ctm_global_dist
        # NNGuide PCA global for distribution
        if gate("global", "NNGuide"):
            nnguide_global_dist = NNGuide(module,study_name,cf)
            nnguide_global_dist.tune_hyperparameters(encoded_global_dist_train, encoded_global_dist_val, 1-correct_mcd_val,
                                            labels_train = labels_train, logits_train= logits_global_dist_train,)
            nnguide_global_dist.save_params(filename='NNGuide_global_distribution_params'+model_opts)
            del nnguide_global_dist
        # fDBD PCA global for distribution
        if gate("global", "fDBD"):
            fDBD_global_dist = fDBD(module,study_name,cf)
            fDBD_global_dist.compute_fDBD_params(encoded_global_dist_train)
            fDBD_global_dist.save_params(filename='fDBD_global_distribution_params'+model_opts)
            del fDBD_global_dist
        # Mahalanobis distance global for distribution
        if gate("global", "MahalanobisDistance"):
            maha_distance_global_dist = MahalanobisDistance(cf) 
            maha_distance_global_dist.compute_MahaDist_params(encoded_global_dist_train, labels_train)
            maha_distance_global_dist.save_params(filename='MahalanobisDistance_global_distribution_params'+model_opts)
            del maha_distance_global_dist
        # pNML global for distribution
        if gate("global", "pNML"):
            pnml_global_dist = pNML(module,study_name,cf)
            pnml_global_dist.compute_pNML_params(encoded_global_dist_train)
            pnml_global_dist.save_params(filename='pNML_global_distribution_params'+model_opts)
            del pnml_global_dist
        if "global" in projections and pf_needed:
            del encoded_global_dist_train
            logits_global_dist_val = projection_filtering_global_dist.get_logits(encoded_dist_val.mean(dim=2))
            del projection_filtering_global_dist
            # Temperature global for distribution
            temperature_global_dist = TemperatureScaling(cf)
            temperature_global_dist.load_params(filename='Temperature_global_distribution_params'+model_opts)
            # Softmax global for distribution
            softmax_global_dist_val = temperature_global_dist.get_scaled_softmax(logits_global_dist_val) if temp_scaled else F.softmax(logits_global_dist_val, dim=1, dtype=torch.float64)
            del temperature_global_dist
            del logits_global_dist_val
        # Generalized entropy Global for distribution
        if gate("global", "GEN"):
            generalized_entropy_global_dist = EntropyScores(cf, 'generalized')
            generalized_entropy_global_dist.compute_entropy_params(softmax_global_dist_val, 1-correct_mcd_val)
            generalized_entropy_global_dist.save_params(filename='GEN_global_distribution_params'+model_opts)
            del generalized_entropy_global_dist
        # Renyi entropy Global for distribution
        if gate("global", "REN"):
            renyi_entropy_global_dist = EntropyScores(cf, 'renyi')
            renyi_entropy_global_dist.compute_entropy_params(softmax_global_dist_val, 1-correct_mcd_val)
            renyi_entropy_global_dist.save_params(filename='REN_global_distribution_params'+model_opts)
            del renyi_entropy_global_dist
        if "global" in projections and pf_needed:
            del softmax_global_dist_val
        # Kernel PCA Class
        if gate("class", "KPCA_RecError"):
            kpca_class_dist = KernelPCA(module, study_name, cf, mode='class')
            kpca_class_dist.tune_hyperparameters(encoded_dist_train.mean(dim=2), encoded_dist_val.mean(dim=2), 1-correct_mcd_val, 
                                                labels_train=labels_train, only_correct=True, 
                                                temperature=temperature_scale_dist.temperature, 
                                                center_on='all', kernel='rbf')
            kpca_class_dist.save_params(filename='KernelPCA_class_distribution_params'+model_opts)
        # Projection Filtering Class for distribution (only loaded if any class- or class_pred-mode CSF is requested).
        if ("class" in projections or "class_pred" in projections) and pf_needed:
            projection_filtering_class_dist = ProjectionFiltering(module, study_name, cf, mode='class')
            projection_filtering_class_dist.load_params(filename='ProjectionFiltering_class_distribution_params'+model_opts)
            logits_class_dist_train = projection_filtering_class_dist.get_logits(encoded_dist_train.mean(dim=2))
            # Backprojections for class for distribution
            encoded_class_dist_train = projection_filtering_class_dist.get_backprojection(encoded_dist_train.mean(dim=2))
            encoded_class_dist_val = projection_filtering_class_dist.get_backprojection(encoded_dist_val.mean(dim=2))
            # Backprojections for class w/predictions for distribution
            softmax_dist_train = model_evaluations['train']['softmax_scaled_dist'] if temp_scaled else model_evaluations['train']['softmax_dist']
            preds_dist_train = softmax_dist_train.mean(dim=2).max(dim=1).indices
            softmax_dist_val = model_evaluations['val']['softmax_scaled_dist'] if temp_scaled else model_evaluations['val']['softmax_dist']
            preds_dist_val = softmax_dist_val.mean(dim=2).max(dim=1).indices
            del softmax_dist_train
            encoded_class_pred_dist_train, logits_class_pred_dist_train = projection_filtering_class_dist.get_combined_backprojection(encoded_class_dist_train, combine='prediction', preds=preds_dist_train)
            encoded_class_pred_dist_val, logits_class_pred_dist_val = projection_filtering_class_dist.get_combined_backprojection(encoded_class_dist_val, combine='prediction', preds=preds_dist_val)
        # Neural Collapse Global Metrics for Distribution
        if gate("class_pred", "NeuralCollapse"):
            neural_collapse_class_pred_dist = NeuralCollapseMetrics(module, study_name, cf)
            neural_collapse_class_pred_dist.compute_NeuralCollapse_params(encoded_class_pred_dist_train, labels_train)
            neural_collapse_class_pred_dist.save_params(filename='NeuralCollapse_class_pred_distribution_params'+model_opts)
        # Class Typical Matching Class for distribution
        if gate("class", "CTM"):
            ctm_class_dist = ClassTypicalMatching(module, study_name, cf, mode='class')
            ctm_class_dist.compute_CTM_params(encoded_class_dist_train, labels_train)
            ctm_class_dist.save_params(filename='CTM_class_distribution_params'+model_opts)
            del ctm_class_dist
        if "class_pred" in projections:
            # Temperature Class Pred for distribution
            temperature_class_pred_dist = TemperatureScaling(cf)
            temperature_class_pred_dist.load_params(filename='Temperature_class_pred_distribution_params'+model_opts)
            # Softmax from filtered logits
            softmax_class_pred_dist_val = temperature_class_pred_dist.get_scaled_softmax(logits_class_pred_dist_val) if temp_scaled else F.softmax(logits_class_pred_dist_val, dim=1, dtype=torch.float64)
            del logits_class_pred_dist_val
            del temperature_class_pred_dist
        # Generalized entropy Class for distribution
        if gate("class_pred", "GEN"):
            generalized_entropy_class_pred_dist = EntropyScores(cf, 'generalized')
            generalized_entropy_class_pred_dist.compute_entropy_params(softmax_class_pred_dist_val, 1-correct_mcd_val)
            generalized_entropy_class_pred_dist.save_params(filename='GEN_class_pred_distribution_params'+model_opts)
            del generalized_entropy_class_pred_dist
        # Renyi entropy Class for distribution
        if gate("class_pred", "REN"):
            renyi_entropy_class_pred_dist = EntropyScores(cf, 'renyi')
            renyi_entropy_class_pred_dist.compute_entropy_params(softmax_class_pred_dist_val, 1-correct_mcd_val)
            renyi_entropy_class_pred_dist.save_params(filename='REN_class_pred_distribution_params'+model_opts)
            del renyi_entropy_class_pred_dist
        if "class_pred" in projections:
            del softmax_class_pred_dist_val
        #
        # Class Typical Matching Class w/predictions for distribution
        if gate("class_pred", "CTM"):
            ctm_class_pred_dist = ClassTypicalMatching(module, study_name, cf, mode='global')
            ctm_class_pred_dist.compute_CTM_params(encoded_class_pred_dist_train, labels_train)
            ctm_class_pred_dist.save_params(filename='CTM_class_pred_distribution_params'+model_opts)
            del ctm_class_pred_dist
        if ("class" in projections or "class_pred" in projections) and pf_needed:
            del encoded_class_dist_train
        # NNGuide PCA class w/predictions for distribution
        if gate("class_pred", "NNGuide"):
            nnguide_class_pred_dist = NNGuide(module,study_name,cf)
            nnguide_class_pred_dist.tune_hyperparameters(encoded_class_pred_dist_train, encoded_class_pred_dist_val, 1-correct_mcd_val,
                                            labels_train = labels_train, logits_train= logits_class_dist_train,)
            nnguide_class_pred_dist.save_params(filename='NNGuide_class_pred_distribution_params'+model_opts)
            del nnguide_class_pred_dist
        # fDBD PCA class w/predictions for distribution
        if gate("class_pred", "fDBD"):
            fDBD_class_pred_dist = fDBD(module,study_name,cf)
            fDBD_class_pred_dist.compute_fDBD_params(encoded_class_pred_dist_train)
            fDBD_class_pred_dist.save_params(filename='fDBD_class_pred_distribution_params'+model_opts)
            del fDBD_class_pred_dist
        # Mahalanobis distance class w/predictions
        if gate("class_pred", "MahalanobisDistance"):
            maha_distance_class_pred_dist = MahalanobisDistance(cf) 
            maha_distance_class_pred_dist.compute_MahaDist_params(encoded_class_pred_dist_train, labels_train)
            maha_distance_class_pred_dist.save_params(filename='MahalanobisDistance_class_pred_distribution_params'+model_opts)
            del maha_distance_class_pred_dist
        # pNML class w/predictions
        if gate("class_pred", "pNML"):
            pnml_class_pred_dist = pNML(module,study_name,cf)
            pnml_class_pred_dist.compute_pNML_params(encoded_class_pred_dist_train)
            pnml_class_pred_dist.save_params(filename='pNML_class_pred_distribution_params'+model_opts)
            del pnml_class_pred_dist
        if "class_pred" in projections:
            del encoded_class_pred_dist_train
        if "class" in projections:
            # Validation set logits from backprojections (MCD)
            logits_class_dist_val = projection_filtering_class_dist.get_logits(encoded_dist_val.mean(dim=2))
            # Temperature Class for distribution
            temperature_class_dist = TemperatureScaling(cf)
            temperature_class_dist.load_params(filename='Temperature_class_distribution_params'+model_opts)
            # Softmax from filtered logits for distribution
            softmax_class_dist_val = temperature_class_dist.get_scaled_softmax(logits_class_dist_val) if temp_scaled else F.softmax(logits_class_dist_val, dim=1, dtype=torch.float64)
            del temperature_class_dist
            del logits_class_dist_val
        if ("class" in projections or "class_pred" in projections) and pf_needed:
            del projection_filtering_class_dist
        # Generalized entropy Class for distribution
        if gate("class", "GEN"):
            generalized_entropy_class_dist = EntropyScores(cf, 'generalized')
            generalized_entropy_class_dist.compute_entropy_params(softmax_class_dist_val, 1-correct_mcd_val)
            generalized_entropy_class_dist.save_params(filename='GEN_class_distribution_params'+model_opts)
            del generalized_entropy_class_dist
        # Renyi entropy Class for distribution
        if gate("class", "REN"):
            renyi_entropy_class_dist = EntropyScores(cf, 'renyi')
            renyi_entropy_class_dist.compute_entropy_params(softmax_class_dist_val, 1-correct_mcd_val)
            renyi_entropy_class_dist.save_params(filename='REN_class_distribution_params'+model_opts)
            del renyi_entropy_class_dist
        if "class" in projections:
            del softmax_class_dist_val
        if ("class" in projections or "class_pred" in projections) and pf_needed:
            del logits_class_dist_train
        # Validation evaluations for distribution
        softmax_dist_val = model_evaluations['val']['softmax_scaled_dist'] if temp_scaled else model_evaluations['val']['softmax_dist']
        # Class Typical Matching
        if gate("plain", "CTM"):
            ctm_dist = ClassTypicalMatching(module, study_name, cf, mode='global')
            ctm_dist.compute_CTM_params(encoded_dist_train.mean(dim=2), labels_train)
            ctm_dist.save_params(filename='CTM_distribution_params'+model_opts)
            del ctm_dist
        # Class Typical Matching for only correct predictions
        if gate("plain", "CTM"):
            ctm_oc_dist = ClassTypicalMatching(module, study_name, cf, mode='global')
            ctm_oc_dist.compute_CTM_params(encoded_dist_train.mean(dim=2), labels_train, only_correct=True)
            ctm_oc_dist.save_params(filename='CTM_oc_distribution_params'+model_opts)
            del ctm_oc_dist
        # Generalized entropy for distribution
        if gate("plain", "GEN"):
            generalized_entropy_dist = EntropyScores(cf, 'generalized')
            generalized_entropy_dist.compute_entropy_params(softmax_dist_val.mean(dim=2), 1-correct_mcd_val)
            generalized_entropy_dist.save_params(filename='GEN_distribution_params'+model_opts)
            del generalized_entropy_dist
        # Renyi entropy for distribution
        if gate("plain", "REN"):
            renyi_entropy_dist = EntropyScores(cf, 'renyi')
            renyi_entropy_dist.compute_entropy_params(softmax_dist_val.mean(dim=2), 1-correct_mcd_val)
            renyi_entropy_dist.save_params(filename='REN_distribution_params'+model_opts)
            del renyi_entropy_dist
        # del correct_mcd_val
        # NNGuide for distribution
        if gate("plain", "NNGuide"):
            nnguide_dist = NNGuide(module,study_name,cf)
            nnguide_dist.tune_hyperparameters(encoded_dist_train.mean(dim=2), encoded_dist_val.mean(dim=2), 1-correct_mcd_val,
                                            labels_train = labels_train, logits_train= logits_dist_train.mean(dim=2),)
            nnguide_dist.save_params(filename='NNGuide_distribution_params'+model_opts)
            del nnguide_dist
        # fDBD for distribution
        if gate("plain", "fDBD"):
            fDBD_dist = fDBD(module,study_name,cf)
            fDBD_dist.compute_fDBD_params(encoded_dist_train.mean(dim=2))
            fDBD_dist.save_params(filename='fDBD_distribution_params'+model_opts)
            del fDBD_dist
        # Mahalanobis distance for distribution
        if gate("plain", "MahalanobisDistance"):
            maha_distance_dist = MahalanobisDistance(cf) 
            maha_distance_dist.compute_MahaDist_params(encoded_dist_train.mean(dim=2), labels_train)
            maha_distance_dist.save_params(filename='MahalanobisDistance_distribution_params'+model_opts)
            del maha_distance_dist
        # pNML for distribution
        if gate("plain", "pNML"):
            pnml_dist = pNML(module,study_name,cf)
            pnml_dist.compute_pNML_params(encoded_dist_train.mean(dim=2))
            pnml_dist.save_params(filename='pNML_distribution_params'+model_opts)
            del pnml_dist
        # ViM Score for distribution
        if gate("plain", "ViM"):
            vim_dist = ViMScore(module,study_name,cf)
            vim_dist.compute_ViM_params(encoded_dist_train.mean(dim=2))
            vim_dist.save_params(filename='ViM_distribution_params'+model_opts)
            del vim_dist
        # Residual Score for distribution
        if gate("plain", "Residual"):
            residual_dist = ResidualScore(module,study_name,cf)
            residual_dist.compute_Residual_params(encoded_dist_train.mean(dim=2))
            residual_dist.save_params(filename='Residual_distribution_params'+model_opts)
            del residual_dist
        # NeCo Score for distribution
        if gate("plain", "NeCo"):
            neco_dist = NeCo(module,study_name,cf)
            neco_dist.compute_NeCo_params(encoded_dist_train.mean(dim=2))
            neco_dist.save_params(filename='NeCo_distribution_params'+model_opts)
            del neco_dist
        del encoded_dist_train
        
        
#%%
# Load parameters
def load_score_methods(cf, module, study_name, do_enabled:bool, model_opts:str='', active: set | None = None, projections: set | None = None):
    active = set(ALL_FAMILIES) if active is None else set(active)
    projections = set(ALL_PROJECTIONS) if projections is None else set(projections)
    pf_needed = _needs_pf(active)
    def gate(mode, family):
        return mode in projections and family in active
    # Pre-initialize all gated CSF instances to _MISSING_CSF. If the family
    # is active, the load block below overwrites the placeholder; if the
    # family was skipped, the placeholder stays and the funcs dict at the
    # end still constructs cleanly. Stats() then either filters the entries
    # via filter_confids() or sees None returned from _MissingCSF.
    kpca_global = kpca_class = _MISSING_CSF
    ctm_global = ctm_class = ctm_class_pred = ctm = ctm_oc = _MISSING_CSF
    nnguide_global = nnguide_class_pred = nnguide = _MISSING_CSF
    fDBD_global = fDBD_class_pred = fdbd_inst = _MISSING_CSF
    maha_distance_global = maha_distance_class_pred = maha_distance = _MISSING_CSF
    pnml_global = pnml_class_pred = pnml = _MISSING_CSF
    generalized_entropy_global = generalized_entropy_class_pred = generalized_entropy_class = generalized_entropy = _MISSING_CSF
    renyi_entropy_global = renyi_entropy_class_pred = renyi_entropy_class = renyi_entropy = _MISSING_CSF
    vim = residual = neco = _MISSING_CSF
    # Temperature
    temperature_scale = TemperatureScaling(cf)
    temperature_scale.load_params(filename='Temperature_params'+model_opts)
    # Global
    # Kernel PCA Global
    if gate("global", "KPCA_RecError"):
        kpca_global = KernelPCA(module, study_name, cf, mode='global')
        kpca_global.load_params(filename='KernelPCA_global_params'+model_opts)
    # Projection Filtering Global (only loaded if any global-mode CSF is requested).
    if "global" in projections and pf_needed:
        projection_filtering_global = ProjectionFiltering(module, study_name, cf, mode='global')
        projection_filtering_global.load_params(filename='ProjectionFiltering_global_params'+model_opts)
    else:
        projection_filtering_global = _MISSING_CSF
    # Class Typical Matching Global
    if gate("global", "CTM"):
        ctm_global = ClassTypicalMatching(module, study_name, cf, mode='global')
        ctm_global.load_params(filename='CTM_global_params'+model_opts)
    # NNGuide PCA global
    if gate("global", "NNGuide"):
        nnguide_global = NNGuide(module,study_name,cf)
        nnguide_global.load_params(filename='NNGuide_global_params'+model_opts)
    # fDBD PCA global
    if gate("global", "fDBD"):
        fDBD_global = fDBD(module,study_name,cf)
        fDBD_global.load_params(filename='fDBD_global_params'+model_opts)
    # Mahalanobis distance global
    if gate("global", "MahalanobisDistance"):
        maha_distance_global = MahalanobisDistance(cf) 
        maha_distance_global.load_params(filename='MahalanobisDistance_global_params'+model_opts)
    # pNML global
    if gate("global", "pNML"):
        pnml_global = pNML(module,study_name,cf)
        pnml_global.load_params(filename='pNML_global_params'+model_opts)
    # Temperature Global
    if "global" in projections and pf_needed:
        temperature_global = TemperatureScaling(cf)
        temperature_global.load_params(filename='Temperature_global_params'+model_opts)
    else:
        temperature_global = _MISSING_CSF
    # Generalized entropy Global
    if gate("global", "GEN"):
        generalized_entropy_global = EntropyScores(cf, 'generalized')
        generalized_entropy_global.load_params(filename='GEN_global_params'+model_opts)
    # Renyi entropy Global
    if gate("global", "REN"):
        renyi_entropy_global = EntropyScores(cf, 'renyi')
        renyi_entropy_global.load_params(filename='REN_global_params'+model_opts)
    # Kernel PCA Class
    if gate("class", "KPCA_RecError"):
        kpca_class = KernelPCA(module, study_name, cf, mode='class')
        kpca_class.load_params(filename='KernelPCA_class_params'+model_opts)
    # Projection Filtering Class (only loaded if any class- or class_pred-mode CSF is requested).
    if ("class" in projections or "class_pred" in projections) and pf_needed:
        projection_filtering_class = ProjectionFiltering(module, study_name, cf, mode='class')
        projection_filtering_class.load_params(filename='ProjectionFiltering_class_params'+model_opts)
    else:
        projection_filtering_class = _MISSING_CSF
    # Class Typical Matching Class
    if gate("class", "CTM"):
        ctm_class = ClassTypicalMatching(module, study_name, cf, mode='class')
        ctm_class.load_params(filename='CTM_class_params'+model_opts)
    #
    # Temperature Class Pred
    if "class_pred" in projections:
        temperature_class_pred = TemperatureScaling(cf)
        temperature_class_pred.load_params(filename='Temperature_class_pred_params'+model_opts)
    else:
        temperature_class_pred = _MISSING_CSF
    # Generalized entropy Class Pred
    if gate("class_pred", "GEN"):
        generalized_entropy_class_pred = EntropyScores(cf, 'generalized')
        generalized_entropy_class_pred.load_params(filename='GEN_class_pred_params'+model_opts)
    # Renyi entropy Class Pred
    if gate("class_pred", "REN"):
        renyi_entropy_class_pred = EntropyScores(cf, 'renyi')
        renyi_entropy_class_pred.load_params(filename='REN_class_pred_params'+model_opts)
    #
    # Class Typical Matching Class w/predictions
    if gate("class_pred", "CTM"):
        ctm_class_pred = ClassTypicalMatching(module, study_name, cf, mode='global')
        ctm_class_pred.load_params(filename='CTM_class_pred_params'+model_opts)
    # NNGuide PCA class w/predictions
    if gate("class_pred", "NNGuide"):
        nnguide_class_pred = NNGuide(module,study_name,cf)
        nnguide_class_pred.load_params(filename='NNGuide_class_pred_params'+model_opts)
    # fDBD PCA class w/predictions
    if gate("class_pred", "fDBD"):
        fDBD_class_pred = fDBD(module,study_name,cf)
        fDBD_class_pred.load_params(filename='fDBD_class_pred_params'+model_opts)
    # Mahalanobis distance class w/predictions
    if gate("class_pred", "MahalanobisDistance"):
        maha_distance_class_pred = MahalanobisDistance(cf) 
        maha_distance_class_pred.load_params(filename='MahalanobisDistance_class_pred_params'+model_opts)
    # pNML class w/predictions
    if gate("class_pred", "pNML"):
        pnml_class_pred = pNML(module,study_name,cf)
        pnml_class_pred.load_params(filename='pNML_class_pred_params'+model_opts)
    # Temperature Class
    if "class" in projections:
        temperature_class = TemperatureScaling(cf)
        temperature_class.load_params(filename='Temperature_class_params'+model_opts)
    else:
        temperature_class = _MISSING_CSF
    # Generalized entropy Class
    if gate("class", "GEN"):
        generalized_entropy_class = EntropyScores(cf, 'generalized')
        generalized_entropy_class.load_params(filename='GEN_class_params'+model_opts)
    # Renyi entropy Class
    if gate("class", "REN"):
        renyi_entropy_class = EntropyScores(cf, 'renyi')
        renyi_entropy_class.load_params(filename='REN_class_params'+model_opts)

    # Class Typical Matching
    if gate("plain", "CTM"):
        ctm = ClassTypicalMatching(module, study_name, cf, mode='global')
        ctm.load_params(filename='CTM_params'+model_opts)
    # Class Typical Matching for correct only
    if gate("plain", "CTM"):
        ctm_oc = ClassTypicalMatching(module, study_name, cf, mode='global')
        ctm_oc.load_params(filename='CTM_oc_params'+model_opts)
    # Generalized entropy
    if gate("plain", "GEN"):
        generalized_entropy = EntropyScores(cf, 'generalized')
        generalized_entropy.load_params(filename='GEN_params'+model_opts)
    # Renyi entropy
    if gate("plain", "REN"):
        renyi_entropy = EntropyScores(cf, 'renyi')
        renyi_entropy.load_params(filename='REN_params'+model_opts)
    # NNGuide
    if gate("plain", "NNGuide"):
        nnguide = NNGuide(module,study_name,cf)
        nnguide.load_params(filename='NNGuide_params'+model_opts)
    # fDBD
    if gate("plain", "fDBD"):
        fdbd_inst = fDBD(module,study_name,cf)
        fdbd_inst.load_params(filename='fDBD_params'+model_opts)
    # Mahalanobis distance
    if gate("plain", "MahalanobisDistance"):
        maha_distance = MahalanobisDistance(cf)
        maha_distance.load_params(filename='MahalanobisDistance_params'+model_opts)
    # Mahalanobis++
    maha_pp = _MISSING_CSF
    if gate("plain", "MahalanobisPP"):
        maha_pp = MahalanobisPP(cf)
        maha_pp.load_params(filename='MahalanobisPP_params'+model_opts)
    # NCI
    nci_score = _MISSING_CSF
    if gate("plain", "NCI"):
        nci_score = NCI(module, study_name, cf)
        nci_score.load_params(filename='NCI_params'+model_opts)
    # pNML
    if gate("plain", "pNML"):
        pnml = pNML(module,study_name,cf)
        pnml.load_params(filename='pNML_params'+model_opts)
    # ViM Score 
    if gate("plain", "ViM"):
        vim = ViMScore(module,study_name,cf)
        vim.load_params(filename='ViM_params'+model_opts)
    # Residual Score 
    if gate("plain", "Residual"):
        residual = ResidualScore(module,study_name,cf)
        residual.load_params(filename='Residual_params'+model_opts)
    # NeCo Score 
    if gate("plain", "NeCo"):
        neco = NeCo(module,study_name,cf)
        neco.load_params(filename='NeCo_params'+model_opts)

    funcs = {
            'temperature_scale':temperature_scale,
            'kpca_global':kpca_global,
            'projection_filtering_global':projection_filtering_global,
            'ctm_global':ctm_global,
            'nnguide_global':nnguide_global,
            'fDBD_global':fDBD_global,
            'maha_distance_global':maha_distance_global,
            'pnml_global':pnml_global,
            'temperature_global':temperature_global,
            'generalized_entropy_global':generalized_entropy_global,
            'renyi_entropy_global':renyi_entropy_global,
            'kpca_class':kpca_class,
            'projection_filtering_class':projection_filtering_class,
            'ctm_class':ctm_class,
            'temperature_class_pred':temperature_class_pred,
            'generalized_entropy_class_pred':generalized_entropy_class_pred,
            'renyi_entropy_class_pred':renyi_entropy_class_pred,
            'ctm_class_pred':ctm_class_pred,
            'nnguide_class_pred':nnguide_class_pred,
            'fDBD_class_pred':fDBD_class_pred,
            'maha_distance_class_pred':maha_distance_class_pred,
            'pnml_class_pred':pnml_class_pred,
            'temperature_class':temperature_class,
            'generalized_entropy_class':generalized_entropy_class,
            'renyi_entropy_class':renyi_entropy_class,
            'ctm':ctm,
            'ctm_oc':ctm_oc,
            'generalized_entropy':generalized_entropy,
            'renyi_entropy':renyi_entropy,
            'nnguide':nnguide,
            'fDBD':fdbd_inst,
            'maha_distance':maha_distance,
            'maha_pp':maha_pp,
            'nci':nci_score,
            'pnml':pnml,
            'vim':vim,
            'residual':residual,
            'neco':neco,
            }

    if do_enabled:
        # Pre-initialize MCD-side gated variables. See the deterministic
        # block above for rationale.
        kpca_global_dist = kpca_class_dist = _MISSING_CSF
        ctm_global_dist = ctm_class_dist = ctm_class_pred_dist = ctm_dist = ctm_oc_dist = _MISSING_CSF
        nnguide_global_dist = nnguide_class_pred_dist = nnguide_dist = _MISSING_CSF
        fDBD_global_dist = fDBD_class_pred_dist = fDBD_dist = _MISSING_CSF
        maha_distance_global_dist = maha_distance_class_pred_dist = maha_distance_dist = _MISSING_CSF
        pnml_global_dist = pnml_class_pred_dist = pnml_dist = _MISSING_CSF
        generalized_entropy_global_dist = generalized_entropy_class_pred_dist = generalized_entropy_class_dist = generalized_entropy_dist = _MISSING_CSF
        renyi_entropy_global_dist = renyi_entropy_class_pred_dist = renyi_entropy_class_dist = renyi_entropy_dist = _MISSING_CSF
        vim_dist = residual_dist = neco_dist = _MISSING_CSF
        # Temperature for distribution
        temperature_scale_dist = TemperatureScaling(cf)
        temperature_scale_dist.load_params(filename='Temperature_distribution_params'+model_opts)
        # Kernel PCA Global
        if gate("global", "KPCA_RecError"):
            kpca_global_dist = KernelPCA(module, study_name, cf, mode='global')
            kpca_global_dist.load_params(filename='KernelPCA_global_distribution_params'+model_opts)
        # Projection Filtering Global for distribution (only loaded if any global-mode CSF is requested).
        if "global" in projections and pf_needed:
            projection_filtering_global_dist = ProjectionFiltering(module, study_name, cf, mode='global')
            projection_filtering_global_dist.load_params(filename='ProjectionFiltering_global_distribution_params'+model_opts)
        else:
            projection_filtering_global_dist = _MISSING_CSF
        # Class Typical Matching Global
        if gate("global", "CTM"):
            ctm_global_dist = ClassTypicalMatching(module, study_name, cf, mode='global')
            ctm_global_dist.load_params(filename='CTM_global_distribution_params'+model_opts)
        # NNGuide PCA global
        if gate("global", "NNGuide"):
            nnguide_global_dist = NNGuide(module,study_name,cf)
            nnguide_global_dist.load_params(filename='NNGuide_global_distribution_params'+model_opts)
        # fDBD PCA global
        if gate("global", "fDBD"):
            fDBD_global_dist = fDBD(module,study_name,cf)
            fDBD_global_dist.load_params(filename='fDBD_global_distribution_params'+model_opts)
        # Mahalanobis distance global for distribution
        if gate("global", "MahalanobisDistance"):
            maha_distance_global_dist = MahalanobisDistance(cf) 
            maha_distance_global_dist.load_params(filename='MahalanobisDistance_global_distribution_params'+model_opts)
        # pNML global for distribution
        if gate("global", "pNML"):
            pnml_global_dist = pNML(module,study_name,cf)
            pnml_global_dist.load_params(filename='pNML_global_distribution_params'+model_opts)
        # Temperature global for distribution
        if "global" in projections and pf_needed:
            temperature_global_dist = TemperatureScaling(cf)
            temperature_global_dist.load_params(filename='Temperature_global_distribution_params'+model_opts)
        else:
            temperature_global_dist = _MISSING_CSF
        # Generalized entropy Global for distribution
        if gate("global", "GEN"):
            generalized_entropy_global_dist = EntropyScores(cf, 'generalized')
            generalized_entropy_global_dist.load_params(filename='GEN_global_distribution_params'+model_opts)
        # Renyi entropy Global for distribution
        if gate("global", "REN"):
            renyi_entropy_global_dist = EntropyScores(cf, 'renyi')
            renyi_entropy_global_dist.load_params(filename='REN_global_distribution_params'+model_opts)
        # Kernel PCA Class
        if gate("class", "KPCA_RecError"):
            kpca_class_dist = KernelPCA(module, study_name, cf, mode='class')
            kpca_class_dist.load_params(filename='KernelPCA_class_distribution_params'+model_opts)
        # Projection Filtering Class for distribution (only loaded if any class- or class_pred-mode CSF is requested).
        if ("class" in projections or "class_pred" in projections) and pf_needed:
            projection_filtering_class_dist = ProjectionFiltering(module, study_name, cf, mode='class')
            projection_filtering_class_dist.load_params(filename='ProjectionFiltering_class_distribution_params'+model_opts)
        else:
            projection_filtering_class_dist = _MISSING_CSF
        # Class Typical Matching Class for distribution
        if gate("class", "CTM"):
            ctm_class_dist = ClassTypicalMatching(module, study_name, cf, mode='class')
            ctm_class_dist.load_params(filename='CTM_class_distribution_params'+model_opts)
        #
        # Temperature Class Pred for distribution
        if "class_pred" in projections:
            temperature_class_pred_dist = TemperatureScaling(cf)
            temperature_class_pred_dist.load_params(filename='Temperature_class_pred_distribution_params'+model_opts)
        else:
            temperature_class_pred_dist = _MISSING_CSF
        # Generalized entropy Class for distribution
        if gate("class_pred", "GEN"):
            generalized_entropy_class_pred_dist = EntropyScores(cf, 'generalized')
            generalized_entropy_class_pred_dist.load_params(filename='GEN_class_pred_distribution_params'+model_opts)
        # Renyi entropy Class for distribution
        if gate("class_pred", "REN"):
            renyi_entropy_class_pred_dist = EntropyScores(cf, 'renyi')
            renyi_entropy_class_pred_dist.load_params(filename='REN_class_pred_distribution_params'+model_opts)
        #
        # Class Typical Matching Class w/predictions for distribution
        if gate("class_pred", "CTM"):
            ctm_class_pred_dist = ClassTypicalMatching(module, study_name, cf, mode='global')
            ctm_class_pred_dist.load_params(filename='CTM_class_pred_distribution_params'+model_opts)
        # NNGuide PCA class w/predictions for distribution
        if gate("class_pred", "NNGuide"):
            nnguide_class_pred_dist = NNGuide(module,study_name,cf)
            nnguide_class_pred_dist.load_params(filename='NNGuide_class_pred_distribution_params'+model_opts)
        # fDBD PCA class w/predictions for distribution
        if gate("class_pred", "fDBD"):
            fDBD_class_pred_dist = fDBD(module,study_name,cf)
            fDBD_class_pred_dist.load_params(filename='fDBD_class_pred_distribution_params'+model_opts)
        # Mahalanobis distance class w/predictions
        if gate("class_pred", "MahalanobisDistance"):
            maha_distance_class_pred_dist = MahalanobisDistance(cf) 
            maha_distance_class_pred_dist.load_params(filename='MahalanobisDistance_class_pred_distribution_params'+model_opts)
        # pNML class w/predictions
        if gate("class_pred", "pNML"):
            pnml_class_pred_dist = pNML(module,study_name,cf)
            pnml_class_pred_dist.load_params(filename='pNML_class_pred_distribution_params'+model_opts)
        # Temperature Class for distribution
        if "class" in projections:
            temperature_class_dist = TemperatureScaling(cf)
            temperature_class_dist.load_params(filename='Temperature_class_distribution_params'+model_opts)
        else:
            temperature_class_dist = _MISSING_CSF
        # Generalized entropy Class for distribution
        if gate("class", "GEN"):
            generalized_entropy_class_dist = EntropyScores(cf, 'generalized')
            generalized_entropy_class_dist.load_params(filename='GEN_class_distribution_params'+model_opts)
        # Renyi entropy Class for distribution
        if gate("class", "REN"):
            renyi_entropy_class_dist = EntropyScores(cf, 'renyi')
            renyi_entropy_class_dist.load_params(filename='REN_class_distribution_params'+model_opts)
        # Class Typical Matching
        if gate("plain", "CTM"):
            ctm_dist = ClassTypicalMatching(module, study_name, cf, mode='global')
            ctm_dist.load_params(filename='CTM_distribution_params'+model_opts)
        # Class Typical Matching for only correct predictions
        if gate("plain", "CTM"):
            ctm_oc_dist = ClassTypicalMatching(module, study_name, cf, mode='global')
            ctm_oc_dist.load_params(filename='CTM_oc_distribution_params'+model_opts)
        # Generalized entropy for distribution
        if gate("plain", "GEN"):
            generalized_entropy_dist = EntropyScores(cf, 'generalized')
            generalized_entropy_dist.load_params(filename='GEN_distribution_params'+model_opts)
        # Renyi entropy for distribution
        if gate("plain", "REN"):
            renyi_entropy_dist = EntropyScores(cf, 'renyi')
            renyi_entropy_dist.load_params(filename='REN_distribution_params'+model_opts)
        # NNGuide for distribution
        if gate("plain", "NNGuide"):
            nnguide_dist = NNGuide(module,study_name,cf)
            nnguide_dist.load_params(filename='NNGuide_distribution_params'+model_opts)
        # fDBD for distribution
        if gate("plain", "fDBD"):
            fDBD_dist = fDBD(module,study_name,cf)
            fDBD_dist.load_params(filename='fDBD_distribution_params'+model_opts)
        # Mahalanobis distance for distribution
        if gate("plain", "MahalanobisDistance"):
            maha_distance_dist = MahalanobisDistance(cf) 
            maha_distance_dist.load_params(filename='MahalanobisDistance_distribution_params'+model_opts)
        # pNML for distribution
        if gate("plain", "pNML"):
            pnml_dist = pNML(module,study_name,cf)
            pnml_dist.load_params(filename='pNML_distribution_params'+model_opts)
        # ViM Score for distribution
        if gate("plain", "ViM"):
            vim_dist = ViMScore(module,study_name,cf)
            vim_dist.load_params(filename='ViM_distribution_params'+model_opts)
        # Residual Score for distribution
        if gate("plain", "Residual"):
            residual_dist = ResidualScore(module,study_name,cf)
            residual_dist.load_params(filename='Residual_distribution_params'+model_opts)
        # NeCo Score for distribution
        if gate("plain", "NeCo"):
            neco_dist = NeCo(module,study_name,cf)
            neco_dist.load_params(filename='NeCo_distribution_params'+model_opts)

        funcs_do = {
            'temperature_scale_dist':               temperature_scale_dist,
            'kpca_global_dist':                     kpca_global_dist,
            'projection_filtering_global_dist':     projection_filtering_global_dist,
            'ctm_global_dist':                      ctm_global_dist,
            'nnguide_global_dist':                  nnguide_global_dist,
            'fDBD_global_dist':                     fDBD_global_dist,
            'maha_distance_global_dist':            maha_distance_global_dist,
            'pnml_global_dist':                     pnml_global_dist,
            'temperature_global_dist':              temperature_global_dist,
            'generalized_entropy_global_dist':      generalized_entropy_global_dist,
            'renyi_entropy_global_dist':            renyi_entropy_global_dist,
            'kpca_class_dist':                      kpca_class_dist,
            'projection_filtering_class_dist':      projection_filtering_class_dist,
            'ctm_class_dist':                       ctm_class_dist,
            'temperature_class_pred_dist':          temperature_class_pred_dist,
            'generalized_entropy_class_pred_dist':  generalized_entropy_class_pred_dist,
            'renyi_entropy_class_pred_dist':        renyi_entropy_class_pred_dist,
            'ctm_class_pred_dist':                  ctm_class_pred_dist,
            'nnguide_class_pred_dist':              nnguide_class_pred_dist,
            'fDBD_class_pred_dist':                 fDBD_class_pred_dist,
            'maha_distance_class_pred_dist':        maha_distance_class_pred_dist,
            'pnml_class_pred_dist':                 pnml_class_pred_dist,
            'temperature_class_dist':               temperature_class_dist,
            'generalized_entropy_class_dist':       generalized_entropy_class_dist,
            'renyi_entropy_class_dist':             renyi_entropy_class_dist,
            'ctm_dist':                             ctm_dist,
            'ctm_oc_dist':                          ctm_oc_dist,
            'generalized_entropy_dist':             generalized_entropy_dist,
            'renyi_entropy_dist':                   renyi_entropy_dist,
            'nnguide_dist':                         nnguide_dist,
            'fDBD_dist':                            fDBD_dist,
            'maha_distance_dist':                   maha_distance_dist,
            'pnml_dist':                            pnml_dist,
            'vim_dist':                             vim_dist,
            'residual_dist':                        residual_dist,
            'neco_dist':                            neco_dist,
            }

    if do_enabled: 
        output = (funcs, funcs_do)
    else:
        output = (funcs) 
    return output 

#%%
def stats(module, study_name, cf, model_evaluations, eval_name:str, do_enabled:bool, model_opts:str='', n_bins:int=20, temp_scaled:bool=False, active: set | None = None, projections: set | None = None):
    active = set(ALL_FAMILIES) if active is None else set(active)
    projections = set(ALL_PROJECTIONS) if projections is None else set(projections)
    pf_needed = _needs_pf(active)
    def gate(mode, family):
        return mode in projections and family in active
    if do_enabled:
        score_methods, score_methods_do = load_score_methods(cf, module, study_name, do_enabled, model_opts=model_opts, active=active, projections=projections)
        gradnorm_score = GradNorm(module, study_name, cf) if "GradNorm" in active else _MISSING_CSF
        encoded_distribution = model_evaluations['encoded_dist']
        logits_distribution = model_evaluations['logits_dist']
        softmax_distribution = model_evaluations['softmax_scaled_dist'] if temp_scaled else model_evaluations['softmax_dist']
        preds_distribution = softmax_distribution.mean(dim=2).max(dim=1).indices
        #
        # Pre-initialize MCD projection-derived tensors; populate only for the
        # requested projection modes. Confids entries that use them are gated
        # by mode below.
        encoded_global_distribution_mcd = encoded_class_distribution_mcd = None
        encoded_class_pred_distribution_mcd = None
        logits_global_distribution_mcd = logits_class_distribution_mcd = None
        logits_class_pred_distribution_mcd = None
        softmax_global_distribution_mcd = softmax_class_distribution_mcd = None
        softmax_class_pred_distribution_mcd = None
        if "global" in projections and pf_needed:
            encoded_global_distribution_mcd = scores_funcs.mcd_function(score_methods_do['projection_filtering_global_dist'].get_backprojection, encoded_distribution)
            logits_global_distribution_mcd = scores_funcs.mcd_function(score_methods_do['projection_filtering_global_dist'].get_logits, encoded_distribution)
            softmax_global_distribution_mcd = score_methods_do['temperature_global_dist'].get_scaled_softmax(logits_global_distribution_mcd) if temp_scaled else F.softmax(logits_global_distribution_mcd, dim=1, dtype=torch.float64)
        if ("class" in projections or "class_pred" in projections) and pf_needed:
            encoded_class_distribution_mcd = scores_funcs.mcd_function(score_methods_do['projection_filtering_class_dist'].get_backprojection, encoded_distribution)
            logits_class_distribution_mcd = scores_funcs.mcd_function(score_methods_do['projection_filtering_class_dist'].get_logits, encoded_distribution)
            if "class" in projections:
                softmax_class_distribution_mcd = score_methods_do['temperature_class_dist'].get_scaled_softmax(logits_class_distribution_mcd) if temp_scaled else F.softmax(logits_class_distribution_mcd, dim=1, dtype=torch.float64)
            if "class_pred" in projections:
                encoded_class_pred_distribution_mcd, logits_class_pred_distribution_mcd = score_methods_do['projection_filtering_class_dist'].get_combined_backprojection(encoded_class_distribution_mcd, combine='prediction', preds=preds_distribution)
                softmax_class_pred_distribution_mcd = score_methods_do['temperature_class_pred_dist'].get_scaled_softmax(logits_class_pred_distribution_mcd) if temp_scaled else F.softmax(logits_class_pred_distribution_mcd, dim=1, dtype=torch.float64)
        # 
        confid_distribution = model_evaluations['confid_dist']
        correct_distribution = model_evaluations['correct_mcd']
        residuals_distribution = 1-correct_distribution
        mcd_confids = {
            # Kernel RecError global for distribution
            'MCD-KPCA_RecError_global' : scores_funcs.mcd_function(score_methods_do['kpca_global_dist'].get_scores, encoded_distribution) if "global" in projections and pf_needed else None,
            'MCD-KPCA_ERecError_global' : scores_funcs.mcd_expected_function(score_methods_do['kpca_global_dist'].get_scores, encoded_distribution) if "global" in projections and pf_needed else None,
            # RecError global for distribution
            'MCD-PCA_RecError_global' : scores_funcs.mcd_function(score_methods_do['projection_filtering_global_dist'].get_scores, encoded_distribution) if "global" in projections and pf_needed else None,
            'MCD-PCA_ERecError_global' : scores_funcs.mcd_expected_function(score_methods_do['projection_filtering_global_dist'].get_scores, encoded_distribution) if "global" in projections and pf_needed else None,
            # CTM global for distribution
            'MCD-CTM_global' :          score_methods_do['ctm_global_dist'].get_scores(encoded_global_distribution_mcd, similarity='weight'),
            'MCD-CTM_global_mean' :     score_methods_do['ctm_global_dist'].get_scores(encoded_global_distribution_mcd, similarity='mean'),
            'MCD-ECTM_global' :         score_methods_do['ctm_global_dist'].get_scores(encoded_distribution, similarity='weight'),
            'MCD-ECTM_global_mean' :    score_methods_do['ctm_global_dist'].get_scores(encoded_distribution, similarity='mean'),
            # NNGuide global for distribution
            'MCD-NNGuide_global':       score_methods_do['nnguide_global_dist'].get_scores(encoded_global_distribution_mcd),
            'MCD-ENNGuide_global':      scores_funcs.mcd_expected_function(score_methods_do['nnguide_global_dist'].get_scores, encoded_distribution) if "global" in projections and pf_needed else None,
            # fDBD global for distribution
            'MCD-fDBD_global':          score_methods_do['fDBD_global_dist'].get_scores(encoded_global_distribution_mcd, logits_eval=logits_global_distribution_mcd),
            'MCD-EfDBD_global':         scores_funcs.mcd_expected_function(score_methods_do['fDBD_global_dist'].get_scores, encoded_distribution, logits_eval=logits_distribution) if "global" in projections and pf_needed else None,
            # Maha Distance global for distribution
            'MCD-Maha_global':          score_methods_do['maha_distance_global_dist'].get_scores(encoded_global_distribution_mcd),
            'MCD-EMaha_global':         scores_funcs.mcd_expected_function(score_methods_do['maha_distance_global_dist'].get_scores, encoded_distribution) if "global" in projections and pf_needed else None,
            # pNML global for distribution
            'MCD-pNML_global':          score_methods_do['pnml_global_dist'].get_scores(encoded_global_distribution_mcd),
            'MCD-EpNML_global':         scores_funcs.mcd_expected_function(score_methods_do['pnml_global_dist'].get_scores, encoded_distribution) if "global" in projections and pf_needed else None,
            # Entropies global for distribution
            'MCD-GEN_global' :          score_methods_do['generalized_entropy_global_dist'].get_scores(softmax_global_distribution_mcd),
            'MCD-REN_global' :          score_methods_do['renyi_entropy_global_dist'].get_scores(softmax_global_distribution_mcd),
            # Kernel RecError global for distribution
            'MCD-KPCA_RecError_class' : scores_funcs.mcd_function(score_methods_do['kpca_class_dist'].get_scores, encoded_distribution) if "class" in projections and pf_needed else None,
            'MCD-KPCA_ERecError_class' : scores_funcs.mcd_expected_function(score_methods_do['kpca_class_dist'].get_scores, encoded_distribution) if "class" in projections and pf_needed else None,
            # RecError class for distribution
            'MCD-PCA_RecError_class' : scores_funcs.mcd_function(score_methods_do['projection_filtering_class_dist'].get_scores, encoded_distribution) if "class" in projections and pf_needed else None,
            'MCD-PCA_ERecError_class' : scores_funcs.mcd_expected_function(score_methods_do['projection_filtering_class_dist'].get_scores, encoded_distribution) if "class" in projections and pf_needed else None,
            # Kernel RecError global for distribution
            'MCD-KPCA_RecError_class_pred' : scores_funcs.mcd_function(score_methods_do['kpca_class_dist'].get_scores, encoded_distribution, predictions_eval=preds_distribution) if "class_pred" in projections and pf_needed else None,
            'MCD-KPCA_ERecError_class_pred' : scores_funcs.mcd_expected_function(score_methods_do['kpca_class_dist'].get_scores, encoded_distribution, predictions_eval=softmax_distribution.max(dim=1).indices) if "class_pred" in projections and pf_needed else None,
            # RecError class pred for distribution
            'MCD-PCA_RecError_class_pred' : scores_funcs.mcd_function(score_methods_do['projection_filtering_class_dist'].get_scores, encoded_distribution, X_back_projected_eval=encoded_class_pred_distribution_mcd) if "class_pred" in projections and pf_needed else None,
            'MCD-PCA_ERecError_class_pred' : scores_funcs.mcd_expected_function(score_methods_do['projection_filtering_class_dist'].get_scores, encoded_distribution, predictions_eval=softmax_distribution.max(dim=1).indices) if "class_pred" in projections and pf_needed else None,
            # CTM class for distribution
            'MCD-CTM_class' :           score_methods_do['ctm_class_dist'].get_scores(encoded_class_distribution_mcd, similarity='weight'),
            'MCD-CTM_class_mean' :      score_methods_do['ctm_class_dist'].get_scores(encoded_class_distribution_mcd, similarity='mean'),
            # CTM class pred for distribution
            'MCD-CTM_class_pred' :      score_methods_do['ctm_class_pred_dist'].get_scores(encoded_class_pred_distribution_mcd, similarity='weight'),
            'MCD-CTM_class_pred_mean' : score_methods_do['ctm_class_pred_dist'].get_scores(encoded_class_pred_distribution_mcd, similarity='mean'),
            'MCD-ECTM_class_pred' :     score_methods_do['ctm_class_pred_dist'].get_scores( encoded_distribution, similarity='weight'),
            'MCD-ECTM_class_pred_mean': score_methods_do['ctm_class_pred_dist'].get_scores( encoded_distribution, similarity='mean'),   
            # NNGuide class pred for distribution
            'MCD-NNGuide_class_pred':   score_methods_do['nnguide_class_pred_dist'].get_scores(encoded_class_pred_distribution_mcd),
            'MCD-ENNGuide_class_pred':  scores_funcs.mcd_expected_function(score_methods_do['nnguide_class_pred_dist'].get_scores, encoded_distribution) if "class_pred" in projections and pf_needed else None,
            # fDBD class pred for distribution
            'MCD-fDBD_class_pred':      score_methods_do['fDBD_class_pred_dist'].get_scores(encoded_class_pred_distribution_mcd, logits_eval=logits_class_distribution_mcd),
            'MCD-EfDBD_class_pred':     scores_funcs.mcd_expected_function(score_methods_do['fDBD_class_pred_dist'].get_scores, encoded_distribution, logits_eval=logits_distribution) if "class_pred" in projections and pf_needed else None,
            # Maha class pred for distribution
            'MCD-Maha_class_pred':      score_methods_do['maha_distance_class_pred_dist'].get_scores(encoded_class_pred_distribution_mcd),
            'MCD-EMaha_class_pred':     scores_funcs.mcd_expected_function(score_methods_do['maha_distance_class_pred_dist'].get_scores, encoded_distribution) if "class_pred" in projections and pf_needed else None,
            # pNML class pred for distribution
            'MCD-pNML_class_pred':      score_methods_do['pnml_class_pred_dist'].get_scores(encoded_class_pred_distribution_mcd),
            'MCD-EpNML_class_pred':     scores_funcs.mcd_expected_function(score_methods_do['pnml_class_pred_dist'].get_scores, encoded_distribution) if "class_pred" in projections and pf_needed else None,
            # Entropies class for distribution
            'MCD-GEN_class' :           score_methods_do['generalized_entropy_class_dist'].get_scores(softmax_class_distribution_mcd),
            'MCD-REN_class' :           score_methods_do['renyi_entropy_class_dist'].get_scores(softmax_class_distribution_mcd),
            'MCD-GEN_class_pred' :       score_methods_do['generalized_entropy_class_pred_dist'].get_scores(softmax_class_pred_distribution_mcd),
            'MCD-REN_class_pred' :       score_methods_do['renyi_entropy_class_pred_dist'].get_scores(softmax_class_pred_distribution_mcd),
            # CTM for distribution
            'MCD-CTM' :                scores_funcs.mcd_function(score_methods_do['ctm_dist'].get_scores, encoded_distribution, similarity='weight'),
            'MCD-ECTM' :               score_methods_do['ctm_dist'].get_scores(encoded_distribution, similarity='weight'),
            'MCD-CTM_mean' :           scores_funcs.mcd_function(score_methods_do['ctm_dist'].get_scores, encoded_distribution, similarity='mean'),
            'MCD-ECTM_mean' :          score_methods_do['ctm_dist'].get_scores(encoded_distribution, similarity='mean'),
            # CTM (only correct) for distribution
            'MCD-CTM_oc_mean':          scores_funcs.mcd_function(score_methods_do['ctm_oc_dist'].get_scores, encoded_distribution, similarity='mean'),
            'MCD-ECTM_oc_mean':         score_methods_do['ctm_oc_dist'].get_scores(encoded_distribution, similarity='mean'),            
            # Entropies for distribution
            'MCD-GEN' :                 scores_funcs.mcd_function(score_methods_do['generalized_entropy_dist'].get_scores, softmax_distribution),
            'MCD-EGEN' :                scores_funcs.mcd_expected_function(score_methods_do['generalized_entropy_dist'].get_scores, softmax_distribution),
            'MCD-REN' :                 scores_funcs.mcd_function(score_methods_do['renyi_entropy_dist'].get_scores, softmax_distribution),
            'MCD-EREN' :                scores_funcs.mcd_expected_function(score_methods_do['renyi_entropy_dist'].get_scores, softmax_distribution),
            # NNGuide for distribution
            'MCD-NNGuide':              scores_funcs.mcd_function(score_methods_do['nnguide_dist'].get_scores, encoded_distribution),
            'MCD-ENNGuide':             scores_funcs.mcd_expected_function(score_methods_do['nnguide_dist'].get_scores, encoded_distribution),
            # fDBD for distribution
            'MCD-fDBD':                 scores_funcs.mcd_function(score_methods_do['fDBD_dist'].get_scores, encoded_distribution, logits_eval=logits_distribution),
            'MCD-EfDBD':                scores_funcs.mcd_expected_function(score_methods_do['fDBD_dist'].get_scores, encoded_distribution, logits_eval=logits_distribution),
            # Maha for distribution
            'MCD-Maha':                 scores_funcs.mcd_function(score_methods_do['maha_distance_dist'].get_scores, encoded_distribution),
            'MCD-EMaha':                scores_funcs.mcd_expected_function(score_methods_do['maha_distance_dist'].get_scores, encoded_distribution),
            # pNML for distribution
            'MCD-pNML':                 scores_funcs.mcd_function(score_methods_do['pnml_dist'].get_scores, encoded_distribution),
            'MCD-EpNML':                scores_funcs.mcd_expected_function(score_methods_do['pnml_dist'].get_scores, encoded_distribution),
            # ViM for distribution
            'MCD-ViM':                 scores_funcs.mcd_function(score_methods_do['vim_dist'].get_scores, encoded_distribution),
            'MCD-EViM':                scores_funcs.mcd_expected_function(score_methods_do['vim_dist'].get_scores, encoded_distribution),
            # Residual for distribution
            'MCD-Residual':             scores_funcs.mcd_function(score_methods_do['residual_dist'].get_scores, encoded_distribution),
            'MCD-EResidual':            scores_funcs.mcd_expected_function(score_methods_do['residual_dist'].get_scores, encoded_distribution),
            # NeCo for distribution
            'MCD-NeCo':                 scores_funcs.mcd_function(score_methods_do['neco_dist'].get_scores, encoded_distribution),
            'MCD-ENeCo':                scores_funcs.mcd_expected_function(score_methods_do['neco_dist'].get_scores, encoded_distribution),            
            # Scores that do not requiere preprocessing
            'MCD-MSR' :                 scores_funcs.mcd_function(scores_funcs.maximum_softmax_response, softmax_distribution),
            'MCD-PE' :                  scores_funcs.mcd_function(scores_funcs.predictive_entropy, softmax_distribution),
            'MCD-MLS' :                 scores_funcs.mcd_function(scores_funcs.maximum_logit_score, logits_distribution, temperature=score_methods['temperature_scale'].temperature),
            'MCD-PCE' :                 scores_funcs.mcd_function(scores_funcs.predictive_collision_entropy, softmax_distribution),
            'MCD-GE' :                  scores_funcs.mcd_function(scores_funcs.guessing_entropy, softmax_distribution),
            'MCD-Energy' :              scores_funcs.mcd_function(scores_funcs.energy, logits_distribution, temperature=score_methods_do['temperature_scale_dist'].temperature),
            'MCD-EMSR' :                scores_funcs.mcd_expected_function(scores_funcs.maximum_softmax_response, softmax_distribution),
            'MCD-EPE' :                 scores_funcs.mcd_expected_function(scores_funcs.predictive_entropy, softmax_distribution),
            'MCD-EMLS' :                scores_funcs.mcd_expected_function(scores_funcs.maximum_logit_score, logits_distribution, temperature=score_methods['temperature_scale'].temperature),    
            'MCD-EPCE' :                scores_funcs.mcd_expected_function(scores_funcs.predictive_collision_entropy, softmax_distribution),
            'MCD-EGE' :                 scores_funcs.mcd_expected_function(scores_funcs.guessing_entropy, softmax_distribution),
            'MCD-EEnergy' :             scores_funcs.mcd_expected_function(scores_funcs.energy, logits_distribution, temperature=score_methods_do['temperature_scale_dist'].temperature),    
            # Scores that do not requiere preprocessing using global projection filtering
            'MCD-MSR_global' :         scores_funcs.maximum_softmax_response(softmax_global_distribution_mcd) if "global" in projections and pf_needed else None,
            'MCD-PE_global' :          scores_funcs.predictive_entropy(softmax_global_distribution_mcd) if "global" in projections and pf_needed else None,
            'MCD-MLS_global' :         scores_funcs.maximum_logit_score(logits_global_distribution_mcd, temperature=score_methods_do['temperature_global_dist'].temperature) if "global" in projections and pf_needed else None,
            'MCD-PCE_global' :         scores_funcs.predictive_collision_entropy(softmax_global_distribution_mcd) if "global" in projections and pf_needed else None,
            'MCD-GE_global' :          scores_funcs.guessing_entropy(softmax_global_distribution_mcd) if "global" in projections and pf_needed else None,
            'MCD-Energy_global' :      scores_funcs.energy(logits_global_distribution_mcd, temperature=score_methods_do['temperature_global_dist'].temperature) if "global" in projections and pf_needed else None,
            # Scores that do not requiere preprocessing using class projection filtering
            'MCD-MSR_class' :         scores_funcs.maximum_softmax_response(softmax_class_distribution_mcd) if "class" in projections and pf_needed else None,
            'MCD-PE_class' :          scores_funcs.predictive_entropy(softmax_class_distribution_mcd) if "class" in projections and pf_needed else None,
            'MCD-MLS_class' :         scores_funcs.maximum_logit_score(logits_class_distribution_mcd, temperature=score_methods_do['temperature_class_dist'].temperature) if "class" in projections and pf_needed else None,
            'MCD-PCE_class' :         scores_funcs.predictive_collision_entropy(softmax_class_distribution_mcd) if "class" in projections and pf_needed else None,
            'MCD-GE_class' :          scores_funcs.guessing_entropy(softmax_class_distribution_mcd) if "class" in projections and pf_needed else None,
            'MCD-Energy_class' :      scores_funcs.energy(logits_class_distribution_mcd, temperature=score_methods_do['temperature_class_dist'].temperature) if "class" in projections and pf_needed else None,
            # Scores that do not requiere preprocessing using class projection filtering
            'MCD-MSR_class_pred' :         scores_funcs.maximum_softmax_response(softmax_class_pred_distribution_mcd) if "class_pred" in projections and pf_needed else None,
            'MCD-PE_class_pred' :          scores_funcs.predictive_entropy(softmax_class_pred_distribution_mcd) if "class_pred" in projections and pf_needed else None,
            'MCD-MLS_class_pred' :         scores_funcs.maximum_logit_score(logits_class_pred_distribution_mcd, temperature=score_methods_do['temperature_class_pred_dist'].temperature) if "class_pred" in projections and pf_needed else None,
            'MCD-PCE_class_pred' :         scores_funcs.predictive_collision_entropy(softmax_class_pred_distribution_mcd) if "class_pred" in projections and pf_needed else None,
            'MCD-GE_class_pred' :          scores_funcs.guessing_entropy(softmax_class_pred_distribution_mcd) if "class_pred" in projections and pf_needed else None,
            'MCD-Energy_class_pred' :      scores_funcs.energy(logits_class_pred_distribution_mcd, temperature=score_methods_do['temperature_class_pred_dist'].temperature) if "class_pred" in projections and pf_needed else None,
            #             
            'MCD-GradNorm' :            scores_funcs.mcd_function(gradnorm_score.get_scores, encoded_distribution, use_cuda=use_cuda, temperature=score_methods_do['temperature_scale_dist'].temperature),
            'MCD-GradNorm_global' :     gradnorm_score.get_scores(encoded_global_distribution_mcd, use_cuda=use_cuda, temperature=score_methods_do['temperature_global_dist'].temperature) if "global" in projections and pf_needed else None,
            'MCD-GradNorm_class_pred' : gradnorm_score.get_scores(encoded_class_pred_distribution_mcd, use_cuda=use_cuda, temperature=score_methods_do['temperature_class_pred_dist'].temperature) if "class_pred" in projections and pf_needed else None,
            #
            'MCD-MI' :          scores_funcs.mcd_mutual_information(softmax_distribution),
            'MCD-Confidence' :  confid_distribution.mean(dim=1),
        }
        # Filter confids to active families only (the rest are preserved
        # from any existing CSV via the merge below).
        mcd_confids = filter_confids(mcd_confids, active, projections)
        mcd_confids_df = pd.DataFrame(mcd_confids)
        mcd_confids_df['residuals'] = residuals_distribution
        mcd_stats = {
                        key:[RiskCoverageStats(confids=mcd_confids[key], residuals=residuals_distribution), metrics.StatsCache(mcd_confids[key],correct_distribution,n_bins) ] for key in mcd_confids
                    }

        mcd_stats_df = pd.DataFrame( {  
                                        'AUGRC': { k:mcd_stats[k][0].augrc for k in mcd_stats },
                                        'AURC': { k:mcd_stats[k][0].aurc for k in mcd_stats },
                                        'AUROC_f': { k:metrics.failauc(mcd_stats[k][1]) for k in mcd_stats },
                                        'FPR@95TPR': { k:metrics.fpr_at_95_tpr(mcd_stats[k][1]) for k in mcd_stats },
                                        'ECE': { k:metrics.expected_calibration_error(mcd_stats[k][1]) for k in mcd_stats },
                                        'MCE': { k:metrics.maximum_calibration_error(mcd_stats[k][1]) for k in mcd_stats },
                                        'AP_ferr': { k:metrics.failap_err(mcd_stats[k][1]) for k in mcd_stats },
                                        'AP_fsuc': { k:metrics.failap_suc(mcd_stats[k][1]) for k in mcd_stats },
                                    } )
        filename = f'mcdstats{model_opts}_{eval_name}.csv'
        filename_confids = f'mcdconfids{model_opts}_{eval_name}.csv'
        if os.path.exists(f'{cf.exp.dir}/analysis'):
            path = f'{cf.exp.dir}/analysis/{filename}'
            path_confids = f'{cf.exp.dir}/analysis/{filename_confids}'
        else:
            os.mkdir(f'{cf.exp.dir}/analysis')
            path = f'{cf.exp.dir}/analysis/{filename}'
            path_confids = f'{cf.exp.dir}/analysis/{filename_confids}'
        # Merge with any existing CSV so other families' results are preserved.
        merged_mcd_stats = _merge_csv_rows(mcd_stats_df, path)
        merged_mcd_confids = _merge_csv_cols(mcd_confids_df, path_confids)
        merged_mcd_stats.sort_values(by=['AUGRC']).to_csv(path)
        merged_mcd_confids.to_csv(path_confids)
    else:
        score_methods = load_score_methods(cf, module, study_name, do_enabled, model_opts=model_opts, active=active, projections=projections)

    gradnorm_score = GradNorm(module, study_name, cf) if "GradNorm" in active else _MISSING_CSF
    encoded = model_evaluations['encoded']
    logits = model_evaluations['logits']
    softmax = model_evaluations['softmax_scaled'] if temp_scaled else model_evaluations['softmax']
    preds = softmax.max(dim=1).indices
    # Pre-initialize projection-derived tensors. Each block below sets them
    # only when the corresponding mode is in `projections`; otherwise they
    # stay None. Confids entries that reference them are gated by mode in
    # the dict literal below and filtered out via filter_confids().
    encoded_global = encoded_class = encoded_class_pred = None
    logits_global = logits_class = logits_class_pred = None
    softmax_global = softmax_class = softmax_class_pred = None
    if "global" in projections and pf_needed:
        encoded_global = score_methods['projection_filtering_global'].get_backprojection(encoded)
        logits_global = score_methods['projection_filtering_global'].get_logits(encoded)
        softmax_global = score_methods['temperature_global'].get_scaled_softmax(logits_global) if temp_scaled else F.softmax(logits_global, dim=1, dtype=torch.float64)
    if ("class" in projections or "class_pred" in projections) and pf_needed:
        encoded_class = score_methods['projection_filtering_class'].get_backprojection(encoded)
        logits_class = score_methods['projection_filtering_class'].get_logits(encoded)
        if "class" in projections:
            softmax_class = score_methods['temperature_class'].get_scaled_softmax(logits_class) if temp_scaled else F.softmax(logits_class, dim=1, dtype=torch.float64)
        if "class_pred" in projections:
            encoded_class_pred, logits_class_pred = score_methods['projection_filtering_class'].get_combined_backprojection(encoded_class, combine='prediction', preds=preds)
            softmax_class_pred = score_methods['temperature_class_pred'].get_scaled_softmax(logits_class_pred) if temp_scaled else F.softmax(logits_class_pred, dim=1, dtype=torch.float64)
    #
    confid = model_evaluations['confid']
    correct = model_evaluations['correct']
    residuals = 1-correct

    confids= {
                # KPCA RecError global
                'KPCA_RecError_global':  score_methods['kpca_global'].get_scores(encoded),
                # RecError global
                'PCA_RecError_global':  score_methods['projection_filtering_global'].get_scores(encoded),
                # CTM global
                'CTM_global':           score_methods['ctm_global'].get_scores(encoded_global, similarity='weight'),
                'CTM_global_mean':      score_methods['ctm_global'].get_scores(encoded_global, similarity='mean'),
                # NNGuide global 
                'NNGuide_global':       score_methods['nnguide_global'].get_scores(encoded_global),
                # fDBD global
                'fDBD_global':          score_methods['fDBD_global'].get_scores(encoded_global, logits_eval=logits_global),
                # Maha global
                'Maha_global':          score_methods['maha_distance_global'].get_scores(encoded_global),
                # pNML global
                'pNML_global':          score_methods['pnml_global'].get_scores(encoded_global),
                # Entropies
                'GEN_global' :          score_methods['generalized_entropy_global'].get_scores(softmax_global),
                'REN_global' :          score_methods['renyi_entropy_global'].get_scores(softmax_global),
                # KPCA RecError class
                'KPCA_RecError_class':  score_methods['kpca_class'].get_scores(encoded),
                # KPCA RecError class pred
                'KPCA_RecError_class_pred':  score_methods['kpca_class'].get_scores(encoded,predictions_eval=preds),
                # RecError class
                'PCA_RecError_class':   score_methods['projection_filtering_class'].get_scores(encoded),
                # RecError class pred
                'PCA_RecError_class_pred':  score_methods['projection_filtering_class'].get_scores(encoded, X_back_projected_eval=encoded_class_pred),
                # CTM class
                'CTM_class':            score_methods['ctm_class'].get_scores(encoded_class, similarity='weight'),
                'CTM_class_mean':       score_methods['ctm_class'].get_scores(encoded_class, similarity='mean'),
                # CTM class pred
                'CTM_class_pred':       score_methods['ctm_class_pred'].get_scores(encoded_class_pred, similarity='weight'),
                'CTM_class_pred_mean':  score_methods['ctm_class_pred'].get_scores(encoded_class_pred, similarity='mean'),
                # NNGuide class pred
                'NNGuide_class_pred':   score_methods['nnguide_class_pred'].get_scores(encoded_class_pred),
                # fDBD class pred
                'fDBD_class_pred':      score_methods['fDBD_class_pred'].get_scores(encoded_class_pred, logits_eval=logits_class),
                # Maha class pred
                'Maha_class_pred':      score_methods['maha_distance_class_pred'].get_scores(encoded_class_pred),
                # pNML class pred
                'pNML_class_pred':      score_methods['pnml_class_pred'].get_scores(encoded_class_pred),
                # Entropies
                'GEN_class' :           score_methods['generalized_entropy_class'].get_scores(softmax_class),
                'REN_class' :           score_methods['renyi_entropy_class'].get_scores(softmax_class),
                'GEN_class_pred' :      score_methods['generalized_entropy_class_pred'].get_scores(softmax_class_pred),
                'REN_class_pred' :      score_methods['renyi_entropy_class_pred'].get_scores(softmax_class_pred),
                # CTM
                'CTM':                  score_methods['ctm'].get_scores(encoded, similarity='weight'),
                'CTM_mean':             score_methods['ctm'].get_scores(encoded, similarity='mean'),
                'CTM_oc_mean':          score_methods['ctm_oc'].get_scores(encoded, similarity='mean'),
                # Entropies
                'GEN' :                 score_methods['generalized_entropy'].get_scores(softmax),
                'REN' :                 score_methods['renyi_entropy'].get_scores(softmax),
                # NNGuide
                'NNGuide':              score_methods['nnguide'].get_scores(encoded),
                # fDBD
                'fDBD':                 score_methods['fDBD'].get_scores(encoded, logits_eval=logits),
                # Maha
                'Maha':                 score_methods['maha_distance'].get_scores(encoded),
                # Mahalanobis++ (L2-normalized features)
                'MahaPP':               score_methods['maha_pp'].get_scores(encoded),
                # NCI (weight alignment + L1 norm filter)
                'NCI':                  score_methods['nci'].get_scores(encoded, logits_eval=logits),
                # pNML
                'pNML':                 score_methods['pnml'].get_scores(encoded),
                # ViM
                'ViM':                  score_methods['vim'].get_scores(encoded),
                # Residual
                'Residual':             score_methods['residual'].get_scores(encoded),
                # NeCo
                'NeCo':                 score_methods['neco'].get_scores(encoded),  
                # Scores that do not requiere preprocessing
                'MSR' :                 scores_funcs.maximum_softmax_response(softmax),
                'PE' :                  scores_funcs.predictive_entropy(softmax),
                'MLS' :                 scores_funcs.maximum_logit_score(logits, temperature=score_methods['temperature_scale'].temperature),
                'PCE' :                 scores_funcs.predictive_collision_entropy(softmax),
                'GE' :                  scores_funcs.guessing_entropy(softmax),
                'Energy' :              scores_funcs.energy(logits, temperature=score_methods['temperature_scale'].temperature),
                # Scores that do not requiere preprocessing using global projection filtering
                'MSR_global' :          scores_funcs.maximum_softmax_response(softmax_global) if "global" in projections and pf_needed else None,
                'PE_global' :           scores_funcs.predictive_entropy(softmax_global) if "global" in projections and pf_needed else None,
                'MLS_global' :          scores_funcs.maximum_logit_score(logits_global, temperature=score_methods['temperature_global'].temperature) if "global" in projections and pf_needed else None,
                'PCE_global' :          scores_funcs.predictive_collision_entropy(softmax_global) if "global" in projections and pf_needed else None,
                'GE_global' :           scores_funcs.guessing_entropy(softmax_global) if "global" in projections and pf_needed else None,
                'Energy_global' :       scores_funcs.energy(logits_global, temperature=score_methods['temperature_global'].temperature) if "global" in projections and pf_needed else None,
                # Scores that do not requiere preprocessing using class projection filtering
                'MSR_class' :           scores_funcs.maximum_softmax_response(softmax_class) if "class" in projections and pf_needed else None,
                'MSR_class_pred' :      scores_funcs.maximum_softmax_response(softmax_class_pred) if "class_pred" in projections and pf_needed else None,
                'PE_class' :            scores_funcs.predictive_entropy(softmax_class) if "class" in projections and pf_needed else None,
                'PE_class_pred' :       scores_funcs.predictive_entropy(softmax_class_pred) if "class_pred" in projections and pf_needed else None,
                'MLS_class' :           scores_funcs.maximum_logit_score(logits_class, temperature=score_methods['temperature_class'].temperature) if "class" in projections and pf_needed else None,
                'MLS_class_pred' :      scores_funcs.maximum_logit_score(logits_class_pred, temperature=score_methods['temperature_class_pred'].temperature) if "class_pred" in projections and pf_needed else None,
                'PCE_class' :           scores_funcs.predictive_collision_entropy(softmax_class) if "class" in projections and pf_needed else None,
                'PCE_class_pred' :      scores_funcs.predictive_collision_entropy(softmax_class_pred) if "class_pred" in projections and pf_needed else None,
                'GE_class' :            scores_funcs.guessing_entropy(softmax_class) if "class" in projections and pf_needed else None,
                'GE_class_pred' :       scores_funcs.guessing_entropy(softmax_class_pred) if "class_pred" in projections and pf_needed else None,
                'Energy_class' :        scores_funcs.energy(logits_class, temperature=score_methods['temperature_class'].temperature) if "class" in projections and pf_needed else None,
                'Energy_class_pred' :   scores_funcs.energy(logits_class_pred, temperature=score_methods['temperature_class_pred'].temperature) if "class_pred" in projections and pf_needed else None,
                # 
                'GradNorm' :            gradnorm_score.get_scores(encoded, temperature=score_methods['temperature_scale'].temperature, use_cuda=use_cuda),
                'GradNorm_global' :     gradnorm_score.get_scores(encoded_global, temperature=score_methods['temperature_global'].temperature, use_cuda=use_cuda) if "global" in projections and pf_needed else None,
                'GradNorm_class_pred' : gradnorm_score.get_scores(encoded_class_pred, temperature=score_methods['temperature_class_pred'].temperature, use_cuda=use_cuda) if "class_pred" in projections and pf_needed else None,    
                'Confidence' :          confid,
    }
    # Filter confids to active families only (the rest are preserved
    # from any existing CSV via the merge below).
    confids = filter_confids(confids, active, projections)
    confids_df = pd.DataFrame(confids)
    confids_df['residuals'] = residuals
    stats = {
                key:[ RiskCoverageStats(confids=confids[key], residuals=residuals), metrics.StatsCache(confids[key],correct,n_bins) ] for key in confids
            }
    # print([ print(f'{k}:{stats[k][0].augrc}') for k in stats])
    stats_df = pd.DataFrame( {  
                                'AUGRC': { k:stats[k][0].augrc for k in stats },
                                'AURC': { k:stats[k][0].aurc for k in stats },
                                'AUROC_f': { k:metrics.failauc(stats[k][1]) for k in stats },
                                'FPR@95TPR': { k:metrics.fpr_at_95_tpr(stats[k][1]) for k in stats },
                                'ECE': { k:metrics.expected_calibration_error(stats[k][1]) for k in stats },
                                'MCE': { k:metrics.maximum_calibration_error(stats[k][1]) for k in stats },
                                'AP_ferr': { k:metrics.failap_err(stats[k][1]) for k in stats },
                                'AP_fsuc': { k:metrics.failap_suc(stats[k][1]) for k in stats },
                            } )
    filename = f'stats{model_opts}_{eval_name}.csv'
    filename_confids = f'confids{model_opts}_{eval_name}.csv'
    if os.path.exists(f'{cf.exp.dir}/analysis'):
        path = f'{cf.exp.dir}/analysis/{filename}'
        path_confids = f'{cf.exp.dir}/analysis/{filename_confids}'
    else:
        os.mkdir(f'{cf.exp.dir}/analysis')
        path = f'{cf.exp.dir}/analysis/{filename}'
        path_confids = f'{cf.exp.dir}/analysis/{filename_confids}'
    # Merge with any existing CSV so other families' results are preserved.
    merged_stats = _merge_csv_rows(stats_df, path)
    merged_confids = _merge_csv_cols(confids_df, path_confids)
    merged_stats.sort_values(by=['AUGRC']).to_csv(path)
    merged_confids.to_csv(path_confids)

#%%
def compute_metrics(module, study_name, cf, model_evaluations, eval_name:str, do_enabled:bool, model_opts:str='', n_bins:int=20, temp_scaled:bool=False, active: set | None = None, projections: set | None = None):
    active = set(ALL_FAMILIES) if active is None else set(active)
    projections = set(ALL_PROJECTIONS) if projections is None else set(projections)
    def gate(mode, family):
        return mode in projections and family in active
    if 'cifar' in cf.data.dataset:
        if eval_name == 'iid_test':
            key_dict = 'test_1'
            stats(module, study_name, cf, model_evaluations[key_dict], eval_name, do_enabled, model_opts=model_opts, n_bins=n_bins, temp_scaled=temp_scaled, active=active, projections=projections)
        elif eval_name == 'iid_val':
            key_dict = 'val'
            stats(module, study_name, cf, model_evaluations[key_dict], eval_name, do_enabled, model_opts=model_opts, n_bins=n_bins, temp_scaled=temp_scaled, active=active, projections=projections)
        elif eval_name == 'iid_test_corruptions':
            key_dict = 'test_2_sampled_20'
            evaluations = model_evaluations[key_dict]
            # Make sure that all the keys have the same length of 
            lengths = [ len(evaluations[key]) for key in evaluations.keys() if evaluations[key] is not None ]
            assert len(set(lengths))==1, 'Evaluations do not have the same dimensions'
            n_samples = lengths[0]//5 # Corruptions of 5 different types
            for i in tqdm(range(5)):
                logger.info(f'Evaluating test set with corruption type {i+1}...')
                evaluations_grouped = {key:evaluations[key][n_samples*i:n_samples*(i+1)] for key in evaluations.keys() if evaluations[key] is not None}
                stats(module, study_name, cf, evaluations_grouped, eval_name+f'_{i+1}', do_enabled, model_opts=model_opts, n_bins=n_bins, temp_scaled=temp_scaled, active=active, projections=projections)
        elif 'ood' in eval_name:
            # pick only correct predictions of the iid test set. Predictions are based on max softmax.
            evaluations_iid = model_evaluations['test_1']
            ood_set_number = [set_name for set_name in model_evaluations.keys() if set_name!='test_1']
            assert len(ood_set_number)==1, 'Just one OOD set should be include at a time...'
            evaluations_ood = model_evaluations[ood_set_number[0]]
            # predictions for ood samples should be all incorrect
            evaluations_ood['correct'] = torch.zeros_like( evaluations_ood['correct'] ).long()
            # filtering criteria
            correct = evaluations_iid['correct']
            # print(evaluations_iid.keys())
            evaluations_iid_filtered = { key:(evaluations_iid[key][correct==1] if ('_dist' not in key) else None) for key in evaluations_iid.keys() }
            if do_enabled:
                # predictions for ood samples should be all incorrect
                evaluations_ood['correct_mcd'] = torch.zeros_like( evaluations_ood['correct_mcd'] ).long()
                # filtering criteria
                correct_mcd = evaluations_iid['correct_mcd']
                evaluations_filtered_mcd = { key:evaluations_iid[key][correct_mcd==1] for key in evaluations_iid.keys() if (('_dist' in key) or ('_mcd' in key)) }
                print(evaluations_iid_filtered.keys())
                print(evaluations_filtered_mcd.keys())
                evaluations_iid_filtered = evaluations_iid_filtered | evaluations_filtered_mcd
            assert evaluations_iid_filtered.keys()==evaluations_ood.keys(), 'IID and OOD dictionaries should have the same keys...'
            keys = evaluations_iid_filtered.keys()
            lengths = [ (key,len(evaluations_iid_filtered[key]),len(evaluations_ood[key])) for key in keys if ((evaluations_iid_filtered[key] is not None) and (evaluations_ood[key] is not None)) ]
            # print(lengths)
            evaluations_joint = {key:torch.concat([evaluations_iid_filtered[key],evaluations_ood[key]],dim=0) for key in keys if ((evaluations_iid_filtered[key] is not None) and (evaluations_ood[key] is not None)) }
            stats(module, study_name, cf, evaluations_joint, eval_name, do_enabled, model_opts=model_opts, n_bins=n_bins, temp_scaled=temp_scaled, active=active, projections=projections)
    elif 'tiny' in cf.data.dataset:
        logger.info(f'Evaluating {eval_name} with {cf.data.dataset}')
        if eval_name == 'iid_test':
            key_dict = 'test_1'
            stats(module, study_name, cf, model_evaluations[key_dict], eval_name, do_enabled, model_opts=model_opts, n_bins=n_bins, temp_scaled=temp_scaled, active=active, projections=projections)
        elif eval_name == 'iid_val':
            key_dict = 'val'
            stats(module, study_name, cf, model_evaluations[key_dict], eval_name, do_enabled, model_opts=model_opts, n_bins=n_bins, temp_scaled=temp_scaled, active=active, projections=projections) 
        elif 'ood' in eval_name:
            # pick only correct predictions of the iid test set. Predictions are based on max softmax.
            evaluations_iid = model_evaluations['test_1']
            ood_set_number = [set_name for set_name in model_evaluations.keys() if set_name!='test_1']
            assert len(ood_set_number)==1, 'Just one OOD set should be include at a time...'
            evaluations_ood = model_evaluations[ood_set_number[0]]
            # predictions for ood samples should be all incorrect
            evaluations_ood['correct'] = torch.zeros_like( evaluations_ood['correct'] ).long()
            # filtering criteria
            correct = evaluations_iid['correct']
            # print(evaluations_iid.keys())
            evaluations_iid_filtered = { key:(evaluations_iid[key][correct==1] if ('_dist' not in key) else None) for key in evaluations_iid.keys() }
            if do_enabled:
                # predictions for ood samples should be all incorrect
                evaluations_ood['correct_mcd'] = torch.zeros_like( evaluations_ood['correct_mcd'] ).long()
                # filtering criteria
                correct_mcd = evaluations_iid['correct_mcd']
                evaluations_filtered_mcd = { key:evaluations_iid[key][correct_mcd==1] for key in evaluations_iid.keys() if (('_dist' in key) or ('_mcd' in key)) }
                print(evaluations_iid_filtered.keys())
                print(evaluations_filtered_mcd.keys())
                evaluations_iid_filtered = evaluations_iid_filtered | evaluations_filtered_mcd
            assert evaluations_iid_filtered.keys()==evaluations_ood.keys(), 'IID and OOD dictionaries should have the same keys...'
            keys = evaluations_iid_filtered.keys()
            lengths = [ (key,len(evaluations_iid_filtered[key]),len(evaluations_ood[key])) for key in keys if ((evaluations_iid_filtered[key] is not None) and (evaluations_ood[key] is not None)) ]
            # print(lengths)
            evaluations_joint = {key:torch.concat([evaluations_iid_filtered[key],evaluations_ood[key]],dim=0) for key in keys if ((evaluations_iid_filtered[key] is not None) and (evaluations_ood[key] is not None)) }
            stats(module, study_name, cf, evaluations_joint, eval_name, do_enabled, model_opts=model_opts, n_bins=n_bins, temp_scaled=temp_scaled, active=active, projections=projections)
#%%
# def compute_metrics_eval(module, study_name, cf, model_evaluations, eval_name:str, do_enabled:bool, model_opts:str='', n_bins:int=20, temp_scaled:bool=False):
#     # if cf.data?
#     if eval_name == 'iid_test':
#         key_dict = 'test_1'
#         stats(module, study_name, cf, model_evaluations[key_dict], eval_name, do_enabled, model_opts=model_opts, n_bins=n_bins, temp_scaled=temp_scaled, active=active, projections=projections)
#     elif eval_name == 'iid_val':
#         key_dict = 'val'
#         stats(module, study_name, cf, model_evaluations[key_dict], eval_name, do_enabled, model_opts=model_opts, n_bins=n_bins, temp_scaled=temp_scaled, active=active, projections=projections)
#     elif eval_name == 'iid_test_corruptions':
#         key_dict = 'test_2'
#         evaluations = model_evaluations[key_dict]
#         # Make sure that all the keys have the same length of 
#         lengths = [ len(evaluations[key]) for key in evaluations.keys() if evaluations[key] is not None ]
#         assert len(set(lengths))==1, 'Evaluations do not have the same dimensions'
#         n_samples = lengths[0]//5 # Corruptions of 5 different types
#         for i in tqdm(range(5)):
#             logger.info(f'Evaluating test set with corruption type {i+1}...')
#             evaluations_grouped = {key:evaluations[key][n_samples*i:n_samples*(i+1)] for key in evaluations.keys() if evaluations[key] is not None}
#             stats(module, study_name, cf, evaluations_grouped, eval_name+f'_{i+1}', do_enabled, model_opts=model_opts, n_bins=n_bins, temp_scaled=temp_scaled, active=active, projections=projections)
#     elif 'ood' in eval_name:
#         # pick only correct predictions of the iid test set. Predictions are based on max softmax.
#         evaluations_iid = model_evaluations['test_1']
#         ood_set_number = [set_name for set_name in model_evaluations.keys() if set_name!='test_1']
#         assert len(ood_set_number)==1, 'Just one OOD set should be include at a time...'
#         evaluations_ood = model_evaluations[ood_set_number[0]]
#         # predictions for ood samples should be all incorrect
#         evaluations_ood['correct'] = torch.zeros_like( evaluations_ood['correct'] ).long()
#         # filtering criteria
#         correct = evaluations_iid['correct']
#         evaluations_iid_filtered = { key:(evaluations_iid[key][correct==1] if (('_dist' not in key) or ('_mcd' not in key)) else None) for key in evaluations_iid.keys() }
#         if do_enabled:
#             # predictions for ood samples should be all incorrect
#             evaluations_ood['correct_mcd'] = torch.zeros_like( evaluations_ood['correct_mcd'] ).long()
#             # filtering criteria
#             correct_mcd = evaluations_iid['correct_mcd']
#             evaluations_filtered_mcd = { key:evaluations_iid[key][correct_mcd==1] for key in evaluations_iid.keys() if (('_dist' in key) or ('_mcd' in key)) }
#             evaluations_iid_filtered = evaluations_iid_filtered | evaluations_filtered_mcd
#         assert evaluations_iid_filtered.keys()==evaluations_ood.keys(), 'IID and OOD dictionaries should have the same keys...'
#         keys = evaluations_iid_filtered.keys()
#         lengths = [(key,len(evaluations_iid_filtered[key]),len(evaluations_ood[key])) for key in keys]
#         # print(lengths)
#         evaluations_joint = {key:torch.concat([evaluations_iid_filtered[key],evaluations_ood[key]],dim=0) for key in keys if ((evaluations_iid_filtered[key] is not None) and (evaluations_ood[key] is not None)) }
#         stats(module, study_name, cf, evaluations_joint, eval_name, do_enabled, model_opts=model_opts, n_bins=n_bins, temp_scaled=temp_scaled, active=active, projections=projections)
        # change here
    # if do_enabled:
    #     score_methods, score_methods_do = load_score_methods(cf, module, study_name, do_enabled)    
    #     gradnorm_score = GradNorm(module, study_name, cf)
    #     encoded_distribution = model_evaluations[key_dict]['encoded_dist']
    #     logits_distribution = model_evaluations[key_dict]['logits_dist']
    #     softmax_distribution = model_evaluations[key_dict]['softmax_scaled_dist'] if temp_scaled else model_evaluations[key_dict]['softmax_dist']
    #     # model_evaluations['train']['softmax_scaled_dist'] if temp_scaled else model_evaluations['train']['softmax_dist'] 
    #     confid_distribution = model_evaluations[key_dict]['confid_dist']
    #     correct_distribution = model_evaluations[key_dict]['correct_mcd']
    #     residuals_distribution = 1-correct_distribution
    #     mcd_confids = {
    #         'MCD-Confidence' :  confid_distribution.mean(dim=1),
    #         'MCD-MSR' :         scores_funcs.mcd_function(scores_funcs.maximum_softmax_response, softmax_distribution),
    #         'MCD-EMSR' :        scores_funcs.mcd_expected_function(scores_funcs.maximum_softmax_response, softmax_distribution),
    #         'MCD-PE' :          scores_funcs.mcd_function(scores_funcs.predictive_entropy, softmax_distribution),
    #         'MCD-EPE' :         scores_funcs.mcd_expected_function(scores_funcs.predictive_entropy, softmax_distribution),
    #         'MCD-MLS' :         scores_funcs.mcd_function(scores_funcs.maximum_logit_score, logits_distribution),
    #         'MCD-EMLS' :        scores_funcs.mcd_expected_function(scores_funcs.maximum_logit_score, logits_distribution),    
    #         'MCD-PCE' :         scores_funcs.mcd_function(scores_funcs.predictive_collision_entropy, softmax_distribution),
    #         'MCD-EPCE' :        scores_funcs.mcd_expected_function(scores_funcs.predictive_collision_entropy, softmax_distribution),
    #         'MCD-GE' :          scores_funcs.mcd_function(scores_funcs.guessing_entropy, softmax_distribution),
    #         'MCD-EGE' :         scores_funcs.mcd_expected_function(scores_funcs.guessing_entropy, softmax_distribution),
    #         'MCD-Energy' :      scores_funcs.mcd_function(scores_funcs.energy, logits_distribution, T=score_methods_do['temperature_scale_dist'].temperature),
    #         'MCD-EEnergy' :     scores_funcs.mcd_expected_function(scores_funcs.energy, logits_distribution, T=score_methods_do['temperature_scale_dist'].temperature),    
    #         'MCD-Maha':         scores_funcs.mcd_function(score_methods_do['maha_distance_dist'].get_scores, encoded_distribution),
    #         'MCD-EMaha':        scores_funcs.mcd_expected_function(score_methods_do['maha_distance_dist'].get_scores, encoded_distribution),
    #         'MCD-ViM':          scores_funcs.mcd_function(score_methods_do['vim_score_dist'].get_scores, encoded_distribution),
    #         'MCD-EViM':         scores_funcs.mcd_expected_function(score_methods_do['vim_score_dist'].get_scores, encoded_distribution),
    #         'MCD-Residual':     scores_funcs.mcd_function(score_methods_do['residual_score_dist'].get_scores, encoded_distribution),
    #         'MCD-EResidual':    scores_funcs.mcd_expected_function(score_methods_do['residual_score_dist'].get_scores, encoded_distribution),
    #         'MCD-NeCo':         scores_funcs.mcd_function(score_methods_do['neco_score_dist'].get_scores, encoded_distribution),
    #         'MCD-ENeCo':        scores_funcs.mcd_expected_function(score_methods_do['neco_score_dist'].get_scores, encoded_distribution),
    #         'MCD-pNML':         scores_funcs.mcd_function(score_methods_do['pnml_score_dist'].get_scores, encoded_distribution),
    #         'MCD-EpNML':        scores_funcs.mcd_expected_function(score_methods_do['pnml_score_dist'].get_scores, encoded_distribution),
    #         'MCD-KLMatching' :  scores_funcs.mcd_function(score_methods_do['klmatching_score_dist'].get_scores, softmax_distribution),
    #         'MCD-EKLMatching' : scores_funcs.mcd_expected_function(score_methods_do['klmatching_score_dist'].get_scores, softmax_distribution),
    #         'MCD-GEN' :         scores_funcs.mcd_function(score_methods_do['generalized_entropy_dist'].get_scores, softmax_distribution),
    #         'MCD-EGEN' :        scores_funcs.mcd_expected_function(score_methods_do['generalized_entropy_dist'].get_scores, softmax_distribution),
    #         'MCD-REN' :         scores_funcs.mcd_function(score_methods_do['renyi_entropy_dist'].get_scores, softmax_distribution),
    #         'MCD-EREN' :        scores_funcs.mcd_expected_function(score_methods_do['renyi_entropy_dist'].get_scores, softmax_distribution),
    #         'MCD-TEN' :         scores_funcs.mcd_function(score_methods_do['tsallis_entropy_dist'].get_scores, softmax_distribution),
    #         'MCD-ETEN' :        scores_funcs.mcd_expected_function(score_methods_do['tsallis_entropy_dist'].get_scores, softmax_distribution),
    #         'MCD-GradNorm' :    scores_funcs.mcd_function(gradnorm_score.get_scores, encoded_distribution),
    #         'MCD-MI' :          scores_funcs.mcd_mutual_information(softmax_distribution),
    #     } 
    #     mcd_stats = {
    #                     key:[RiskCoverageStats(confids=mcd_confids[key], residuals=residuals_distribution), metrics.StatsCache(mcd_confids[key],correct_distribution,n_bins) ] for key in mcd_confids   
    #                 }

    #     mcd_stats_df = pd.DataFrame( {  
    #                                     'AUGRC': { k:mcd_stats[k][0].augrc for k in mcd_stats },
    #                                     'AURC': { k:mcd_stats[k][0].aurc for k in mcd_stats },
    #                                     'AUROC_f': { k:metrics.failauc(mcd_stats[k][1]) for k in mcd_stats },
    #                                     'FPR95': { k:metrics.fpr_at_95_tpr(mcd_stats[k][1]) for k in mcd_stats },
    #                                     'ECE': { k:metrics.expected_calibration_error(mcd_stats[k][1]) for k in mcd_stats },
    #                                     'MCE': { k:metrics.maximum_calibration_error(mcd_stats[k][1]) for k in mcd_stats },
    #                                 } )
    #     filename = f'mcd_stats_{model_name}_{eval_name}.csv'
    #     if os.path.exists(f'{cf.exp.dir}/analysis'):
    #         path = f'{cf.exp.dir}/analysis/{filename}'
    #     else:
    #         os.mkdir(f'{cf.exp.dir}/analysis')
    #         path = f'{cf.exp.dir}/analysis/{filename}'
    #     mcd_stats_df.sort_values(by=['AUGRC']).to_csv(path)
    # else:
    #     score_methods = load_score_methods(cf, module, study_name, do_enabled)

    # gradnorm_score = GradNorm(module, study_name, cf)
    # encoded = model_evaluations[key_dict]['encoded']
    # logits = model_evaluations[key_dict]['logits']
    # softmax = model_evaluations[key_dict]['softmax_scaled'] if temp_scaled else model_evaluations[key_dict]['softmax']
    # # model_evaluations['train']['softmax_scaled'] if temp_scaled else model_evaluations['train']['softmax']
    # confid = model_evaluations[key_dict]['confid']
    # correct = model_evaluations[key_dict]['correct']
    # residuals = 1-correct

    # confids= {
    #             'MSR' :         scores_funcs.maximum_softmax_response(softmax),
    #             'PE' :          scores_funcs.predictive_entropy(softmax),
    #             'MLS' :         scores_funcs.maximum_logit_score(logits),
    #             'PCE' :         scores_funcs.predictive_collision_entropy(softmax),
    #             'GE' :          scores_funcs.guessing_entropy(softmax),
    #             'Energy' :      scores_funcs.energy(logits, T=score_methods['temperature_scale'].temperature),
    #             'Maha':         score_methods['maha_distance'].get_scores(encoded),
    #             'ViM':          score_methods['vim_score'].get_scores(encoded),
    #             'Residual':     score_methods['residual_score'].get_scores(encoded),    
    #             'NeCo':         score_methods['neco_score'].get_scores(encoded),
    #             'pNML':         score_methods['pnml_score'].get_scores(encoded),
    #             'KLMatching':   score_methods['klmatching_score'].get_scores(softmax),
    #             'GEN' :         score_methods['generalized_entropy'].get_scores(softmax),
    #             'REN' :         score_methods['renyi_entropy'].get_scores(softmax),
    #             'TEN' :         score_methods['tsallis_entropy'].get_scores(softmax),    
    #             'GradNorm' :    gradnorm_score.get_scores(encoded, use_cuda=True),    
    #             'Confidence' :  confid,
    # }
    # stats = {
    #             key:[RiskCoverageStats(confids=confids[key], residuals=residuals), metrics.StatsCache(confids[key],correct,n_bins) ] for key in confids
    #         }
    # stats_df = pd.DataFrame( {  
    #                             'AUGRC': { k:stats[k][0].augrc for k in stats },
    #                             'AURC': { k:stats[k][0].aurc for k in stats },
    #                             'AUROC_f': { k:metrics.failauc(stats[k][1]) for k in stats },
    #                             'FPR95': { k:metrics.fpr_at_95_tpr(stats[k][1]) for k in stats },
    #                             'ECE': { k:metrics.expected_calibration_error(stats[k][1]) for k in stats },
    #                             'MCE': { k:metrics.maximum_calibration_error(stats[k][1]) for k in stats },
    #                         } )
    # filename = f'stats_{model_name}_{eval_name}.csv'
    # if os.path.exists(f'{cf.exp.dir}/analysis'):
    #     path = f'{cf.exp.dir}/analysis/{filename}'
    # else:
    #     os.mkdir(f'{cf.exp.dir}/analysis')
    #     path = f'{cf.exp.dir}/analysis/{filename}'
    # # mcd_stats_df.to_csv(path)
    # stats_df.sort_values(by=['AUGRC']).to_csv(path)

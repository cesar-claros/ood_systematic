"""Backwards-compatibility shim for the scores_funcs split.

The CSF scoring functions used to live in this single module. They have been
split into four files under `src/csfs/`:

  - `src/csfs/_validators.py`  — validation decorators + helpers
  - `src/csfs/base_detectors.py` — MSR, PE, MLS, MCS, PCE, GE, Energy
  - `src/csfs/entropy_funcs.py`  — generalized / renyi / tsallis entropies
  - `src/csfs/mcd.py`            — MCD adapters and MCD-specific CSFs

This shim re-exports all of them under their original `src.scores_funcs.*`
paths so existing imports of the form `from src import scores_funcs;
scores_funcs.maximum_softmax_response(...)` keep working. New code should
import from the canonical locations above.
"""

from src.csfs._validators import (
    ArrayType,
    T,
    _assert_softmax_logit_finite,
    _assert_softmax_logit_distribution,
    _assert_softmax_numerically_stable,
    validate_logit,
    validate_softmax,
    validate_softmax_logit_distribution,
)
from src.csfs.base_detectors import (
    energy,
    guessing_entropy,
    maximum_cosine_similarity,
    maximum_logit_score,
    maximum_softmax_response,
    predictive_collision_entropy,
    predictive_entropy,
)
from src.csfs.entropy_funcs import (
    generalized_entropy,
    renyi_entropy,
    tsallis_entropy,
)
from src.csfs.mcd import (
    mcd_expected_function,
    mcd_function,
    mcd_mutual_information,
    mcd_softmax_variance,
    mcd_watanabe_aic,
)

__all__ = [
    # Validators
    "ArrayType",
    "T",
    "_assert_softmax_logit_finite",
    "_assert_softmax_logit_distribution",
    "_assert_softmax_numerically_stable",
    "validate_logit",
    "validate_softmax",
    "validate_softmax_logit_distribution",
    # Base detectors
    "energy",
    "guessing_entropy",
    "maximum_cosine_similarity",
    "maximum_logit_score",
    "maximum_softmax_response",
    "predictive_collision_entropy",
    "predictive_entropy",
    # Tunable entropies
    "generalized_entropy",
    "renyi_entropy",
    "tsallis_entropy",
    # MCD adapters + scorers
    "mcd_expected_function",
    "mcd_function",
    "mcd_mutual_information",
    "mcd_softmax_variance",
    "mcd_watanabe_aic",
]

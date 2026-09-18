"""AUROC input validation in the experiment driver. Written by the 2026-09-17 audit as expected failures against commit bcfd5e1
(which ranked nonfinite scores); the driver now rejects them, so the cases are plain tests."""

import ast
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

ScoreFunction = Callable[[NDArray[np.float64], NDArray[np.float64]], float]


@pytest.fixture
def recorded_auroc() -> ScoreFunction:
    """Load the exact metric function without importing the GPU training stack.

    Returns:
        The committed driver's AUROC function with its NumPy dependency.

    Raises:
        StopIteration: The driver no longer defines the audited function.
    """
    source = Path(__file__).parents[1] / "x8_pool_a/adaptation_trajectory.py"
    tree = ast.parse(source.read_text())
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "auroc"
    )
    namespace: dict[str, object] = {"np": np}
    exec(
        compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return cast(ScoreFunction, namespace["auroc"])


def test_recorded_auroc_handles_valid_ties(recorded_auroc: ScoreFunction) -> None:
    """Three wins and one tie among four ID/OOD pairs give AUROC 0.875."""
    assert recorded_auroc(np.array([1.0, 2.0]), np.array([0.0, 1.0])) == 0.875


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("invalid_in_id", [False, True])
def test_recorded_auroc_rejects_nonfinite_scores(
    recorded_auroc: ScoreFunction, invalid: float, invalid_in_id: bool
) -> None:
    """An invalid raw score must not become an apparently valid AUROC."""
    id_scores, ood_scores = np.array([0.2, 0.8]), np.array([0.1, 0.9])
    if invalid_in_id:
        id_scores[0] = invalid
    else:
        ood_scores[0] = invalid
    with pytest.raises(ValueError):
        recorded_auroc(id_scores, ood_scores)

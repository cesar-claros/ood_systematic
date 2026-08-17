"""Post-training head-rotation operator on frozen features (control A0).

Rotates each classifier row in direction space while preserving row norms
and the bias, so the only degree of freedom exercised is the self-duality
angle X1 Theorem 2 analyzes. Two families:

- toward: per-row geodesic interpolation from the learned direction to the
  centered class-mean direction (tau = 1 reaches exact direction
  self-duality).
- away: per-row rotation by angle theta into a random unit direction drawn
  in the orthogonal complement of the class-mean span (the X1 perturbation
  mode). The large-angle response is quenched-draw dependent (Theorem 2.3),
  so several draws are generated per angle.
"""
from __future__ import annotations

import numpy as np

from pilot0.geometry import FeatureModel


def _slerp_rows(w: np.ndarray, targets: np.ndarray, tau: float) -> np.ndarray:
    """Norm-preserving spherical interpolation of each row toward its target.

    Args:
        w: (C, D) classifier rows.
        targets: (C, D) unit target directions.
        tau: interpolation fraction in [0, 1] of the initial angle.

    Returns:
        (C, D) rotated rows with original norms.
    """
    norms = np.linalg.norm(w, axis=1, keepdims=True)
    w_hat = w / norms
    cos = np.clip(np.einsum("cd,cd->c", w_hat, targets), -1.0, 1.0)
    angle = np.arccos(cos)
    out = np.empty_like(w_hat)
    small = angle < 1e-9
    out[small] = w_hat[small]
    if (~small).any():
        ang = angle[~small][:, None]
        sin = np.sin(ang)
        out[~small] = (np.sin((1.0 - tau) * ang) / sin * w_hat[~small]
                       + np.sin(tau * ang) / sin * targets[~small])
    return out * norms


def rotate_toward(w: np.ndarray, model: FeatureModel,
                  tau: float) -> np.ndarray:
    """Rotate rows toward the centered class-mean directions by fraction tau."""
    targets = model.class_means / model.radii[:, None]
    return _slerp_rows(w.astype(np.float64), targets, tau)


def rotate_away(w: np.ndarray, model: FeatureModel, theta: float,
                rng: np.random.Generator) -> np.ndarray:
    """Rotate each row by angle theta into a random span-complement direction.

    Args:
        w: (C, D) classifier rows.
        model: fitted feature model (provides the span to avoid).
        theta: rotation angle in radians.
        rng: generator for the quenched complement draw.

    Returns:
        (C, D) rotated rows with original norms.
    """
    w64 = w.astype(np.float64)
    norms = np.linalg.norm(w64, axis=1, keepdims=True)
    w_hat = w64 / norms
    v = rng.standard_normal(w64.shape)
    v -= (v @ model.span_basis) @ model.span_basis.T
    v -= np.einsum("cd,cd->c", v, w_hat)[:, None] * w_hat
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return (np.cos(theta) * w_hat + np.sin(theta) * v) * norms


def head_grid(w: np.ndarray, model: FeatureModel, taus: tuple[float, ...],
              thetas_deg: tuple[float, ...], n_draws: int,
              seed: int) -> list[dict]:
    """Build the full operator grid of head states.

    Returns:
        List of records {name, kind, param, draw, w} including the
        baseline (kind='baseline', w unchanged).
    """
    states: list[dict] = [
        {"name": "baseline", "kind": "baseline", "param": 0.0, "draw": 0,
         "w": w.astype(np.float64)}]
    for tau in taus:
        states.append({"name": f"toward_{tau:g}", "kind": "toward",
                       "param": tau, "draw": 0,
                       "w": rotate_toward(w, model, tau)})
    for theta in thetas_deg:
        for draw in range(n_draws):
            rng = np.random.default_rng(seed + 1000 * draw + int(theta))
            states.append({
                "name": f"away_{theta:g}deg_d{draw}", "kind": "away",
                "param": float(theta), "draw": draw,
                "w": rotate_away(w, model, np.radians(theta), rng)})
    return states

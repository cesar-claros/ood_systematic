"""Directional retention measures between a reference and an adapted representation, from ID data only (2026-09-17).

Motivation: on Pets the within-family detectability loss under full fine-tuning occurred with no radial motion at all, so a retention
measure must see changes in class-mean relations and within-class structure, not norms. Detectors refit on the adapted features are
invariant to a joint orthogonal rotation of the feature space, so the primary measures must be too.

Measures (all in [0, 1], 1 = fully retained):
  class_mean_cka: linear CKA between the two centered class-mean matrices (C rows). Invariant to orthogonal rotation, translation and
      uniform scaling; moves when the relations among class means change (which classes are near which).
  class_centred_paired_cka: paired linear CKA of within-class residuals (h minus the class mean) on corresponding ID examples.
      Same invariances; moves when within-class structure changes with class means fixed.
  between_class_subspace_overlap: mean squared cosine of the principal angles between the reference and adapted centered class-mean
      subspaces (numerical rank via descriptors_v3.mean_span_projector). Coordinate dependent: a joint rotation moves it although no
      refit detector changes; report it as a secondary, coordinate-level diagnostic only.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from descriptors_v3 import mean_span_projector, paired_linear_cka  # noqa: E402


def centred_class_means(H: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64); y = np.asarray(y)
    if H.ndim != 2 or len(H) != len(y): raise ValueError("features must be (N, D) with one label per row")
    counts = np.bincount(y, minlength=n_classes)
    if (counts == 0).any(): raise ValueError(f"classes without examples: {np.where(counts == 0)[0].tolist()}")
    M = np.stack([H[y == c].mean(0) for c in range(n_classes)]); return M - M.mean(0)


def class_mean_cka(H0, y0, Ht, yt, n_classes: int) -> float:
    M0, Mt = centred_class_means(H0, y0, n_classes), centred_class_means(Ht, yt, n_classes)
    ids = np.arange(n_classes); return float(paired_linear_cka(M0, Mt, ids, ids))


def class_centred_paired_cka(H0, y0, ids0, Ht, yt, idst, n_classes: int) -> float:
    H0 = np.asarray(H0, dtype=np.float64); Ht = np.asarray(Ht, dtype=np.float64); y0 = np.asarray(y0); yt = np.asarray(yt)
    if not np.array_equal(np.asarray(ids0), np.asarray(idst)) or not np.array_equal(y0, yt): raise ValueError("examples and labels must correspond in the same order")
    M0, Mt = centred_class_means(H0, y0, n_classes), centred_class_means(Ht, yt, n_classes)
    R0 = (H0 - H0.mean(0)) - M0[y0]; Rt = (Ht - Ht.mean(0)) - Mt[yt]
    return float(paired_linear_cka(R0, Rt, ids0, idst))


def between_class_subspace_overlap(H0, y0, Ht, yt, n_classes: int, rank_rtol: float | None = None) -> float:
    P0, _ = mean_span_projector(np.asarray(H0, dtype=np.float64), np.asarray(y0), n_classes, rank_rtol=rank_rtol)
    Pt, _ = mean_span_projector(np.asarray(Ht, dtype=np.float64), np.asarray(yt), n_classes, rank_rtol=rank_rtol)
    r0, rt = int(round(np.trace(P0))), int(round(np.trace(Pt)))
    if min(r0, rt) == 0: raise ValueError("a class-mean subspace has rank zero")
    return float(np.trace(P0 @ Pt) / min(r0, rt))   # = sum of squared principal-angle cosines / min rank


def retention_record(H0, y0, ids0, Ht, yt, idst, n_classes: int) -> dict[str, float]:
    return {"cm_cka_ref": class_mean_cka(H0, y0, Ht, yt, n_classes),
            "cc_cka_ref": class_centred_paired_cka(H0, y0, ids0, Ht, yt, idst, n_classes),
            "bc_overlap_ref": between_class_subspace_overlap(H0, y0, Ht, yt, n_classes)}

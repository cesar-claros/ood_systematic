"""Validation decorators and assertions for CSF scoring functions.

Each scoring function in `src/csfs/base_detectors.py`,
`src/csfs/entropy_funcs.py`, and `src/csfs/mcd.py` is decorated with one of
the validators here to assert that its input softmax/logit tensor is finite
(no NaN/Inf) and, for softmax inputs, numerically stable. The decorators
preserve return values; they only raise AssertionError on bad input.
"""

import logging
from typing import Any, Callable, TypeVar, cast

import torch


ArrayType = torch.Tensor

T = TypeVar(
    "T",
    Callable[[ArrayType], ArrayType],
    Callable[[ArrayType, ArrayType], ArrayType],
)


def _assert_softmax_logit_finite(softmax_logit: ArrayType):
    assert torch.isfinite(softmax_logit).all(), "NaN or INF in softmax/logit output"


def _assert_softmax_logit_distribution(softmax_logit: ArrayType):
    assert softmax_logit.ndimension() > 2, (
        "softmax/logit distribution needs to be shaped as (N,C,M), where N "
        "is the number of instances, C is the number of classes, and M is "
        "the number of Monte Carlo dropout samples"
    )
    assert softmax_logit.shape[2] > 1, (
        "softmax/logit distribution has only one Monte Carlo sample"
    )


def _assert_softmax_numerically_stable(softmax: ArrayType):
    msr, _ = softmax.max(dim=1)
    errors = (msr == 1) & ((softmax > 0) & (softmax < 1)).any(dim=1)
    if softmax.dtype != torch.float64:
        logging.warning("Softmax is not 64bit, not checking for numerical stability")
        return
    # alert if more than 10% are erroneous
    assert (
        errors.float().mean() < 0.1
    ), f"Numerical errors in softmax: {errors.float().mean() * 100:.2f}%"


def validate_softmax(func: T) -> T:
    """Decorator: assert softmax args are finite and numerically stable."""

    def _inner_wrapper(*args: ArrayType, **kwargs) -> ArrayType:
        for arg in args:
            _assert_softmax_logit_finite(arg)
            _assert_softmax_numerically_stable(arg)
        return func(*args, **kwargs)

    return cast(T, _inner_wrapper)


def validate_logit(func: T) -> T:
    """Decorator: assert logit args are finite."""

    def _inner_wrapper(*args: ArrayType, **kwargs) -> ArrayType:
        for arg in args:
            _assert_softmax_logit_finite(arg)
        return func(*args, **kwargs)

    return cast(T, _inner_wrapper)


def validate_softmax_logit_distribution(func: T) -> T:
    """Decorator: assert MCD-distribution args have shape (N,C,M) with M>1 and are finite."""

    def _inner_wrapper(*args: ArrayType, **kwargs) -> ArrayType:
        for arg in args:
            if torch.is_tensor(arg):
                _assert_softmax_logit_finite(arg)
                _assert_softmax_logit_distribution(arg)
        return func(*args, **kwargs)

    return cast(T, _inner_wrapper)

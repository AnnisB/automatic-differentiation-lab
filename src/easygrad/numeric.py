"""Numerical differentiation: finite differences and the complex step."""

from typing import Callable

import numpy as np


def finite_difference(
    f: Callable[[float], float],
    x: float,
    h: float,
    scheme: str = "central",
) -> float:
    """Approximate ``f'(x)`` by a finite difference.

    scheme:
        ``"forward"``, ``"backward"``, or ``"central"``.

    Exercise
    --------
    Implement the three schemes (see docs/primer/numerical.md). Raise
    ``ValueError`` for an unknown ``scheme``.
    """
    if scheme == "forward":
        return (f(x+h)-f(x))/h
    
    elif scheme == "backward":
        return (f(x)-f(x-h))/h

    elif scheme == "central":
        return (f(x+h)-f(x-h))/(2*h)  

    else:
        raise ValueError("unknown scheme")


def complex_step(
    f: Callable[[complex], complex],
    x: float,
    h: float = 1e-200,
) -> float:
    """Approximate ``f'(x)`` by the complex-step method.

    ``f`` must be implemented with operations that hold for complex inputs

    Exercise
    --------
    Implement the complex-step estimate (see docs/primer/complex-step.md).
    """
    return np.imag(f(x+1j*h)) / h


# functions for the docs
def error_curve(
    f: Callable[[float], float],
    fprime: Callable[[float], float],
    x: float,
    hs: np.ndarray,
    scheme: str = "central",
) -> np.ndarray:
    """Absolute error of a finite-difference scheme over a range of steps."""
    exact = fprime(x)
    return np.array(
        [abs(finite_difference(f, x, float(h), scheme) - exact) for h in hs]
    )


def complex_step_error_curve(
    f: Callable[[complex], complex],
    fprime: Callable[[float], float],
    x: float,
    hs: np.ndarray,
) -> np.ndarray:
    """Absolute error of the complex step over a range of steps."""
    exact = fprime(x)
    return np.array([abs(complex_step(f, x, float(h)) - exact) for h in hs])

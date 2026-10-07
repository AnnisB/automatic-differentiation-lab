"""Scalar forward mode via dual number"""

import math
from typing import Callable


class Dual:
    """A dual number ``primal + tangent * eps`` with ``eps**2 = 0``."""

    __slots__ = ("primal", "tangent")

    def __init__(self, primal: float, tangent: float = 0.0) -> None:
        self.primal = float(primal)
        self.tangent = float(tangent)

    # helpers
    @staticmethod
    def _coerce(other: "Dual | float") -> "Dual":
        return other if isinstance(other, Dual) else Dual(other, 0.0)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Dual({self.primal}, {self.tangent})"

    def __eq__(self, other: object) -> bool:  # pragma: no cover - convenience
        o = self._coerce(other) if isinstance(other, (int, float, Dual)) else None
        return o is not None and self.primal == o.primal and self.tangent == o.tangent

    # arithmetic (EXERCISE)
    # Implement the rule for the primal and tangent components.
    def __add__(self, other: "Dual | float") -> "Dual":
        other = self._coerce(other)
        return Dual(self.primal + other.primal, self.tangent + other.tangent)
        

    __radd__ = __add__

    def __sub__(self, other: "Dual | float") -> "Dual":
        other = self._coerce(other)
        return Dual(self.primal - other.primal, self.tangent - other.tangent)

    def __rsub__(self, other: "Dual | float") -> "Dual":
        # given once you have __sub__
        return self._coerce(other).__sub__(self)

    def __mul__(self, other: "Dual | float") -> "Dual":
        other = self._coerce(other)
        return Dual(self.primal * other.primal, self.primal * other.tangent + self.tangent * other.primal)

    __rmul__ = __mul__

    def __truediv__(self, other: "Dual | float") -> "Dual":
        other = self._coerce(other)
        return Dual(self.primal/other.primal, (self.tangent * other.primal - self.primal * other.tangent)/other.primal**2)

    def __rtruediv__(self, other: "Dual | float") -> "Dual":
        # given once you have __truediv__
        return self._coerce(other).__truediv__(self)

    def __pow__(self, p: float) -> "Dual":
        # constant real exponent only
        return Dual(self.primal**p, self.tangent*p*self.primal**(p-1))

    def __neg__(self) -> "Dual":
        return Dual(-self.primal, -self.tangent)


# elementary functions (EXERCISE)
# Each accepts a Dual or a plain float. The float branch is given.
# Fill in the Dual one. The return type is always Dual.
def exp(x: float | Dual) -> Dual:
    if isinstance(x, Dual):
        val = math.exp(x.primal)
        return Dual(val, x.tangent * val) 
    return Dual(math.exp(x))


def log(x: float | Dual) -> Dual:
    if isinstance(x, Dual):
        return Dual(math.log(x.primal), x.tangent * 1 / x.primal)
    return Dual(math.log(x))


def sin(x: float | Dual) -> Dual:
    if isinstance(x, Dual):
        return Dual(math.sin(x.primal), x.tangent * math.cos(x.primal))
    return Dual(math.sin(x))


def cos(x: float | Dual) -> Dual:
    if isinstance(x, Dual):
        return Dual(math.cos(x.primal), - x.tangent * math.sin(x.primal))
    return Dual(math.cos(x))


def tanh(x: float | Dual) -> Dual:
    if isinstance(x, Dual):
        val = math.tanh(x.primal)
        return Dual(val, x.tangent * (1 - val**2))
    return Dual(math.tanh(x))


def sqrt(x: float | Dual) -> Dual:
    if isinstance(x, Dual):
        val = math.sqrt(x.primal)
        return Dual(val, x.tangent * 1 / (2*val))
    return Dual(math.sqrt(x))


# utils
def derivative(f: Callable[[float | Dual], Dual], x: float) -> float:
    """Exact derivative ``f'(x)`` via one forward pass with seed tangent 1.

    ``f`` must be written in terms of arithmetic operators and the elementary
    functions exported by this module.
    """
    return f(Dual(x, 1.0)).tangent


def value_and_derivative(
    f: Callable[[float | Dual], Dual], x: float
) -> tuple[float, float]:
    """Return ``(f(x), f'(x))`` from a single forward pass."""
    out = f(Dual(x, 1.0))
    return out.primal, out.tangent

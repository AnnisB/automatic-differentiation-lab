"""Forward mode over numpy arrays."""

from typing import Callable, Sequence

import numpy as np

type ArrayLike = DualArray | np.ndarray | float | int


class DualArray:
    """An array-valued dual number ``primal + tangent * eps`` (``eps**2 = 0``)."""

    __slots__ = ("primal", "tangent")

    # tell numpy to defer `ndarray @ DualArray` etc. to our reflected operators
    __array_ufunc__ = None

    def __init__(self, primal: ArrayLike, tangent: ArrayLike | None = None) -> None:
        self.primal = np.asarray(primal, dtype=float)
        if tangent is None:
            self.tangent = np.zeros_like(self.primal)
        else:
            self.tangent = np.broadcast_to(np.asarray(tangent, dtype=float),
                                           self.primal.shape).astype(float)

    # helpers (given)
    @staticmethod
    def _coerce(other: ArrayLike) -> "DualArray":
        return other if isinstance(other, DualArray) else DualArray(other)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.primal.shape

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"DualArray(primal={self.primal}, tangent={self.tangent})"

    # arithmetic (EXERCISE)
    # Return a new DualArray
    def __add__(self, other: ArrayLike) -> "DualArray":
        o = self._coerce(other)
        return DualArray(self.primal + o.primal, self.tangent + o.tangent)
   
    __radd__ = __add__

    def __sub__(self, other: ArrayLike) -> "DualArray":
        o = self._coerce(other)
        return DualArray(self.primal - o.primal, self.tangent - o.tangent)

    def __rsub__(self, other: ArrayLike) -> "DualArray":
        return self._coerce(other).__sub__(self)

    def __mul__(self, other: ArrayLike) -> "DualArray":
        o = self._coerce(other)
        return DualArray(self.primal * o.primal, self.primal * o.tangent + self.tangent * o.primal)

    __rmul__ = __mul__

    def __truediv__(self, other: ArrayLike) -> "DualArray":
        o = self._coerce(other)
        return DualArray(self.primal/o.primal, (self.tangent * o.primal - self.primal * o.tangent)/o.primal**2)

    def __rtruediv__(self, other: ArrayLike) -> "DualArray":
        return self._coerce(other).__truediv__(self)

    def __pow__(self, p: float) -> "DualArray":
        return DualArray(self.primal**p, self.tangent*p*self.primal**(p-1))

    def __neg__(self) -> "DualArray":
        return DualArray(-self.primal, -self.tangent)

    def __matmul__(self, other: ArrayLike) -> "DualArray":
        o = self._coerce(other)
        return DualArray(self.primal @ o.primal, self.primal @ o.tangent + self.tangent @ o.primal)

    def __rmatmul__(self, other: ArrayLike) -> "DualArray":
        o = self._coerce(other)
        return DualArray(o.primal @ self.primal, o.primal @ self.tangent + o.tangent @ self.primal)

    # reductions (given)
    def sum(self, axis=None) -> "DualArray":
        return DualArray(self.primal.sum(axis=axis), self.tangent.sum(axis=axis))

    def mean(self, axis=None) -> "DualArray":
        return DualArray(self.primal.mean(axis=axis), self.tangent.mean(axis=axis))


# elementwise functions (EXERCISE)
# Each must work on a DualArray and a plain array.
# The numpy branch is given. fill in the DualArray one.
def exp(x: ArrayLike) -> ArrayLike:
    if isinstance(x, DualArray):
        val = np.exp(x.primal)
        return DualArray(val, x.tangent * val)    
    return np.exp(x)


def log(x: ArrayLike) -> ArrayLike:
    if isinstance(x, DualArray):
        return DualArray(np.log(x.primal), x.tangent * 1 / x.primal)
    return np.log(x)


def tanh(x: ArrayLike) -> ArrayLike:
    if isinstance(x, DualArray):
        val = np.tanh(x.primal)
        return DualArray(val, x.tangent * (1 - val**2))
    return np.tanh(x)


def relu(x: ArrayLike) -> ArrayLike:
    if isinstance(x, DualArray):
        return DualArray(np.maximum(0.0, x.primal), x.tangent * (x.primal > 0))
    return np.maximum(x, 0.0)


# jax-like interface
def jvp(
    f: Callable[..., DualArray],
    primals: Sequence[np.ndarray],
    tangents: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Jacobian-vector product (forward mode).

    Returns ``(f(primals), J(primals) @ tangents)`` in a pass.
    ``primals`` and ``tangents`` are matched sequences of arrays.
    """
    duals = [DualArray(p, t) for p, t in zip(primals, tangents)]
    out = f(*duals)
    if not isinstance(out, DualArray):
        out = DualArray(out)
    return out.primal, out.tangent

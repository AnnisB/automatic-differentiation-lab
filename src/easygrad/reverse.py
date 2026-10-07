"""Reverse mode over numpy arrays: VJPs and backprop."""

from typing import Callable, Sequence

import numpy as np

type ArrayLike = Node | np.ndarray | float | int


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` so it matches ``shape`` after numpy broadcasting.

    EXERCISE -- see docs/vector/reverse.md.
    """
    grad = np.asarray(grad)

    while len(shape) < grad.ndim:
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(shape)

class Node:
    """A node wrapping a numpy array in the reverse-mode tape."""

    # see https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_ufunc__
    __array_ufunc__ = None

    def __init__(
        self, value: ArrayLike, _children: tuple["Node", ...] = (), _op: str = ""
    ) -> None:
        self.value: np.ndarray = np.asarray(value, dtype=float)
        self.grad: np.ndarray = np.zeros_like(self.value)
        self._backward: Callable[[], None] = lambda: None
        self._prev: tuple["Node", ...] = tuple(_children)
        self._op: str = _op

    # -- helpers
    @staticmethod
    def _coerce(other: ArrayLike) -> "Node":
        return other if isinstance(other, Node) else Node(other)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Node(shape={self.value.shape}, op={self._op!r})"

    # binary ops (EXERCISE)
    # Compute `out` with the forward value + children, a `_backward` that
    # accumulates VJPs into each parent's .grad (use `unbroadcast`),
    # and return out.
    def __add__(self, other: ArrayLike) -> "Node":
        other = self._coerce(other)
        out = Node(self.value + other.value, (self, other), "+")

        def _backward():
            self.grad += unbroadcast(out.grad, self.shape)
            other.grad += unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    __radd__ = __add__

    def __mul__(self, other: ArrayLike) -> "Node":
        other = self._coerce(other)
        out = Node(self.value * other.value, (self, other), "*")
        
        def _backward():
            self.grad += unbroadcast(out.grad * other.value, self.shape)
            other.grad += unbroadcast(out.grad * self.value, other.shape)
        
        out._backward = _backward
        return out
        
    __rmul__ = __mul__

    def __truediv__(self, other: ArrayLike) -> "Node":
        other = self._coerce(other)
        out = Node(self.value / other.value, (self, other), "/")
        
        def _backward():
            self.grad += unbroadcast(out.grad / other.value, self.shape)
            other.grad += unbroadcast(-out.grad * self.value / (other.value ** 2), other.shape)
        
        out._backward = _backward
        return out
    
    def __pow__(self, p: float) -> "Node":
        out = Node(self.value ** p, (self,), f"**{p}")
        
        def _backward():
            self.grad += out.grad * p * self.value ** (p - 1)
        
        out._backward = _backward
        return out

    def __matmul__(self, other: ArrayLike) -> "Node":
        # use the given _matmul_lhs_grad / _matmul_rhs_grad helpers in _backward
        other = self._coerce(other)
        out = Node(self.value @ other.value, (self, other), "@")
        
        def _backward():
            self.grad += _matmul_lhs_grad(out.grad, self.value, other.value)
            other.grad += _matmul_rhs_grad(out.grad, self.value, other.value)
        
        out._backward = _backward
        return out

    # derived / reflected ops (given)
    def __sub__(self, other: ArrayLike) -> "Node":
        return self + (-self._coerce(other))

    def __rsub__(self, other: ArrayLike) -> "Node":
        return self._coerce(other) + (-self)

    def __neg__(self) -> "Node":
        return self * -1

    def __rmatmul__(self, other: ArrayLike) -> "Node":
        return self._coerce(other).__matmul__(self)

    # reductions
    def sum(self, axis: int | tuple[int, ...] | None = None) -> "Node":
        """Sum reduction. VJP: broadcast the cotangent back over the summed axes.

        You should use :func:`_restore_axes` helper

        EXERCISE -- see docs/vector/reverse.md.
        """
        out = Node(self.value.sum(axis=axis), (self,), "sum")
        def _backward():
            g = _restore_axes(out.grad, self.shape, axis)
            self.grad += np.broadcast_to(g, self.shape)
        
        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None) -> "Node":
        n = (
            self.value.size
            if axis is None
            else np.prod(np.take(self.value.shape, np.atleast_1d(axis)))
        )
        return self.sum(axis=axis) * (1.0 / float(n))

    # elementwise functions (EXERCISE)
    def exp(self) -> "Node":
        val = np.exp(self.value)
        out = Node(val, (self,), "exp")

        def _backward():
            self.grad += out.grad * val

        out._backward = _backward
        return out
        
        
    def log(self) -> "Node":
        out = Node(np.log(self.value), (self,), "log")

        def _backward():
            self.grad += out.grad / self.value

        out._backward = _backward
        return out
    
    def tanh(self) -> "Node":
        val = np.tanh(self.value)
        out = Node(val, (self,), "tanh")

        def _backward():
            self.grad += out.grad * (1 - val**2)

        out._backward = _backward
        return out

    def relu(self) -> "Node":
        out = Node(np.maximum(self.value, 0.0), (self,), "relu")

        def _backward():
            self.grad += out.grad * (self.value > 0)

        out._backward = _backward
        return out

    # reverse pass (EXERCISE)
    def backward(self) -> None:
        """Seed the (scalar) output with 1 and accumulate gradients.

        Use the given :func:`_topo` to order the graph. Zero every node's grad,
        seed ``self.grad = ones_like(self.value)``, then call each ``_backward``

        EXERCISE -- see docs/vector/reverse.md.
        """
        order = _topo(self)
        for node in order:
            node.grad = np.zeros_like(node.value)
        self.grad = np.ones_like(self.value)
        for node in reversed(order):
            node._backward()


# matrix helpers
def _matmul_lhs_grad(g: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    g = np.asarray(g)
    if a.ndim == 1 and b.ndim == 1:  # vector dot -> scalar
        return g * b
    if a.ndim == 1:  # (k,) @ (k,n) -> (n,)
        return g @ b.T
    if b.ndim == 1:  # (m,k) @ (k,) -> (m,)
        return np.outer(g, b)
    return g @ b.T  # (m,k) @ (k,n) -> (m,n)


def _matmul_rhs_grad(g: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    g = np.asarray(g)
    if a.ndim == 1 and b.ndim == 1:
        return g * a
    if a.ndim == 1:  # (k,) @ (k,n) -> (n,)
        return np.outer(a, g)
    if b.ndim == 1:  # (m,k) @ (k,) -> (m,)
        return a.T @ g
    return a.T @ g


def _restore_axes(
    grad: np.ndarray, shape: tuple[int, ...], axis: int | tuple[int, ...] | None
) -> np.ndarray:
    """Reinsert reduced axes (as size 1) so the cotangent can broadcast back."""
    if axis is None:
        return np.asarray(grad)
    grad = np.asarray(grad)
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    axes = tuple(a % len(shape) for a in axes)
    return np.expand_dims(grad, axes)


def _topo(root: Node) -> list[Node]:
    order: list[Node] = []
    visited: set[int] = set()

    def build(node: Node) -> None:
        if id(node) in visited:
            return
        visited.add(id(node))
        for child in node._prev:
            build(child)
        order.append(node)

    build(root)
    return order


# elementwise free functions
def exp(x: Node) -> Node:
    return x.exp()


def log(x: Node) -> Node:
    return x.log()


def tanh(x: Node) -> Node:
    return x.tanh()


def relu(x: Node) -> Node:
    return x.relu()

# jax-like interface
def grad(
    f: Callable[..., Node], argnums: int | Sequence[int] = 0
):
    """Return a function computing the gradient of scalar-output ``f``.

    ``f`` takes one or more numpy arrays and returns a scalar :class:`Node`.
    ``grad(f)(x)`` returns ``df/dx`` with the same shape as ``x``. Pass
    ``argnums`` to differentiate with respect to other / several positional
    arguments (mirrors ``jax.grad``).
    """
    single = isinstance(argnums, int)
    nums: tuple[int, ...] = (argnums,) if single else tuple(argnums)

    def grad_fn(*args: np.ndarray):
        nodes = [Node(a) if i in nums else a for i, a in enumerate(args)]
        node_args = [n if isinstance(n, Node) else Node(n) for n in nodes]
        out = f(*node_args)
        if out.value.size != 1:
            raise ValueError("grad requires a scalar-output function")
        out.backward()
        grads = [node_args[i].grad for i in nums]
        return grads[0] if single else tuple(grads)

    return grad_fn


def value_and_grad(f: Callable[..., Node], argnums: int | Sequence[int] = 0):
    """Like :func:`grad` but also returns the scalar value of ``f``."""
    single = isinstance(argnums, int)
    nums: tuple[int, ...] = (argnums,) if single else tuple(argnums)

    def vg(*args: np.ndarray):
        node_args = [Node(a) for a in args]
        out = f(*node_args)
        if out.value.size != 1:
            raise ValueError("value_and_grad requires a scalar-output function")
        out.backward()
        grads = [node_args[i].grad for i in nums]
        return float(out.value), (grads[0] if single else tuple(grads))

    return vg

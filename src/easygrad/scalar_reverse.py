"""Scalar reverse mode."""

from typing import Callable, Iterable
import math

class Value:
    """A node in a scalar computational graph."""

    def __init__(self, data: float, _children: tuple["Value", ...] = (), _op: str = "") -> None:
        self.data: float = float(data)
        self.grad: float = 0.0
        # internal autograd
        self._backward: Callable[[], None] = lambda: None
        self._prev: tuple["Value", ...] = tuple(_children)
        self._op: str = _op

    # helpers
    @staticmethod
    def _coerce(other: "Value | float") -> "Value":
        return other if isinstance(other, Value) else Value(other)

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    # primitive ops (EXERCISE)
    # Each should:
    # - compute `out` with the forward value and (self[, other]) as children,
    # - define a `_backward` that accumulates into the parents .grad
    # - assign it to out._backward, and return out.
    def __add__(self, other: "Value | float") -> "Value":
        other = self._coerce(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out 

    __radd__ = __add__

    def __mul__(self, other: "Value | float") -> "Value":
        other = self._coerce(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out 

    __rmul__ = __mul__

    def __pow__(self, p: float) -> "Value":
        assert isinstance(p, (int, float)), "only constant real powers supported"
        out = Value(self.data ** p, (self,), f"**{p}")
        def _backward():
            self.grad += out.grad * p * self.data ** (p-1)
        out._backward = _backward
        return out 

    # derived ops
    def __neg__(self) -> "Value":
        return self * -1

    def __sub__(self, other: "Value | float") -> "Value":
        return self + (-self._coerce(other))

    def __rsub__(self, other: "Value | float") -> "Value":
        return self._coerce(other) + (-self)

    def __truediv__(self, other: "Value | float") -> "Value":
        return self * self._coerce(other) ** -1

    def __rtruediv__(self, other: "Value | float") -> "Value":
        return self._coerce(other) * self ** -1

    # elementary functions (EXERCISE)
    def exp(self) -> "Value":
        val = math.exp(self.data)
        out = Value(val, (self,), "exp")
        def _backward():
            self.grad += out.grad * val
        out._backward = _backward
        return out     

    def log(self) -> "Value":
        out = Value(math.log(self.data), (self,), "log")
        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out   

    def tanh(self) -> "Value":
        val = math.tanh(self.data)
        out = Value(val, (self,), "tanh")
        def _backward():
            self.grad += out.grad * (1 - val**2)
        out._backward = _backward
        return out   
    
    def relu(self) -> "Value":
        out = Value(max(0.0, self.data), (self,), "relu")
        def _backward():
            self.grad += out.grad * (self.data > 0)
        out._backward = _backward
        return out   
    
    # reverse pass (EXERCISE)
    def backward(self) -> None:
        """Run reverse-mode autodiff seeded at ``self`` (assumed scalar output). """
        order = topo_sort(self)
        for node in order:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):
            node._backward()
        


def topo_sort(root: Value) -> list[Value]:
    """Return a topological ordering of the graph ending at ``root``.

    EXERCISE -- see docs/scalar/reverse-tape.md.
    """
    order = []
    visited = set()
    def visit(node):
        if node not in visited:
            visited.add(node)
            for parent in node._prev:
                visit(parent)
            order.append(node)
    visit(root)
    return order


def is_valid_topo_order(order: Iterable[Value]) -> bool:
    """Check that no node precedes one of its parents in ``order`` (given)."""
    seen: set[int] = set()
    for node in order:
        for parent in node._prev:
            if id(parent) not in seen:
                return False
        seen.add(id(node))
    return True

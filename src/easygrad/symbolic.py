"""Symbolic differentiation"""

import sympy as sp


def logistic_map(x: sp.Expr, r: sp.Expr) -> sp.Expr:
    """One step of the logistic map, ``r * x * (1 - x)``."""
    return r * x * (1 - x)


def expression_size(expr: sp.Expr) -> int:
    """Number of nodes in the expression tree."""
    return sum(1 for _ in sp.preorder_traversal(expr))


def swell_demo(max_iter: int = 5) -> list[tuple[int, int]]:
    """Iterate the logistic map and measure the swell of its derivative.

    For each iteration count ``n`` in ``0 .. max_iter``, form the
    composition and  record the :func:`expression_size` of the expanded
    (``sympy.expand``) derivative.

    Returns a list of ``(n, node_count)`` pairs.
    """
    x, r = sp.symbols("x r")
    xn = x
    results = []
    for n in range(max_iter + 1):
        derivative = sp.diff(xn, x)
        expanded = sp.expand(derivative)
        size = expression_size(expanded)
        results.append((n, size))
        xn = logistic_map(xn, r)

    return results
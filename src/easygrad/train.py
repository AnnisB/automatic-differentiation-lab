"""Training loop helpers built on the reverse-mode engine."""

from typing import Callable

import numpy as np

from .reverse import Node, value_and_grad


def gradient_descent(
    loss_fn: Callable[..., Node],
    params: list[np.ndarray],
    lr: float,
    steps: int,
) -> tuple[list[np.ndarray], list[float]]:
    """Run ``steps`` of plain full-batch gradient descent on ``loss_fn``.

    ``loss_fn(*params)`` must return a scalar :class:`Node`. Each step computes
    the loss and the gradient w.r.t. every parameter in one backward pass
    (use :func:`value_and_grad` with ``argnums = (0, 1, ..., len(params)-1)``),
    then replaces each ``p`` with ``p - lr * g``.

    Returns ``(final_params, loss_history)`` where ``loss_history[i]`` is the
    scalar loss *before* the i-th update.

    EXERCISE -- see docs/train-mlp.md.
    """
    argnums = tuple(range(len(params)))
    vg = value_and_grad(loss_fn, argnums=argnums)

    history = []
    for _ in range(steps):
        loss_value, grads = vg(*params)
        history.append(loss_value)
        params = [p - lr * g for p, g in zip(params, grads)]
    return params, history
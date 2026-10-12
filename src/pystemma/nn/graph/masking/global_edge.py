"""Static global edge-mask generation."""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor, nn


class _StrictBinaryThresholdSTE(torch.autograd.Function):
    """Apply a strict binary threshold with an identity surrogate backward."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, scores: Tensor) -> Tensor:
        del ctx
        return (scores > 0.0).to(dtype=scores.dtype)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor) -> tuple[Tensor]:
        del ctx
        return (grad_output,)


class GlobalEdgeMask(nn.Module):
    """Learn a static directed binary edge mask for a fixed node count.

    The module owns only the unconstrained edge scores and generates a binary edge
    mask with a hard threshold. Applying the binary edge mask to an adjacency or
    propagation operator, computing task losses, and applying regularization are
    responsibilities of the caller.
    """

    def __init__(self, num_nodes: int, *, initial_value: float = 0.1) -> None:
        super().__init__()

        if type(num_nodes) is not int:
            raise TypeError("num_nodes must be a built-in int.")
        if num_nodes <= 0:
            raise ValueError("num_nodes must be greater than zero.")

        if isinstance(initial_value, bool) or not isinstance(initial_value, Real):
            raise TypeError("initial_value must be a real scalar.")
        try:
            initial_float = float(initial_value)
        except (OverflowError, ValueError) as error:
            raise ValueError("initial_value must be finite.") from error
        if not math.isfinite(initial_float):
            raise ValueError("initial_value must be finite.")

        self.num_nodes = num_nodes
        self.scores = nn.Parameter(
            torch.full(
                (num_nodes, num_nodes),
                initial_float,
                dtype=torch.float32,
            )
        )

    def forward(self) -> Tensor:
        """Return the exact binary edge mask produced by the strict zero threshold."""

        return _StrictBinaryThresholdSTE.apply(self.scores)

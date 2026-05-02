"""
Reward heads of the form r(h) = phi(w^T h + b).

The preactivation z = w^T h + b is a scalar. phi is a learned activation
applied to that scalar. Concavity in h is guaranteed when phi is concave
in z, because z is affine in h.

Activations included:
    Ident          : phi(z) = z. Linear baseline (weakly concave).
    BoundedAbove   : phi(z) = alpha * log_sigmoid(z). Strictly concave,
                     bounded above by 0, monotonic. The recommended
                     concave variant.
    GeluBoundedAbove: phi(z) = -alpha * GELU(z). Concave, bounded above
                      by 0. Non-monotonic for z > 0.
    Bounded        : phi(z) = alpha * tanh(z). Monotonic, bounded in
                     (-alpha, +alpha), but only one-sided concavity.
                     Useful as a comparison: tests whether saturation
                     in the high-reward regime alone suffices.

All learned positive parameters are softplus(theta) for raw learnable theta.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Activations (scalar input, scalar output) ----------

class Ident(nn.Module):
    """phi(z) = z. Linear baseline."""
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class BoundedAbove(nn.Module):
    """phi(z) = alpha * log_sigmoid(z), alpha = softplus(a) > 0.
    
    Strictly concave, monotonic increasing, bounded above by 0.
    Linear-asymptotic as z -> -inf: phi(z) ~ alpha * z.
    Recommended primary concave head.
    """
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.rand(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.a) * F.logsigmoid(x)
    
    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.a)


class GeluBoundedAbove(nn.Module):
    """phi(z) = -alpha * GELU(z), alpha = softplus(a) > 0.
    
    Concave (since GELU is approximately convex). Bounded above by 0
    at z = 0. NOT monotonic: phi decreases as z grows past 0.
    """
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.rand(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -F.softplus(self.a) * F.gelu(x)


class Bounded(nn.Module):
    """phi(z) = alpha * tanh(z), alpha = softplus(a) > 0.
    
    Monotonic increasing, bounded in (-alpha, +alpha).
    NOT globally concave: concave on z >= 0, convex on z < 0.
    Tests whether one-sided saturation suffices for runaway suppression.
    """
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.rand(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.a) * torch.tanh(x)


# ---------- Registry and metadata ----------

ACTIVATION_REGISTRY: dict[str, type] = {
    "ident": Ident,
    "bounded_above": BoundedAbove,
    "gelu_bounded_above": GeluBoundedAbove,
    "bounded": Bounded,
}

# Globally concave in z (and therefore in h, since z is affine in h)
GLOBALLY_CONCAVE: dict[str, bool] = {
    "ident": True,                 # weakly (affine)
    "bounded_above": True,
    "gelu_bounded_above": True,
    "bounded": False,              # one-sided
}

# Strictly concave (excludes ident)
STRICTLY_CONCAVE: dict[str, bool] = {
    "ident": False,
    "bounded_above": True,
    "gelu_bounded_above": True,
    "bounded": False,
}

# Monotonically increasing (preserves BT pairwise rankings)
MONOTONIC: dict[str, bool] = {
    "ident": True,
    "bounded_above": True,
    "gelu_bounded_above": False,    # decreasing for z > 0
    "bounded": True,
}


# ---------- The head module ----------

class RewardHead(nn.Module):
    """Reward head: r(h) = phi(w^T h + b).
    
    Args:
        hidden_size: dimension of input hidden state h.
        activation_name: key into ACTIVATION_REGISTRY.
        init_scale: stddev of Gaussian init for w.
        init_bias: initial value of b.
    """
    
    def __init__(
        self,
        hidden_size: int,
        activation_name: str = "bounded_above",
        init_scale: float = 0.02,
        init_bias: float = 0.0,
    ):
        super().__init__()
        if activation_name not in ACTIVATION_REGISTRY:
            raise ValueError(
                f"Unknown activation: {activation_name!r}. "
                f"Available: {list(ACTIVATION_REGISTRY)}"
            )
        self.activation_name = activation_name
        self.hidden_size = hidden_size
        
        self.linear = nn.Linear(hidden_size, 1, bias=True)
        nn.init.normal_(self.linear.weight, mean=0.0, std=init_scale)
        nn.init.constant_(self.linear.bias, init_bias)
        
        self.activation = ACTIVATION_REGISTRY[activation_name]()
    
    def preactivation(self, h: torch.Tensor) -> torch.Tensor:
        """z = w^T h + b. Shape: (batch, 1)."""
        return self.linear(h)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """r = phi(w^T h + b). Shape: (batch, 1)."""
        return self.activation(self.preactivation(h))
    
    @property
    def w(self) -> torch.Tensor:
        return self.linear.weight.squeeze(0)
    
    @property
    def b(self) -> torch.Tensor:
        return self.linear.bias.squeeze()
    
    def num_extra_params(self) -> int:
        """Parameter count of the activation beyond w, b."""
        return sum(p.numel() for p in self.activation.parameters())
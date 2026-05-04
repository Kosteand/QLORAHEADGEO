"""
Reward heads matching the collaborator's RL implementation.

Key differences from previous version:
- Activations have PER-DIMENSION parameters (vector instead of scalar).
  Each output channel has its own peak shape.
  This matches worldModels3.py exactly.

Architecture:
    r(h) = mean_k(phi_k(fc2(LeakyReLU(fc1(h))))) + b_out

where each phi_k can have a different peak/curvature for each k.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Ident(nn.Module):
    """phi(z) = z."""
    def __init__(self, dimensionality: int = None):
        super().__init__()
    def forward(self, x):
        return x


class BoundedAbove(nn.Module):
    """phi(z) = alpha * log_sigmoid(z), alpha PER-DIMENSION."""
    def __init__(self, dimensionality: int):
        super().__init__()
        self.a = nn.Parameter(torch.rand(dimensionality))
    def forward(self, x):
        return F.softplus(self.a) * F.logsigmoid(x)


class GaussianActivation(nn.Module):
    """phi(z) = b * exp(-a*z^2). PER-DIMENSION a, b.
    
    Matches worldModels3.py's GaussianActivation exactly.
    Each channel has its own width (a) and amplitude (b).
    """
    def __init__(self, dimensionality: int):
        super().__init__()
        self.a = nn.Parameter(torch.rand(dimensionality))
        self.b = nn.Parameter(torch.rand(dimensionality))
    def forward(self, x):
        return F.softplus(self.b) * torch.exp(-F.softplus(self.a) * torch.square(x))
    def regularization_loss(self):
        return F.softplus(self.b).square().mean()


class QuadraticActivation(nn.Module):
    """phi(z) = -alpha * z^2. PER-DIMENSION alpha."""
    def __init__(self, dimensionality: int):
        super().__init__()
        self.a = nn.Parameter(torch.rand(dimensionality))
    def forward(self, x):
        return -F.softplus(self.a) * torch.square(x)


class BoundedActivation(nn.Module):
    """phi(z) = alpha * tanh(z). PER-DIMENSION alpha."""
    def __init__(self, dimensionality: int):
        super().__init__()
        self.a = nn.Parameter(torch.rand(dimensionality))
    def forward(self, x):
        return F.softplus(self.a) * torch.tanh(x)


class PLinearBoundedAbove(nn.Module):
    """phi(z) = -alpha * |z|. PER-DIMENSION alpha."""
    def __init__(self, dimensionality: int):
        super().__init__()
        self.a = nn.Parameter(torch.rand(dimensionality))
    def forward(self, x):
        return -F.softplus(self.a) * torch.abs(x)


class GeluBoundedAbove(nn.Module):
    """phi(z) = -alpha * GELU(z). PER-DIMENSION alpha."""
    def __init__(self, dimensionality: int):
        super().__init__()
        self.a = nn.Parameter(torch.rand(dimensionality))
    def forward(self, x):
        return -F.softplus(self.a) * F.gelu(x)


# Aliases for backward compat
Bounded = BoundedActivation


# ---------- Registry and metadata ----------

ACTIVATION_REGISTRY: dict[str, type] = {
    "ident": Ident,
    "bounded_above": BoundedAbove,
    "gaussian": GaussianActivation,
    "quadratic": QuadraticActivation,
    "p_linear_bounded_above": PLinearBoundedAbove,
    "gelu_bounded_above": GeluBoundedAbove,
    "bounded": BoundedActivation,
}

GLOBALLY_CONCAVE: dict[str, bool] = {
    "ident": True,
    "bounded_above": True,
    "gaussian": True,
    "quadratic": True,
    "p_linear_bounded_above": True,
    "gelu_bounded_above": False,
    "bounded": False,
}

STRICTLY_CONCAVE: dict[str, bool] = {
    "ident": False,
    "bounded_above": True,
    "gaussian": True,
    "quadratic": True,
    "p_linear_bounded_above": False,
    "gelu_bounded_above": False,
    "bounded": False,
}

MONOTONIC: dict[str, bool] = {
    "ident": True,
    "bounded_above": True,
    "gaussian": False,
    "quadratic": False,
    "p_linear_bounded_above": False,
    "gelu_bounded_above": False,
    "bounded": True,
}

PEAKED: dict[str, bool] = {
    "ident": False,
    "bounded_above": False,
    "gaussian": True,
    "quadratic": True,
    "p_linear_bounded_above": True,
    "gelu_bounded_above": True,
    "bounded": False,
}


# ---------- The head module ----------

class RewardHead(nn.Module):
    """Reward head: Linear → LeakyReLU → Linear → Activation → mean → +bias.
    
    Now passes head_width as `dimensionality` to the activation, matching
    the RL setup where each output channel has its own activation parameters.
    """

    def __init__(
        self,
        hidden_size: int,
        activation_name: str = "bounded_above",
        intermediate_size: int | None = None,
        head_width: int = 32,
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
        self.intermediate_size = hidden_size if intermediate_size is None else intermediate_size
        self.head_width = head_width

        self.fc1 = nn.Linear(hidden_size, self.intermediate_size)
        self.fc2 = nn.Linear(self.intermediate_size, head_width)
        # Activation gets head_width as dimensionality so each output
        # channel has its own activation parameters.
        self.activation = ACTIVATION_REGISTRY[activation_name](dimensionality=head_width)
        self.output_bias = nn.Parameter(torch.tensor(float(init_bias)))

        nn.init.normal_(self.fc1.weight, std=init_scale)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=init_scale)
        nn.init.zeros_(self.fc2.bias)

    def preactivation(self, h):
        return self.fc2(F.leaky_relu(self.fc1(h)))

    def forward(self, h):
        z = self.preactivation(h)
        r = self.activation(z)
        return r.mean(dim=-1, keepdim=True) + self.output_bias

    def num_extra_params(self):
        return sum(p.numel() for p in self.activation.parameters())
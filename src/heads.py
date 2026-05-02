"""
Reward heads of the form:
    r(h) = mean_k(phi(fc2(leaky_relu(fc1(h))))) + b_out

fc1: Linear(hidden_size → intermediate_size)
LeakyReLU (non-linearity between layers)
fc2: Linear(intermediate_size → head_width)
phi: element-wise concave activation applied to each of the head_width units
mean: average over the head_width dimension → scalar per example
b_out: learned scalar bias

Activations included:
    Ident          : phi(z) = z. Linear baseline (weakly concave).
    BoundedAbove   : phi(z) = alpha * log_sigmoid(z). Strictly concave,
                     bounded above by 0, monotonic. The recommended
                     concave variant.
    GeluBoundedAbove: phi(z) = -alpha * GELU(z). Concave, bounded above
                      by 0. Non-monotonic for z > 0.
    Bounded        : phi(z) = alpha * tanh(z). Monotonic, bounded in
                     (-alpha, +alpha), but only one-sided concavity.

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
        self.a = nn.Parameter(torch.tensor([-3.0]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.a) * F.logsigmoid(x)
    
    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.a)


class GeluBoundedAbove(nn.Module):
    """phi(z) = -alpha * GELU(z), alpha = softplus(a) > 0.
    
    NOT globally concave: GELU has a small non-convex region near
    z ~ -0.75, so -GELU has a corresponding non-concave region. Concave
    on most of R but not all.
    
    Bounded above by 0 (achieved at z = 0). NOT monotonic: phi decreases
    as z grows past 0.
    
    Useful as a comparison head; see Bounded for one-sided concavity case.
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
    "gelu_bounded_above": False,   # GELU has a small non-convex region near z=-0.75
    "bounded": False,              # one-sided
}

# Strictly concave (excludes ident)
STRICTLY_CONCAVE: dict[str, bool] = {
    "ident": False,
    "bounded_above": True,
    "gelu_bounded_above": False,
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
    """Reward head: Linear → LeakyReLU → Linear → ConcaveActivation → mean → +bias.

    r(h) = mean_k(phi(fc2(leaky_relu(fc1(h))))) + b_out

    Args:
        hidden_size: dimension of input hidden state h.
        activation_name: key into ACTIVATION_REGISTRY.
        intermediate_size: width of the hidden layer; defaults to hidden_size.
        head_width: number of parallel output units before mean pooling (k).
        init_scale: stddev of Gaussian init for fc1/fc2 weights.
        init_bias: initial value of output bias b_out.
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
        self.activation = ACTIVATION_REGISTRY[activation_name]()
        self.output_bias = nn.Parameter(torch.tensor(float(init_bias)))

        nn.init.normal_(self.fc1.weight, std=init_scale)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=init_scale)
        nn.init.zeros_(self.fc2.bias)

    def preactivation(self, h: torch.Tensor) -> torch.Tensor:
        """z = fc2(leaky_relu(fc1(h))). Shape: (batch, head_width)."""
        return self.fc2(F.leaky_relu(self.fc1(h)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """r = mean_k(phi(z)) + b_out. Shape: (batch, 1)."""
        z = self.preactivation(h)                      # (batch, head_width)
        r = self.activation(z)                         # (batch, head_width)
        return r.mean(dim=-1, keepdim=True) + self.output_bias  # (batch, 1)

    def num_extra_params(self) -> int:
        """Parameter count of the activation beyond fc1/fc2/output_bias."""
        return sum(p.numel() for p in self.activation.parameters())
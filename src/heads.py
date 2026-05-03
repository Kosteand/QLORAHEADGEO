"""
Reward heads of the form:
    r(h) = mean_k(phi(fc2(leaky_relu(fc1(h))))) + b_out

Activations included:
    Ident          : phi(z) = z. Linear baseline.
    BoundedAbove   : phi(z) = alpha * log_sigmoid(z). Strictly concave,
                     bounded above by 0, monotonic. Has a "ceiling" at infinity.
    Gaussian       : phi(z) = b * exp(-a*z^2). Strictly concave, peaked at z=0,
                     decays in BOTH directions. Has no monotonic direction --
                     no direction in which the policy can extract more reward.
    GeluBoundedAbove: phi(z) = -alpha * GELU(z). Concave, bounded above
                      by 0. Non-monotonic for z > 0.
    Bounded        : phi(z) = alpha * tanh(z). Monotonic, bounded in
                     (-alpha, +alpha), but only one-sided concavity.

Important geometric distinction:
    - bounded_above is monotonic: there's a "high z = high reward" direction
      with a ceiling. Policy can push z high; ceiling caps reward magnitude
      but doesn't prevent gaming (any large-z response hits the ceiling).
    - gaussian is peaked: maximum at z=0, decays in both directions.
      Policy that pushes z away from 0 LOSES reward. There is no direction
      to runaway. This is the stronger structural fix.

All learned positive parameters are softplus(theta) for raw learnable theta.

Activations may optionally implement a `regularization_loss()` method that
returns a scalar tensor. The trainer collects and weights these.
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
    """
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([-3.0]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.a) * F.logsigmoid(x)
    
    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.a)


class GaussianActivation(nn.Module):
    """phi(z) = b * exp(-a*z^2), a, b = softplus(.) > 0.
    
    Strictly concave, single peak at z=0, decays exponentially in BOTH
    directions. Unlike bounded_above, there is no monotonic direction --
    every direction away from z=0 loses reward. This is the stronger
    structural fix for runaway: the policy cannot extract more reward
    by pushing z further in any direction.
    
    Provides regularization_loss() = mean(softplus(b)^2), which caps the
    Gaussian peak amplitude. Without it, the optimizer can scale up b
    to make differences larger, partially defeating the bound.
    
    NOT monotonic in z. The "high z = high reward" interpretation is gone:
    "small |z| = high reward" replaces it. Diagnostics should plot |z|.
    """
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.rand(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.b) * torch.exp(
            -F.softplus(self.a) * x.square()
        )
    
    def regularization_loss(self) -> torch.Tensor:
        """L2 on the peak amplitude b. Following the PhD's RL setup."""
        return F.softplus(self.b).square().mean()


class GeluBoundedAbove(nn.Module):
    """phi(z) = -alpha * GELU(z). Bounded above, NOT globally concave."""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.rand(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -F.softplus(self.a) * F.gelu(x)


class Bounded(nn.Module):
    """phi(z) = alpha * tanh(z). Monotonic, bounded, one-sided concave."""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.rand(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.a) * torch.tanh(x)


# ---------- Registry and metadata ----------

ACTIVATION_REGISTRY: dict[str, type] = {
    "ident": Ident,
    "bounded_above": BoundedAbove,
    "gaussian": GaussianActivation,
    "gelu_bounded_above": GeluBoundedAbove,
    "bounded": Bounded,
}

# Globally concave in z
GLOBALLY_CONCAVE: dict[str, bool] = {
    "ident": True,                 # weakly (affine)
    "bounded_above": True,
    "gaussian": True,              # strictly concave (negative def. Hessian)
    "gelu_bounded_above": False,
    "bounded": False,
}

STRICTLY_CONCAVE: dict[str, bool] = {
    "ident": False,
    "bounded_above": True,
    "gaussian": True,
    "gelu_bounded_above": False,
    "bounded": False,
}

# Monotonically increasing
MONOTONIC: dict[str, bool] = {
    "ident": True,
    "bounded_above": True,
    "gaussian": False,             # peaked at 0; decreasing for z > 0
    "gelu_bounded_above": False,
    "bounded": True,
}

# Has interior peak (vs monotonic-with-asymptote)
PEAKED: dict[str, bool] = {
    "ident": False,
    "bounded_above": False,
    "gaussian": True,
    "gelu_bounded_above": True,    # peaks at z=0, decreases for z > 0
    "bounded": False,
}


# ---------- The head module ----------

class RewardHead(nn.Module):
    """Reward head: Linear → LeakyReLU → Linear → ConcaveActivation → mean → +bias."""

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
        return self.fc2(F.leaky_relu(self.fc1(h)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.preactivation(h)
        r = self.activation(z)
        return r.mean(dim=-1, keepdim=True) + self.output_bias

    def num_extra_params(self) -> int:
        return sum(p.numel() for p in self.activation.parameters())
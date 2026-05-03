"""
Reward heads of the form:
    r(h) = mean_k(phi(fc2(leaky_relu(fc1(h))))) + b_out

Activations:
    Ident:         phi(z) = z. Linear baseline.
    BoundedAbove:  phi(z) = alpha * log_sigmoid(z). Strictly concave, monotonic, ceiling at 0.
    Bounded:       phi(z) = alpha * tanh(z). Bounded, one-sided concave.
    GeluBoundedAbove: phi(z) = -alpha * GELU(z). Approx peaked but with non-monotonic region.
    
    GaussianActivation: phi(z) = b * exp(-a*z^2). Peaked at z=0, exponential decay both sides.
    QuadraticActivation: phi(z) = -alpha * z^2. Peaked at z=0, quadratic decay (sharper).
    PLinearBoundedAbove: phi(z) = -alpha * |z|. Peaked at z=0, linear decay (sharpest).

Geometric hierarchy:
    1. Linear: unbounded ascent in direction w.
    2. Monotonic-bounded (bounded_above, bounded): ascent direction with ceiling.
    3. Peaked (gaussian, quadratic, p_linear): no ascent direction, decreases away from peak.
    
    Within peaked: gaussian is smoothest, quadratic is sharper, p_linear is sharpest.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Ident(nn.Module):
    """phi(z) = z. Linear baseline."""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x


class BoundedAbove(nn.Module):
    """phi(z) = alpha * log_sigmoid(z). Concave, monotonic, ceiling at 0."""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([-3.0]))
    def forward(self, x):
        return F.softplus(self.a) * F.logsigmoid(x)
    @property
    def alpha(self):
        return F.softplus(self.a)


class GaussianActivation(nn.Module):
    """phi(z) = b * exp(-a*z^2). Strictly concave, peaked at z=0, exponential decay.
    
    The smoothest of the peaked activations. Far from the peak, gradient
    decays exponentially -- the optimizer sees little signal to push out
    of the peak region.
    """
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.rand(1))
    def forward(self, x):
        return F.softplus(self.b) * torch.exp(-F.softplus(self.a) * x.square())
    def regularization_loss(self):
        return F.softplus(self.b).square().mean()


class QuadraticActivation(nn.Module):
    """phi(z) = -alpha * z^2. Peaked at z=0, quadratic decay.
    
    Globally concave (Hessian = -2*alpha < 0). Sharper than Gaussian:
    gradient grows linearly with |z|, so the optimizer sees stronger
    signal to keep z small. Unbounded below: phi -> -inf as |z| -> inf.
    
    No saturation = no "all bad responses look equally bad" failure mode.
    """
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.rand(1))
    def forward(self, x):
        return -F.softplus(self.a) * x.square()


class PLinearBoundedAbove(nn.Module):
    """phi(z) = -alpha * |z|. Peaked at z=0, linear decay.
    
    Concave (subdifferentiable at z=0). The sharpest peaked activation:
    gradient is constant magnitude -alpha for z>0 and +alpha for z<0,
    discontinuous at z=0. Spurious responses far from peak get strongly
    penalized regardless of how far.
    """
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.rand(1))
    def forward(self, x):
        return -F.softplus(self.a) * x.abs()


class GeluBoundedAbove(nn.Module):
    """phi(z) = -alpha * GELU(z). Approximately peaked, non-monotonic."""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.rand(1))
    def forward(self, x):
        return -F.softplus(self.a) * F.gelu(x)


class Bounded(nn.Module):
    """phi(z) = alpha * tanh(z). Monotonic, bounded."""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.rand(1))
    def forward(self, x):
        return F.softplus(self.a) * torch.tanh(x)


# ---------- Registry and metadata ----------

ACTIVATION_REGISTRY: dict[str, type] = {
    "ident": Ident,
    "bounded_above": BoundedAbove,
    "gaussian": GaussianActivation,
    "quadratic": QuadraticActivation,
    "p_linear_bounded_above": PLinearBoundedAbove,
    "gelu_bounded_above": GeluBoundedAbove,
    "bounded": Bounded,
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
    "p_linear_bounded_above": False,  # affine on each side, not strictly concave
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
    """Reward head: Linear → LeakyReLU → Linear → Activation → mean → +bias."""

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

    def preactivation(self, h):
        return self.fc2(F.leaky_relu(self.fc1(h)))

    def forward(self, h):
        z = self.preactivation(h)
        r = self.activation(z)
        return r.mean(dim=-1, keepdim=True) + self.output_bias

    def num_extra_params(self):
        return sum(p.numel() for p in self.activation.parameters())
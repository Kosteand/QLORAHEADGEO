"""
Verify that each activation is concave when used in r(h) = phi(w^T h + b).

A function f is concave iff for all h1, h2 and t in [0,1]:
    f(t*h1 + (1-t)*h2) >= t*f(h1) + (1-t)*f(h2)

We test this empirically by sampling random pairs of hidden states and
checking the inequality. The test is performed on the FULL head r(h),
not just the scalar activation phi(z), because that's the property the
paper depends on.

Run from project root:
    python -m src.concavity_test
"""

from __future__ import annotations

import torch

from .heads import ACTIVATION_REGISTRY, GLOBALLY_CONCAVE, MONOTONIC, RewardHead


def test_head_concavity_in_h(
    activation_name: str,
    hidden_size: int = 64,
    n_samples: int = 1000,
    h_scale: float = 2.0,
    tol: float = 1e-4,
    seed: int = 0,
) -> dict:
    """Test concavity of r(h) = phi(w^T h + b) in h."""
    torch.manual_seed(seed)
    head = RewardHead(hidden_size=hidden_size, activation_name=activation_name)
    head.eval()
    
    h1 = torch.randn(n_samples, hidden_size) * h_scale
    h2 = torch.randn(n_samples, hidden_size) * h_scale
    t = torch.empty(n_samples, 1).uniform_(0.0, 1.0)
    
    with torch.no_grad():
        h_mid = t * h1 + (1 - t) * h2
        r_mid = head(h_mid)
        r_chord = t * head(h1) + (1 - t) * head(h2)
    
    violation = (r_chord - r_mid).clamp(min=0.0)
    n_violations = (violation > tol).sum().item()
    max_violation = violation.max().item()
    
    return {
        "activation_name": activation_name,
        "violations": n_violations,
        "n_samples": n_samples,
        "max_violation": max_violation,
        "is_concave": n_violations == 0,
    }


def test_head_monotonicity(
    activation_name: str,
    n_samples: int = 1000,
    z_range: tuple[float, float] = (-10.0, 10.0),
    tol: float = 1e-5,
    seed: int = 0,
) -> dict:
    """Test monotonicity of phi as a scalar function (since the head
    is monotonic in h iff phi is monotonic in z and w is fixed)."""
    torch.manual_seed(seed)
    activation = ACTIVATION_REGISTRY[activation_name]()
    activation.eval()
    
    z1 = torch.empty(n_samples).uniform_(*z_range)
    z2 = torch.empty(n_samples).uniform_(*z_range)
    
    with torch.no_grad():
        phi_z1 = activation(z1.unsqueeze(-1)).squeeze(-1)
        phi_z2 = activation(z2.unsqueeze(-1)).squeeze(-1)
    
    higher = z1 > z2
    phi_higher = phi_z1 >= phi_z2 - tol
    near_equal = (z1 - z2).abs() < tol
    consistent = (higher == phi_higher) | near_equal
    
    n_violations = (~consistent).sum().item()
    return {
        "activation_name": activation_name,
        "violations": n_violations,
        "is_monotonic": n_violations == 0,
    }


def run_all_tests():
    print("=" * 70)
    print("HEAD CONCAVITY TESTS: r(h) = phi(w^T h + b) concave in h?")
    print("=" * 70)
    print(f"{'activation':<22} {'concave?':<10} {'expected':<10} {'max_viol':<12}")
    print("-" * 70)
    
    all_pass = True
    for name in ACTIVATION_REGISTRY:
        result = test_head_concavity_in_h(name)
        expected = GLOBALLY_CONCAVE[name]
        passed = result["is_concave"] == expected
        marker = "OK" if passed else "FAIL"
        all_pass = all_pass and passed
        print(
            f"{name:<22} {str(result['is_concave']):<10} "
            f"{str(expected):<10} {result['max_violation']:<12.2e} [{marker}]"
        )
    
    print()
    print("=" * 70)
    print("MONOTONICITY TESTS")
    print("=" * 70)
    print(f"{'activation':<22} {'monotonic?':<12} {'expected':<10}")
    print("-" * 70)
    
    for name in ACTIVATION_REGISTRY:
        result = test_head_monotonicity(name)
        expected = MONOTONIC[name]
        passed = result["is_monotonic"] == expected
        marker = "OK" if passed else "WARN"  # not always strict
        print(
            f"{name:<22} {str(result['is_monotonic']):<12} "
            f"{str(expected):<10} [{marker}]"
        )
    
    print()
    if all_pass:
        print("[ALL CONCAVITY TESTS PASSED]")
        return 0
    else:
        print("[SOME TESTS FAILED] - check the FAIL rows above")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(run_all_tests())
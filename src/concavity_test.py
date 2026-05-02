"""
Verify properties of the activations and the reward head.

Architecture:
    r(h) = mean_k(phi(fc2(LeakyReLU(fc1(h))))) + b_out

Important: the LeakyReLU between fc1 and fc2 means the FULL head r(h) is
NOT globally concave in h, even when phi is concave. Concave-of-LeakyReLU
is not concave because LeakyReLU is convex.

What this test verifies:
    1. ACTIVATION concavity: phi(z) itself is concave in z.
       This is what the architecture currently guarantees.
    2. ACTIVATION monotonicity: phi(z) preserves orderings (needed for
       BT loss to work properly).
    3. HEAD concavity in z (the post-fc2 input to phi): the reward
       averaged over the head_width dimension is concave in z.
       This is also guaranteed by the architecture (mean of concave
       functions is concave).
    4. HEAD concavity in h: this is the STRONGER property the paper's
       global runaway-suppression claim depends on. With the current
       LeakyReLU MLP, this property does NOT hold globally. We test it
       and report results, but FAILURE here is expected with the current
       architecture - it indicates that 'concave reward function in h'
       requires removing the LeakyReLU.

Run from project root:
    python -m src.concavity_test
"""

from __future__ import annotations

import torch

from .heads import (
    ACTIVATION_REGISTRY,
    GLOBALLY_CONCAVE,
    MONOTONIC,
    RewardHead,
)


def test_activation_concavity(
    activation_name: str,
    n_samples: int = 1000,
    z_range: tuple[float, float] = (-10.0, 10.0),
    tol: float = 1e-4,
    seed: int = 0,
) -> dict:
    """Test that phi(z) is concave in its scalar input z.
    
    This is the property the architecture guarantees, regardless of the
    LeakyReLU upstream.
    """
    torch.manual_seed(seed)
    activation = ACTIVATION_REGISTRY[activation_name]()
    activation.eval()
    
    z1 = torch.empty(n_samples, 1).uniform_(*z_range)
    z2 = torch.empty(n_samples, 1).uniform_(*z_range)
    t = torch.empty(n_samples, 1).uniform_(0.0, 1.0)
    
    with torch.no_grad():
        z_mid = t * z1 + (1 - t) * z2
        phi_mid = activation(z_mid)
        phi_chord = t * activation(z1) + (1 - t) * activation(z2)
    
    violation = (phi_chord - phi_mid).clamp(min=0.0)
    n_violations = (violation > tol).sum().item()
    max_violation = violation.max().item()
    
    return {
        "activation_name": activation_name,
        "violations": n_violations,
        "max_violation": max_violation,
        "is_concave": n_violations == 0,
    }


def test_activation_monotonicity(
    activation_name: str,
    n_samples: int = 1000,
    z_range: tuple[float, float] = (-10.0, 10.0),
    tol: float = 1e-5,
    seed: int = 0,
) -> dict:
    """Test that phi(z) is monotonically increasing.
    
    Important for BT loss: if phi is non-monotonic, a 'better' response
    (higher z) can get lower reward, which breaks pairwise discrimination.
    """
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


def test_head_concavity_in_z(
    activation_name: str,
    head_width: int = 32,
    n_samples: int = 500,
    z_scale: float = 2.0,
    tol: float = 1e-4,
    seed: int = 0,
) -> dict:
    """Test that the head is concave in its post-fc2 input z (vector input).
    
    The head computes mean(phi(z)) + b. Mean is a non-negative average
    over the head_width dimension; if phi is concave element-wise, then
    mean(phi(z)) is concave in z.
    
    Test design: we test the chord inequality element-wise across the
    head_width dimension. Each (h, w) entry must satisfy concavity
    independently for the mean to be concave. Otherwise a 'bad'
    dimension can be hidden by averaging with 'good' ones.
    """
    torch.manual_seed(seed)
    activation = ACTIVATION_REGISTRY[activation_name]()
    activation.eval()
    
    z1 = torch.randn(n_samples, head_width) * z_scale
    z2 = torch.randn(n_samples, head_width) * z_scale
    t = torch.empty(n_samples, 1).uniform_(0.0, 1.0)
    
    with torch.no_grad():
        z_mid = t * z1 + (1 - t) * z2
        # Test concavity ELEMENT-WISE before averaging
        phi_mid = activation(z_mid)                    # (n, head_width)
        phi_chord = t * activation(z1) + (1 - t) * activation(z2)
    
    violation = (phi_chord - phi_mid).clamp(min=0.0)
    n_violations = (violation > tol).sum().item()
    max_violation = violation.max().item()
    
    return {
        "activation_name": activation_name,
        "violations": n_violations,
        "max_violation": max_violation,
        "is_concave": n_violations == 0,
    }


def test_full_head_concavity_in_h(
    activation_name: str,
    hidden_size: int = 64,
    n_samples: int = 1000,
    h_scale: float = 2.0,
    tol: float = 1e-4,
    seed: int = 0,
) -> dict:
    """Test concavity of the FULL head r(h) in h (the strongest property).
    
    With the current architecture (Linear -> LeakyReLU -> Linear -> phi),
    this property does NOT hold globally because LeakyReLU is convex.
    Concave-of-convex is not concave in general.
    
    A FAILURE here is EXPECTED with the current MLP architecture. To get
    global concavity in h, the LeakyReLU between fc1 and fc2 would need
    to be removed (collapsing the two linear layers to a single affine
    map, after which concave-of-affine = concave).
    """
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
        "max_violation": max_violation,
        "is_concave": n_violations == 0,
    }


def run_all_tests():
    """Run the four test groups in order from cheapest/most-fundamental
    to most-architectural."""
    
    # ---- Group 1: activation-level concavity (what the arch guarantees) ----
    print("=" * 78)
    print("TEST 1: ACTIVATION CONCAVITY  (phi(z) concave in z)")
    print("       This is what the current architecture guarantees.")
    print("=" * 78)
    print(f"{'activation':<22} {'concave?':<10} {'expected':<10} {'max_viol':<12}")
    print("-" * 78)
    
    activation_concavity_pass = True
    for name in ACTIVATION_REGISTRY:
        result = test_activation_concavity(name)
        expected = GLOBALLY_CONCAVE[name]
        passed = result["is_concave"] == expected
        marker = "OK" if passed else "FAIL"
        activation_concavity_pass = activation_concavity_pass and passed
        print(
            f"{name:<22} {str(result['is_concave']):<10} "
            f"{str(expected):<10} {result['max_violation']:<12.2e} [{marker}]"
        )
    
    # ---- Group 2: monotonicity ----
    print()
    print("=" * 78)
    print("TEST 2: ACTIVATION MONOTONICITY  (phi increasing -> BT-compatible)")
    print("=" * 78)
    print(f"{'activation':<22} {'monotonic?':<12} {'expected':<10}")
    print("-" * 78)
    
    for name in ACTIVATION_REGISTRY:
        result = test_activation_monotonicity(name)
        expected = MONOTONIC[name]
        passed = result["is_monotonic"] == expected
        marker = "OK" if passed else "WARN"  # numerical at boundaries
        print(
            f"{name:<22} {str(result['is_monotonic']):<12} "
            f"{str(expected):<10} [{marker}]"
        )
    
    # ---- Group 3: head concavity in z (post-fc2 input) ----
    print()
    print("=" * 78)
    print("TEST 3: HEAD CONCAVITY IN z  (mean(phi(z)) concave in z)")
    print("       Should match TEST 1; mean preserves concavity.")
    print("=" * 78)
    print(f"{'activation':<22} {'concave?':<10} {'expected':<10} {'max_viol':<12}")
    print("-" * 78)
    
    head_z_concavity_pass = True
    for name in ACTIVATION_REGISTRY:
        result = test_head_concavity_in_z(name)
        expected = GLOBALLY_CONCAVE[name]
        passed = result["is_concave"] == expected
        marker = "OK" if passed else "FAIL"
        head_z_concavity_pass = head_z_concavity_pass and passed
        print(
            f"{name:<22} {str(result['is_concave']):<10} "
            f"{str(expected):<10} {result['max_violation']:<12.2e} [{marker}]"
        )
    
    # ---- Group 4: full-head concavity in h (the stronger property) ----
    print()
    print("=" * 78)
    print("TEST 4: FULL HEAD CONCAVITY IN h  (r(h) concave in h)")
    print("       The STRONG property the paper's global runaway-suppression")
    print("       claim depends on. With the current LeakyReLU MLP, FAILURE")
    print("       IS EXPECTED. Removing the LeakyReLU would restore this.")
    print("=" * 78)
    print(f"{'activation':<22} {'concave_in_h?':<14} {'max_viol':<12}")
    print("-" * 78)
    
    for name in ACTIVATION_REGISTRY:
        result = test_full_head_concavity_in_h(name)
        # We do NOT assert pass/fail here - just report.
        # With the LeakyReLU, all activations will likely FAIL this test.
        # That's an architectural observation, not a code bug.
        print(
            f"{name:<22} {str(result['is_concave']):<14} "
            f"{result['max_violation']:<12.2e}"
        )
    
    # ---- Summary ----
    print()
    print("=" * 78)
    if activation_concavity_pass and head_z_concavity_pass:
        print("[CORE PROPERTIES PASS] Activations are concave in z. Head is")
        print("concave in its post-fc2 input. This is what the current")
        print("architecture guarantees and what TESTS 1 + 3 verify.")
        print()
        print("TEST 4 (concavity in h) is expected to fail with the LeakyReLU")
        print("MLP. This is consistent with the team architecture and is not")
        print("a code bug - it's a design choice with implications for the")
        print("paper's claims.")
        return 0
    else:
        print("[CORE PROPERTIES FAILED] Check FAIL rows above.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(run_all_tests())
"""E123 — Differentiable Sinkhorn top-K (Cuturi 2019, Xie et al. 2020).

Replaces `torch.topk(values, K).mean()` in our per-net-trace congestion with
a Sinkhorn-relaxed assignment that is differentiable through the SELECTION
itself, not just through the values of selected elements.

Background
----------
`torch.topk` gradient: zero on non-selected elements; gradient flows only
through the K largest values' magnitudes. When the K-th and (K+1)-th
values are nearly equal — which is common at the ABU-5 % boundary on
smooth basins — the loss landscape has discontinuities where the
selection set flips. Adam's momentum smooths some of this but the
underlying gradient is still piecewise constant in selection space.

Cuturi-Teboul-Vert (NeurIPS 2019) cast differentiable sorting/top-K as
a regularized optimal-transport (OT) problem. Xie-Wang-Wang
(NeurIPS 2020) gave the specific top-K formulation we use here:

    Inputs:
      x ∈ ℝ^N      (the values to select from)
      K ∈ ℕ        (number to select)
      ε > 0        (entropy regularization; smaller = closer to exact topk)

    Cost matrix C ∈ ℝ^{N×2}:
      C[i, 0] = -x[i]   (cheap to assign value i to "selected" if x[i] is large)
      C[i, 1] =  0      (cost zero to assign to "non-selected")

    Marginal constraints:
      Σ_j T[i, j] = 1/N  ∀i   (each value has unit mass)
      Σ_i T[i, 0] = K/N         (total selected mass = K/N)
      Σ_i T[i, 1] = (N-K)/N     (total non-selected mass = (N-K)/N)

    Regularized OT:
      T* = argmin_T <T, C> - ε · H(T)
    where H(T) = -Σ_{i,j} T[i,j]·log(T[i,j]/(μ_i ν_j)).

    Sinkhorn-Knopp fixed-point: alternate K-S rescaling of rows/columns
    until convergence. T*[i, 0] · N is the relaxed assignment of x[i] to
    "selected" (the soft top-K membership in [0, 1]).

    Differentiable top-K mean:
      mean ≈ (1/K) · Σ_i x[i] · T*[i, 0] · N

As ε → 0, T*[i, 0] · N → 1 for the K largest x[i], 0 otherwise — exact
topk recovered. For ε > 0, the gradient is smooth on all N inputs with
strength proportional to proximity to the K-th value.

Implementation notes
--------------------
- 2-anchor (selected/non-selected) is far cheaper than the N-anchor sort
  formulation. For our use case (top-5 % MEAN, not a full sort), we
  only need the soft membership, not soft ranks.
- We work in log-space (u, v are log-potentials, T = exp((u + v - C)/ε)·μν).
  This is numerically stable and avoids 0/inf in the multiplication form.
- Mass normalization choice: we use μ_i = 1 (unit mass per value) and
  ν_0 = K, ν_1 = N - K (mass at each anchor). This is equivalent up to
  scaling to the standard formulation above.
- Number of Sinkhorn iters: 50 is overkill in our regime (ε ≥ 0.01·std
  converges in ~20 iters). 100 iters costs ~150 µs per call on our N.
"""
from __future__ import annotations

from typing import Tuple

import torch


def sinkhorn_topk_mean(
    values: torch.Tensor,
    k: int,
    epsilon: float = 0.1,
    n_iters: int = 50,
    return_assignment: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Compute differentiable mean of (approximate) top-k values.

    Args:
      values: 1-D float tensor of shape [N]. Higher = "more selected".
      k: int, number of elements in the top-K mean.
      epsilon: > 0, entropy regularization. Smaller = closer to torch.topk,
               but harder to optimize (less gradient on non-top cells).
               Typical: 0.05-1.0 of std(values). We auto-scale by std internally.
      n_iters: number of Sinkhorn fixed-point iterations.
      return_assignment: if True, return (mean, soft_assignment) where
        soft_assignment ∈ [0, 1]^N is the relaxed top-K membership.

    Returns:
      A scalar tensor: differentiable approximation of mean(topk(values, k)).
      If return_assignment, also a [N] tensor with soft membership.
    """
    if values.dim() != 1:
        raise ValueError(f"sinkhorn_topk_mean expects 1-D, got shape {values.shape}")
    N = values.numel()
    if k <= 0 or k > N:
        raise ValueError(f"k={k} out of range [1, {N}]")
    if k == N:
        return values.mean()
    # Auto-bump iters when ε is very small (Sinkhorn convergence rate is ~1/ε):
    # at ε=0.05 we need ~100 iters; at ε=0.02 we need ~200; etc.
    if epsilon < 0.05 and n_iters < 100:
        n_iters = max(n_iters, int(50 / max(epsilon, 0.005)))

    # Auto-scale epsilon by std of values. This makes the smoothness
    # of the OT problem invariant to the absolute scale of the values
    # (which varies widely across benches and iters of Adam descent).
    # When std is very small (degenerate basin), use a small constant
    # so ε doesn't underflow.
    val_std = values.std()
    # Detach: epsilon should be a constant for stable optimization,
    # not a function of values that backprop touches.
    eps = (epsilon * val_std).detach().clamp(min=1e-8)

    # Negative cost on "selected" anchor: high-value cells prefer it.
    # Build the [N, 2] log-kernel matrix
    #   log(K[i, j]) = -C[i, j] / eps
    #   C[i, 0] = -x[i],  C[i, 1] = 0.
    # So log_K[i, 0] = x[i] / eps, log_K[i, 1] = 0.
    log_K = torch.zeros(N, 2, device=values.device, dtype=values.dtype)
    log_K[:, 0] = values / eps  # value at "selected" anchor

    # Mass marginals (in log space):
    #   Σ_j T[i, j] = mu_i = 1   ∀i      → log_mu = 0
    #   Σ_i T[i, 0] = nu_0 = k
    #   Σ_i T[i, 1] = nu_1 = N - k
    log_mu = torch.zeros(N, device=values.device, dtype=values.dtype)
    log_nu = torch.tensor(
        [float(k), float(N - k)], device=values.device, dtype=values.dtype
    ).log()

    # Sinkhorn iterations in log space.
    # T[i, j] = exp(f[i] + g[j] + log_K[i, j])
    # Constraints:
    #   log Σ_j exp(f[i] + g[j] + log_K[i, j]) = log_mu[i]
    #   log Σ_i exp(f[i] + g[j] + log_K[i, j]) = log_nu[j]
    # Update:
    #   f[i] ← log_mu[i] - logsumexp_j (g[j] + log_K[i, j])
    #   g[j] ← log_nu[j] - logsumexp_i (f[i] + log_K[i, j])
    f = torch.zeros(N, device=values.device, dtype=values.dtype)
    g = torch.zeros(2, device=values.device, dtype=values.dtype)
    for _ in range(n_iters):
        # f-update: row marginals match log_mu (= 0)
        # log Σ_j exp(g[j] + log_K[i, j]) → [N]
        f = log_mu - torch.logsumexp(g.unsqueeze(0) + log_K, dim=1)
        # g-update: column marginals match log_nu = [log k, log(N-k)]
        # log Σ_i exp(f[i] + log_K[i, j]) → [2]
        g = log_nu - torch.logsumexp(f.unsqueeze(1) + log_K, dim=0)

    # Final T:  T[i, 0] = exp(f[i] + g[0] + log_K[i, 0])
    # The mass "selected to top-K" of value i is T[i, 0]. By Sinkhorn
    # convergence Σ_i T[i, 0] = k. The soft membership in [0, 1] is
    # T[i, 0] itself (since μ_i = 1 means each value has unit mass and at
    # most one unit can land on the "selected" anchor).
    log_T0 = f + g[0] + log_K[:, 0]  # [N]
    T0 = log_T0.exp()  # soft membership ∈ [0, 1] (approximately)

    # Differentiable top-K mean: weight each value by its membership, then
    # divide by k (the total mass on "selected").
    mean_topk = (values * T0).sum() / k

    if return_assignment:
        return mean_topk, T0
    return mean_topk


# ---------------------------------------------------------------------------
# Convenience wrappers and tests
# ---------------------------------------------------------------------------

def sinkhorn_topk_assignment(
    values: torch.Tensor,
    k: int,
    epsilon: float = 0.1,
    n_iters: int = 50,
) -> torch.Tensor:
    """Return only the [N] soft membership vector."""
    _, T0 = sinkhorn_topk_mean(values, k, epsilon, n_iters, return_assignment=True)
    return T0


def _self_test():
    """Smoke tests: ε→0 recovers torch.topk; ε large gives mean."""
    import time

    torch.manual_seed(0)
    N = 4488  # ibm17 size: 2*44*51
    K = max(1, int(0.05 * N))  # 224

    print(f"N={N}, K={K}")

    # Test 1: epsilon small → match torch.topk
    print("\nTest 1: epsilon sweep (no grad)")
    for eps in [1.0, 0.5, 0.1, 0.05, 0.01, 0.005]:
        x = torch.randn(N) * 10.0
        true_mean = torch.topk(x, K).values.mean()
        with torch.no_grad():
            sink_mean = sinkhorn_topk_mean(x, K, epsilon=eps, n_iters=80)
        rel = (sink_mean.item() - true_mean.item()) / abs(true_mean.item()) * 100
        print(f"  eps={eps:.3f}  topk={true_mean:.4f}  sinkhorn={sink_mean:.4f}  rel={rel:+.2f}%")

    # Test 2: timing
    print("\nTest 2: timing (CPU)")
    x = torch.randn(N) * 10.0
    t0 = time.time()
    for _ in range(10):
        with torch.no_grad():
            sinkhorn_topk_mean(x, K, epsilon=0.1, n_iters=50)
    t_no_grad = (time.time() - t0) / 10
    print(f"  no_grad  50 iters: {t_no_grad*1000:.1f} ms")

    base = torch.randn(N) * 10.0
    t0 = time.time()
    for _ in range(10):
        x = base.clone().requires_grad_(True)
        out = sinkhorn_topk_mean(x, K, epsilon=0.1, n_iters=50)
        out.backward()
    t_grad = (time.time() - t0) / 10
    print(f"  with grad 50 iters: {t_grad*1000:.1f} ms")

    # Test 3: gradient flow on non-top elements
    print("\nTest 3: gradient flow on non-top cells (eps=0.1)")
    x_base = torch.randn(20) * 10.0
    x = x_base.clone().requires_grad_(True)
    out = sinkhorn_topk_mean(x, 3, epsilon=0.1, n_iters=80)
    out.backward()
    sorted_vals, idx = torch.sort(x.detach(), descending=True)
    sorted_grad = x.grad[idx]
    print(f"  values:      {[f'{v:.2f}' for v in sorted_vals.tolist()]}")
    print(f"  gradient:    {[f'{v:.3f}' for v in sorted_grad.tolist()]}")
    # Gradient should be: ~1/K on top-K, but with SMOOTH transition to near-zero on non-top.
    # vs torch.topk which would be exactly 1/K on the first 3, exactly 0 on the rest.

    # Test 4: properties — soft assignment sums to k
    print("\nTest 4: soft assignment sum (eps=0.1)")
    x = torch.randn(N) * 10.0
    with torch.no_grad():
        _, T0 = sinkhorn_topk_mean(x, K, epsilon=0.1, n_iters=50, return_assignment=True)
        sum_T = T0.sum()
        # Min/max should be in [0, 1]
        T_min = T0.min()
        T_max = T0.max()
    print(f"  sum(T0) = {sum_T:.2f}  (expected = K = {K})")
    print(f"  min(T0) = {T_min:.6f}  max(T0) = {T_max:.6f}")


if __name__ == "__main__":
    _self_test()

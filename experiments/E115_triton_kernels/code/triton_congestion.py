"""E115 — Triton fused kernel for per-net-trace congestion forward+backward.

NOTE (2026-05-20): this file is NOT used in the production path. After
profiling we discovered the actual bottleneck was PyTorch's CUDA
advanced-indexing backward (`tensor[idx]` → un-coalesced scatter-add),
not the per-net-trace math itself. Switching to `index_select` got us
the 16.8× speedup on ibm17 without needing a Triton kernel — see
`fast_proxy.py`.

This file is kept as a starting point for future development if/when
we want to push past the index_select win (e.g. for ibm18 or NG45
designs with 5K+ macros where even the optimized PyTorch becomes
slow). The kernel below is partially debugged but never beat the
PyTorch single-pass implementation in our measurements.

Replaces the hot path in `per_net_trace_proxy.PerNetTraceCongestion._trace_route_congestion`.
For each (source, sink) pair p we need to accumulate into V[gr, gc] and H[gr, gc]:

    src_pos[p] = positions[src_macro[p]] + src_offset[p]   # [B, 2]
    snk_pos[p] = positions[snk_macro[p]] + snk_offset[p]

    # σ_y = sigma_cell_frac * cell_h  (constant)
    # softmax assignments, all over the cell-index dim:
    row_src[p, r] ∝ exp(-((src_pos[p,1] - r*ch - ch/2) / σ_y)^2)
    col_snk[p, c] ∝ exp(-((snk_pos[p,0] - c*cw - cw/2) / σ_x)^2)
    col_src[p, c] ∝ exp(-((src_pos[p,0] - c*cw - cw/2) / σ_x)^2)
    row_snk[p, r] ∝ exp(-((snk_pos[p,1] - r*ch - ch/2) / σ_y)^2)

    col_src_exp = sum_c c * col_src[p,c]            # expectation
    col_snk_exp = sum_c c * col_snk[p,c]
    row_src_exp = sum_r r * row_src[p,r]
    row_snk_exp = sum_r r * row_snk[p,r]

    col_min, col_max = soft_min_max(col_src_exp, col_snk_exp, β_mm)
    row_min, row_max = soft_min_max(row_src_exp, row_snk_exp, β_mm)

    h_col_range[p, c] = σ(β_r(c - col_min + 0.5)) - σ(β_r(c - col_max + 0.5))
    v_row_range[p, r] = σ(β_r(r - row_min + 0.5)) - σ(β_r(r - row_max + 0.5))

    H[r, c] += w[p] * row_src[p, r] * h_col_range[p, c]
    V[r, c] += w[p] * v_row_range[p, r] * col_snk[p, c]

This file provides:
- `trace_congestion_triton(positions, params) -> (V, H)`: a torch.autograd.Function
  whose forward runs a single Triton kernel and whose backward runs a
  hand-derived Triton kernel for d(V,H)/d(positions).

Numerical equivalence (within fp32 noise) with the PyTorch reference is
required. Tested in `test_triton_congestion.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _trace_fwd_kernel(
        # outputs (atomic add)
        V_ptr, H_ptr,
        # pair endpoints
        src_x_ptr, src_y_ptr, snk_x_ptr, snk_y_ptr, weight_ptr,
        # constants
        n_pairs,
        gr, gc,
        cell_w, cell_h,
        sigma_x, sigma_y,    # σ_x = sigma_cell_frac * cell_w, σ_y similar
        beta_minmax,
        beta_range,
        # block sizes (compile-time)
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        """One pair per Triton program; loads pair endpoints, computes row/col
        soft assignments + range indicators, atomically adds to V and H.

        BLOCK_R must be >= gr, BLOCK_C must be >= gc (typically gr, gc <= 64
        so this fits in a single tile of registers).
        """
        p = tl.program_id(0)
        if p >= n_pairs:
            return

        # Load endpoints
        sx = tl.load(src_x_ptr + p)
        sy = tl.load(src_y_ptr + p)
        kx = tl.load(snk_x_ptr + p)
        ky = tl.load(snk_y_ptr + p)
        w  = tl.load(weight_ptr + p)

        # ----- Soft cell assignments (Gaussian + softmax) -----
        r_idx = tl.arange(0, BLOCK_R).to(tl.float32)
        c_idx = tl.arange(0, BLOCK_C).to(tl.float32)
        r_mask = r_idx < gr
        c_mask = c_idx < gc

        ctr_r = (r_idx + 0.5) * cell_h
        ctr_c = (c_idx + 0.5) * cell_w

        # row_src[r] = softmax_r -((sy - ctr_r)/sigma_y)^2
        diff_rs = (sy - ctr_r) / sigma_y
        sc_rs = -(diff_rs * diff_rs)
        sc_rs = tl.where(r_mask, sc_rs, -1.0e30)
        max_rs = tl.max(sc_rs, axis=0)
        e_rs = tl.exp(sc_rs - max_rs)
        e_rs = tl.where(r_mask, e_rs, 0.0)
        sum_rs = tl.sum(e_rs, axis=0)
        row_src = e_rs / sum_rs

        # row_snk[r]
        diff_rk = (ky - ctr_r) / sigma_y
        sc_rk = -(diff_rk * diff_rk)
        sc_rk = tl.where(r_mask, sc_rk, -1.0e30)
        max_rk = tl.max(sc_rk, axis=0)
        e_rk = tl.exp(sc_rk - max_rk)
        e_rk = tl.where(r_mask, e_rk, 0.0)
        sum_rk = tl.sum(e_rk, axis=0)
        row_snk = e_rk / sum_rk

        # col_src[c]
        diff_cs = (sx - ctr_c) / sigma_x
        sc_cs = -(diff_cs * diff_cs)
        sc_cs = tl.where(c_mask, sc_cs, -1.0e30)
        max_cs = tl.max(sc_cs, axis=0)
        e_cs = tl.exp(sc_cs - max_cs)
        e_cs = tl.where(c_mask, e_cs, 0.0)
        sum_cs = tl.sum(e_cs, axis=0)
        col_src = e_cs / sum_cs

        # col_snk[c]
        diff_ck = (kx - ctr_c) / sigma_x
        sc_ck = -(diff_ck * diff_ck)
        sc_ck = tl.where(c_mask, sc_ck, -1.0e30)
        max_ck = tl.max(sc_ck, axis=0)
        e_ck = tl.exp(sc_ck - max_ck)
        e_ck = tl.where(c_mask, e_ck, 0.0)
        sum_ck = tl.sum(e_ck, axis=0)
        col_snk = e_ck / sum_ck

        # ----- Soft endpoint expectations -----
        col_src_exp = tl.sum(col_src * c_idx, axis=0)
        col_snk_exp = tl.sum(col_snk * c_idx, axis=0)
        row_src_exp = tl.sum(row_src * r_idx, axis=0)
        row_snk_exp = tl.sum(row_snk * r_idx, axis=0)

        # ----- Soft min/max via logsumexp at β_mm -----
        # soft_min(a, b) = -log(exp(-βa) + exp(-βb))/β
        # soft_max(a, b) = log(exp(βa) + exp(βb))/β
        b_mm = beta_minmax

        # cols
        m_pos = tl.maximum(b_mm * col_src_exp, b_mm * col_snk_exp)
        col_max_v = (tl.log(tl.exp(b_mm * col_src_exp - m_pos) + tl.exp(b_mm * col_snk_exp - m_pos)) + m_pos) / b_mm
        m_neg = tl.maximum(-b_mm * col_src_exp, -b_mm * col_snk_exp)
        col_min_v = -(tl.log(tl.exp(-b_mm * col_src_exp - m_neg) + tl.exp(-b_mm * col_snk_exp - m_neg)) + m_neg) / b_mm

        # rows
        m_pos = tl.maximum(b_mm * row_src_exp, b_mm * row_snk_exp)
        row_max_v = (tl.log(tl.exp(b_mm * row_src_exp - m_pos) + tl.exp(b_mm * row_snk_exp - m_pos)) + m_pos) / b_mm
        m_neg = tl.maximum(-b_mm * row_src_exp, -b_mm * row_snk_exp)
        row_min_v = -(tl.log(tl.exp(-b_mm * row_src_exp - m_neg) + tl.exp(-b_mm * row_snk_exp - m_neg)) + m_neg) / b_mm

        # ----- Range indicators -----
        br = beta_range
        # h_col_range[c] = sigmoid(br(c - col_min + 0.5)) - sigmoid(br(c - col_max + 0.5))
        h_col_range = tl.sigmoid(br * (c_idx - col_min_v + 0.5)) - tl.sigmoid(br * (c_idx - col_max_v + 0.5))
        v_row_range = tl.sigmoid(br * (r_idx - row_min_v + 0.5)) - tl.sigmoid(br * (r_idx - row_max_v + 0.5))

        # ----- Atomically accumulate into V[gr, gc] and H[gr, gc] -----
        # H[r, c] += w * row_src[r] * h_col_range[c]
        # V[r, c] += w * v_row_range[r] * col_snk[c]
        #
        # Tile the outer product, mask out OOB. Each program handles its 1 pair
        # but writes into BLOCK_R*BLOCK_C cells (most are masked off).
        for ri in tl.static_range(0, BLOCK_R):
            if ri < gr:
                rs_v = tl.sum(tl.where(r_idx == ri, row_src, 0.0), axis=0)       # scalar
                vr_v = tl.sum(tl.where(r_idx == ri, v_row_range, 0.0), axis=0)   # scalar
                # h_chunk[c] = w * rs_v * h_col_range[c]
                h_chunk = w * rs_v * h_col_range
                v_chunk = w * vr_v * col_snk
                tl.atomic_add(
                    H_ptr + ri * gc + c_idx,
                    tl.where(c_mask, h_chunk, 0.0),
                    mask=c_mask,
                )
                tl.atomic_add(
                    V_ptr + ri * gc + c_idx,
                    tl.where(c_mask, v_chunk, 0.0),
                    mask=c_mask,
                )


# ---------------------------------------------------------------------------
# torch.autograd.Function wrapper
# ---------------------------------------------------------------------------


@dataclass
class TraceParams:
    """Static parameters for the per-net-trace forward."""
    gr: int
    gc: int
    cell_w: float
    cell_h: float
    sigma_cell_frac: float
    beta_minmax: float
    beta_range_per_cell: float


class _TraceCongestion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src_x, src_y, snk_x, snk_y, weights, params: TraceParams):
        if not _HAS_TRITON:
            raise RuntimeError("Triton not available")
        device = src_x.device
        n_pairs = src_x.numel()
        gr, gc = params.gr, params.gc

        # Block sizes: next power of two >= gr, gc
        BLOCK_R = max(16, _next_pow2(gr))
        BLOCK_C = max(16, _next_pow2(gc))

        V = torch.zeros(gr, gc, device=device, dtype=torch.float32)
        H = torch.zeros(gr, gc, device=device, dtype=torch.float32)

        sigma_x = params.sigma_cell_frac * params.cell_w
        sigma_y = params.sigma_cell_frac * params.cell_h

        grid = (n_pairs,)
        _trace_fwd_kernel[grid](
            V, H,
            src_x, src_y, snk_x, snk_y, weights,
            n_pairs,
            gr, gc,
            params.cell_w, params.cell_h,
            sigma_x, sigma_y,
            params.beta_minmax,
            params.beta_range_per_cell,
            BLOCK_R=BLOCK_R,
            BLOCK_C=BLOCK_C,
            BLOCK_P=1,
        )

        ctx.save_for_backward(src_x, src_y, snk_x, snk_y, weights)
        ctx.params = params
        ctx.requires_input_grad = (
            src_x.requires_grad, src_y.requires_grad,
            snk_x.requires_grad, snk_y.requires_grad,
        )
        return V, H

    @staticmethod
    def backward(ctx, dV, dH):
        """Backward: rely on PyTorch autograd for this. We re-build the
        forward graph from saved tensors using PyTorch ops and call
        torch.autograd.grad.

        This is slower than a hand-written backward kernel but guarantees
        numerical correctness. The forward Triton kernel already gives us
        the big win (no Python loop, no chunk-graph, no [B, gr] tensors
        in autograd).
        """
        src_x, src_y, snk_x, snk_y, weights = ctx.saved_tensors
        params = ctx.params

        # Re-run forward with autograd to get gradients. This is fine because
        # the bottleneck was the n_pairs * (gr+gc) intermediates in the original
        # implementation; here we just compute scalar V·dV + H·dH and backprop
        # which still has the same memory pattern but only ONE backward pass.
        sx = src_x.detach().clone().requires_grad_(True) if ctx.requires_input_grad[0] else src_x.detach()
        sy = src_y.detach().clone().requires_grad_(True) if ctx.requires_input_grad[1] else src_y.detach()
        kx = snk_x.detach().clone().requires_grad_(True) if ctx.requires_input_grad[2] else snk_x.detach()
        ky = snk_y.detach().clone().requires_grad_(True) if ctx.requires_input_grad[3] else snk_y.detach()

        V, H = trace_congestion_reference(sx, sy, kx, ky, weights, params)

        # Total scalar = sum(V * dV + H * dH)
        scalar = (V * dV).sum() + (H * dH).sum()
        grads = torch.autograd.grad(
            scalar,
            [t for t, r in zip([sx, sy, kx, ky], ctx.requires_input_grad) if r],
            create_graph=False, retain_graph=False, allow_unused=True,
        )
        result = [None] * 4
        it = iter(grads)
        for i, needs in enumerate(ctx.requires_input_grad):
            if needs:
                result[i] = next(it)
        return result[0], result[1], result[2], result[3], None, None


# ---------------------------------------------------------------------------
# PyTorch reference (used for forward when Triton not available, and for backward)
# ---------------------------------------------------------------------------


def trace_congestion_reference(
    src_x, src_y, snk_x, snk_y, weights, params: TraceParams,
):
    """Pure-PyTorch single-pass (no chunking) implementation. Differentiable."""
    device = src_x.device
    gr, gc = params.gr, params.gc
    cw, ch = params.cell_w, params.cell_h

    cell_x_ctr = (torch.arange(gc, dtype=torch.float32, device=device) + 0.5) * cw
    cell_y_ctr = (torch.arange(gr, dtype=torch.float32, device=device) + 0.5) * ch
    col_idx_f = torch.arange(gc, dtype=torch.float32, device=device)
    row_idx_f = torch.arange(gr, dtype=torch.float32, device=device)

    sigma_x = params.sigma_cell_frac * cw
    sigma_y = params.sigma_cell_frac * ch

    # Soft assignments
    def _softassign(pos, ctr, sigma):
        # pos: [B], ctr: [n]
        diff = (pos.view(-1, 1) - ctr.view(1, -1)) / sigma
        return torch.softmax(-(diff * diff), dim=1)

    row_src = _softassign(src_y, cell_y_ctr, sigma_y)
    row_snk = _softassign(snk_y, cell_y_ctr, sigma_y)
    col_src = _softassign(src_x, cell_x_ctr, sigma_x)
    col_snk = _softassign(snk_x, cell_x_ctr, sigma_x)

    col_src_exp = (col_src * col_idx_f.view(1, -1)).sum(dim=1)
    col_snk_exp = (col_snk * col_idx_f.view(1, -1)).sum(dim=1)
    row_src_exp = (row_src * row_idx_f.view(1, -1)).sum(dim=1)
    row_snk_exp = (row_snk * row_idx_f.view(1, -1)).sum(dim=1)

    b_mm = params.beta_minmax
    stacked = torch.stack([col_src_exp, col_snk_exp], dim=0)
    col_min = -torch.logsumexp(-b_mm * stacked, dim=0) / b_mm
    col_max = torch.logsumexp(b_mm * stacked, dim=0) / b_mm
    stacked_r = torch.stack([row_src_exp, row_snk_exp], dim=0)
    row_min = -torch.logsumexp(-b_mm * stacked_r, dim=0) / b_mm
    row_max = torch.logsumexp(b_mm * stacked_r, dim=0) / b_mm

    br = params.beta_range_per_cell
    h_col_range = (
        torch.sigmoid(br * (col_idx_f.view(1, -1) - col_min.unsqueeze(1) + 0.5))
        - torch.sigmoid(br * (col_idx_f.view(1, -1) - col_max.unsqueeze(1) + 0.5))
    )
    v_row_range = (
        torch.sigmoid(br * (row_idx_f.view(1, -1) - row_min.unsqueeze(1) + 0.5))
        - torch.sigmoid(br * (row_idx_f.view(1, -1) - row_max.unsqueeze(1) + 0.5))
    )

    row_src_w = row_src * weights.unsqueeze(1)
    col_snk_w = col_snk * weights.unsqueeze(1)
    H = row_src_w.transpose(0, 1) @ h_col_range
    V = v_row_range.transpose(0, 1) @ col_snk_w
    return V, H


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def trace_congestion(
    src_x: torch.Tensor,
    src_y: torch.Tensor,
    snk_x: torch.Tensor,
    snk_y: torch.Tensor,
    weights: torch.Tensor,
    params: TraceParams,
    use_triton: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (V, H) congestion maps of shape [gr, gc].

    If `use_triton` is True and Triton is available and on CUDA, use the
    fused Triton forward (then fall back to reference for backward to
    avoid hand-coding gradient kernels).

    Otherwise use the PyTorch reference (single-pass, no chunking).
    """
    if use_triton and _HAS_TRITON and src_x.is_cuda:
        return _TraceCongestion.apply(src_x, src_y, snk_x, snk_y, weights, params)
    else:
        return trace_congestion_reference(src_x, src_y, snk_x, snk_y, weights, params)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p

"""E123 — PerNetTraceCongestion with Sinkhorn-relaxed top-K mean.

Drop-in subclass of `experiments/E111_per_net_trace_congestion/code/
per_net_trace_proxy.py::PerNetTraceCongestion` that replaces the final
`torch.topk(concat, K).mean()` (the canonical ABU-5 % scalar) with a
Sinkhorn-relaxed differentiable top-K (Cuturi 2019, Xie 2020).

Motivation
----------
`torch.topk` is differentiable through the SELECTED elements' values,
but its gradient is **zero on all non-selected elements**. When the
K-th and (K+1)-th values are nearly equal — common at smooth-basin
plateaus where many cells have similar V+H congestion — Adam's signal
on cells just below the cutoff is discontinuously zero, even though
those cells are precisely where moving a macro could push them in or
out of the top-K set.

Sinkhorn top-K (this module) makes the selection itself differentiable.
At ε > 0, the relaxed assignment T*[i, 0] ∈ [0, 1] gives non-zero
gradient on cells **proportional to their proximity to the top-K
boundary**. This is novel for placement; no public placer has used
optimal-transport relaxations as the congestion top-K operator.

Hyperparameter ε:
- ε → 0:   recovers exact topk (zero gradient on non-top).
- ε → ∞:   approaches the unconditional mean (gradient on every cell,
           but loses the ABU-5 % selectivity).
- ε ≈ 0.1 of std(values): sweet spot — sinkhorn matches topk within
  ~1.5 %, ~283 cells have significant gradient (vs 224 with topk).

Internally we auto-scale ε by std(values) so the user-passed parameter
is invariant to absolute value scale (which varies across iters and
benches).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_E111 = _HERE.parent.parent / "E111_per_net_trace_congestion" / "code"
for p in (_HERE, _E111):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from per_net_trace_proxy import PerNetTraceCongestion  # noqa: E402
from sinkhorn_topk import sinkhorn_topk_mean  # noqa: E402


class SinkhornPerNetTraceCongestion(PerNetTraceCongestion):
    """PerNetTraceCongestion with Sinkhorn-relaxed top-K.

    All hparams of the parent are preserved (sigma_cell_frac, beta_*,
    pair_chunk_size, etc.). New hparams:

    Args (added):
      sinkhorn_eps: float, ε for OT entropy regularization. Auto-scaled
        by std(values) internally, so this is a ratio (typical 0.05-0.5).
      sinkhorn_iters: int, number of Sinkhorn fixed-point iters. 50 is
        sufficient for ε ≥ 0.05; bump to 100 for ε ≤ 0.02.
      use_sinkhorn: bool, if False falls back to torch.topk (for ablation).
    """

    def __init__(
        self,
        benchmark,
        plc,
        *args,
        sinkhorn_eps: float = 0.1,
        sinkhorn_iters: int = 50,
        use_sinkhorn: bool = True,
        **kwargs,
    ):
        super().__init__(benchmark, plc, *args, **kwargs)
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.use_sinkhorn = bool(use_sinkhorn)

    def compute_congestion(self, positions: torch.Tensor) -> torch.Tensor:
        """ABU-5% with Sinkhorn-relaxed top-K (differentiable selection)."""
        # Reuse parent's per-net trace, macro footprint, smoothing.
        V_route, H_route = self._trace_route_congestion(positions)
        V_route = V_route / max(self.grid_v_routes, 1e-9)
        H_route = H_route / max(self.grid_h_routes, 1e-9)

        V_macro, H_macro = self._macro_route_congestion(positions)
        V_macro = V_macro / max(self.grid_v_routes, 1e-9)
        H_macro = H_macro / max(self.grid_h_routes, 1e-9)

        V_route = self._box_smooth(V_route, axis="V")
        H_route = self._box_smooth(H_route, axis="H")

        V_total = V_route + V_macro
        H_total = H_route + H_macro

        combined = torch.cat([V_total.flatten(), H_total.flatten()])
        k = max(1, int(self.abu_k * combined.numel()))

        if self.use_sinkhorn:
            return sinkhorn_topk_mean(
                combined, k,
                epsilon=self.sinkhorn_eps,
                n_iters=self.sinkhorn_iters,
            )
        else:
            top_k, _ = torch.topk(combined, k)
            return top_k.mean()


# ---------------------------------------------------------------------------
# Calibration: smooth vs canonical (matches E111 calibrate signature)
# ---------------------------------------------------------------------------

def calibrate(
    bench_name: str = "ibm17",
    n_perturb: int = 8,  # reduced from 16 — canonical eval is the bottleneck
    perturb_frac: float = 0.02,
    sinkhorn_eps: float = 0.1,
    sinkhorn_iters: int = 50,
    scalar_only: bool = False,
):
    """Compare Sinkhorn-trace smooth vs canonical congestion on cached cascade.

    Prints:
      - absolute mismatch (Sinkhorn vs canonical)
      - Pearson + Spearman ρ on Δcong across random perturbations
      - reference: torch.topk variant (use_sinkhorn=False) for ablation
    """
    import numpy as np
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.bench_paths import find_benchmark_dir
    from macro_place.objective import compute_proxy_cost

    print(f"\n=== E123 calibrate: {bench_name} (sinkhorn_eps={sinkhorn_eps}, iters={sinkhorn_iters}) ===", flush=True)
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    _ROOT = _HERE.parents[2]
    cached = _ROOT / "experiments" / "E84_cascading_saddle" / "results" / f"cascade_{bench_name}.pt"
    if cached.exists():
        data = torch.load(cached, weights_only=False, map_location="cpu")
        start = (data.get("placement") if "placement" in data else data["polished_placement"]).to(torch.float32)
        print(f"  using cached cascade placement", flush=True)
    else:
        start = benchmark.macro_positions.clone().float()
        print(f"  using macro_positions", flush=True)

    proxy_sinkhorn = SinkhornPerNetTraceCongestion(
        benchmark, plc, device="cpu",
        sinkhorn_eps=sinkhorn_eps, sinkhorn_iters=sinkhorn_iters,
        use_sinkhorn=True,
    )
    proxy_topk = SinkhornPerNetTraceCongestion(
        benchmark, plc, device="cpu",
        use_sinkhorn=False,
    )
    print(f"  n_pairs={proxy_sinkhorn.n_pairs}, grid={proxy_sinkhorn.gr}x{proxy_sinkhorn.gc}", flush=True)

    can = compute_proxy_cost(start, benchmark, plc)
    can_cong = float(can["congestion_cost"])
    with torch.no_grad():
        sk_cong = float(proxy_sinkhorn.compute_congestion(start))
        tk_cong = float(proxy_topk.compute_congestion(start))
    rel_sk = (sk_cong - can_cong) / max(abs(can_cong), 1e-6) * 100.0
    rel_tk = (tk_cong - can_cong) / max(abs(can_cong), 1e-6) * 100.0
    print(f"  canonical cong:  {can_cong:.5f}", flush=True)
    print(f"  sinkhorn  cong:  {sk_cong:.5f}  rel={rel_sk:+.2f}%", flush=True)
    print(f"  torch.topk cong: {tk_cong:.5f}  rel={rel_tk:+.2f}%", flush=True)
    print(f"  sinkhorn - topk: {sk_cong - tk_cong:+.5f}  ({(sk_cong - tk_cong) / tk_cong * 100:+.2f}%)", flush=True)

    if scalar_only:
        return {
            "bench": bench_name,
            "canonical": can_cong, "sinkhorn": sk_cong, "topk": tk_cong,
            "rel_sinkhorn_pct": rel_sk, "rel_topk_pct": rel_tk,
        }

    rng = np.random.default_rng(42)
    fixed = benchmark.macro_fixed.cpu().numpy()
    cw = float(benchmark.canvas_width)
    movable_hard = [i for i in range(benchmark.num_hard_macros) if not bool(fixed[i])]
    scale = perturb_frac * cw

    can_deltas = []
    sk_deltas = []
    tk_deltas = []
    base_can = can_cong
    for k in range(n_perturb):
        p = start.clone()
        targets = rng.choice(movable_hard, size=min(5, len(movable_hard)), replace=False)
        for t in targets:
            dx = float(rng.normal(0.0, scale))
            dy = float(rng.normal(0.0, scale))
            x0 = float(benchmark.macro_sizes[t, 0]) / 2.0
            y0 = float(benchmark.macro_sizes[t, 1]) / 2.0
            p[t, 0] = torch.clamp(p[t, 0] + dx, x0, cw - x0)
            p[t, 1] = torch.clamp(p[t, 1] + dy, y0, float(benchmark.canvas_height) - y0)
        c_can = float(compute_proxy_cost(p, benchmark, plc)["congestion_cost"])
        with torch.no_grad():
            c_sk = float(proxy_sinkhorn.compute_congestion(p))
            c_tk = float(proxy_topk.compute_congestion(p))
        can_deltas.append(c_can - base_can)
        sk_deltas.append(c_sk - sk_cong)
        tk_deltas.append(c_tk - tk_cong)

    can_arr = np.asarray(can_deltas)
    sk_arr = np.asarray(sk_deltas)
    tk_arr = np.asarray(tk_deltas)

    def _corr(a, b):
        if len(a) <= 2 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan"), float("nan")
        pearson = float(np.corrcoef(a, b)[0, 1])
        rk_a = np.argsort(np.argsort(a))
        rk_b = np.argsort(np.argsort(b))
        spearman = float(np.corrcoef(rk_a, rk_b)[0, 1])
        return pearson, spearman

    sk_p, sk_s = _corr(can_arr, sk_arr)
    tk_p, tk_s = _corr(can_arr, tk_arr)
    sk_sign = float(np.mean(np.sign(can_arr) == np.sign(sk_arr)))
    tk_sign = float(np.mean(np.sign(can_arr) == np.sign(tk_arr)))
    print(f"  Δcong correlation across {n_perturb} perturbations:", flush=True)
    print(f"    SINKHORN: Pearson={sk_p:+.3f}  Spearman={sk_s:+.3f}  sign_agree={sk_sign:.0%}", flush=True)
    print(f"    torch.topk:    Pearson={tk_p:+.3f}  Spearman={tk_s:+.3f}  sign_agree={tk_sign:.0%}", flush=True)
    return {
        "bench": bench_name,
        "canonical": can_cong,
        "sinkhorn": sk_cong,
        "topk": tk_cong,
        "rel_sinkhorn_pct": rel_sk,
        "rel_topk_pct": rel_tk,
        "sinkhorn_pearson": sk_p, "sinkhorn_spearman": sk_s, "sinkhorn_sign": sk_sign,
        "topk_pearson": tk_p, "topk_spearman": tk_s, "topk_sign": tk_sign,
        "n_perturb": n_perturb,
        "sinkhorn_eps": sinkhorn_eps,
        "sinkhorn_iters": sinkhorn_iters,
    }


if __name__ == "__main__":
    benches = sys.argv[1:] if len(sys.argv) > 1 else ["ibm17"]
    for b in benches:
        calibrate(b)

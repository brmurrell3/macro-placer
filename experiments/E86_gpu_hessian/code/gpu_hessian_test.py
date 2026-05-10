"""Validate: GPU smooth-proxy Hessian eigenvalues match CPU within tolerance.

Loads a saved placement (the E48 plateau ≡ E74 starting state), computes
the smallest-algebraic eigvals on CPU and on the default GPU device,
compares.

Wall is also measured for both, to confirm the speedup.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_E74 = _ROOT / "experiments" / "E74_hessian_saddle" / "code"
if str(_E74) not in sys.path:
    sys.path.insert(0, str(_E74))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hessian_saddle import SmoothProxy as CPUSmoothProxy, find_softest_eigenvectors
from gpu_smooth_proxy import GPUSmoothProxy, default_device

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir


def find_softest_eigenvectors_with_proxy(proxy_obj, state, movable_mask, k, log):
    """Generic Lanczos wrapper that takes any SmoothProxy-shaped object."""
    import scipy.sparse.linalg as spla

    n_macros = state.shape[0]
    n_dim = n_macros * 2
    mask_flat = np.zeros(n_dim, dtype=bool)
    for i, m in enumerate(movable_mask):
        if bool(m):
            mask_flat[2 * i] = True
            mask_flat[2 * i + 1] = True
    n_free = int(mask_flat.sum())

    state_dev = state.detach().clone().to(getattr(proxy_obj, "device", "cpu"))

    def hvp(v):
        v_t = v.to(getattr(proxy_obj, "device", "cpu"))
        sr = state_dev.clone().requires_grad_(True)
        _, hv = torch.autograd.functional.hvp(
            proxy_obj.cost, sr, v_t, create_graph=False, strict=False,
        )
        return hv.detach()

    def matvec(v_free):
        v_full = np.zeros(n_dim, dtype=np.float64)
        v_full[mask_flat] = v_free
        v_t = torch.tensor(v_full.reshape(n_macros, 2), dtype=torch.float32)
        out = hvp(v_t).cpu().numpy().reshape(n_dim)[mask_flat]
        return out

    op = spla.LinearOperator(shape=(n_free, n_free), matvec=matvec, dtype=np.float64)
    log(f"  Lanczos on {getattr(proxy_obj, 'device', 'cpu')} (n_free={n_free}, k={k})")
    t0 = time.time()
    eigvals, eigvecs = spla.eigsh(
        op, k=k, which="SA", tol=1e-4, maxiter=500, ncv=min(2 * k + 20, n_free),
    )
    return eigvals, eigvecs, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", default="ibm01", nargs="?")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--device", default=None, help="override default device")
    args = ap.parse_args()

    bench_name = args.bench
    print(f"[E86] loading {bench_name}...")
    benchmark, plc = load_benchmark_from_dir(str(find_benchmark_dir(bench_name)))

    e69 = _ROOT / "experiments" / "E69_sequence_pair_search" / "results" / "placements"
    e25_pt = e69 / f"e25_{bench_name}.pt"
    e41_pt = e69 / f"e41_{bench_name}.pt"
    if not e25_pt.exists() or not e41_pt.exists():
        print(f"[E86] no cached E25/E41 for {bench_name}; need to produce first")
        sys.exit(2)
    e25 = torch.load(e25_pt, weights_only=False)
    e41 = torch.load(e41_pt, weights_only=False)
    state = e25["placement"] if e25["proxy"] <= e41["proxy"] else e41["placement"]
    print(f"[E86] using {'E25' if e25['proxy'] <= e41['proxy'] else 'E41'} plateau, "
          f"proxy={min(e25['proxy'], e41['proxy']):.5f}")

    movable_np = (~benchmark.macro_fixed.cpu().numpy())

    # CPU baseline.
    print(f"[E86] CPU Hessian baseline...")
    cpu_proxy = CPUSmoothProxy(benchmark, plc)
    cpu_eigvals, _, cpu_wall = find_softest_eigenvectors_with_proxy(
        cpu_proxy, state, movable_np, args.k, log=print,
    )
    print(f"[E86] CPU eigvals: {cpu_eigvals}, wall={cpu_wall:.1f}s")

    # GPU run.
    device = args.device or default_device()
    print(f"[E86] GPU Hessian on device={device}...")
    gpu_proxy = GPUSmoothProxy(benchmark, plc, device=device)
    gpu_eigvals, _, gpu_wall = find_softest_eigenvectors_with_proxy(
        gpu_proxy, state, movable_np, args.k, log=print,
    )
    print(f"[E86] GPU eigvals: {gpu_eigvals}, wall={gpu_wall:.1f}s")

    speedup = cpu_wall / gpu_wall if gpu_wall > 0 else float("inf")
    print(f"[E86] speedup: {speedup:.2f}×")

    # Compare.
    eigval_diff = float(np.max(np.abs(np.sort(cpu_eigvals) - np.sort(gpu_eigvals))))
    print(f"[E86] max |Δeigval| = {eigval_diff:.4e}")
    if eigval_diff < 1e-3:
        print(f"[E86] PASS — eigvals match within 1e-3")
    else:
        print(f"[E86] WARN — eigvals differ by {eigval_diff:.4e}; investigate")


if __name__ == "__main__":
    main()

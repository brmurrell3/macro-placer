"""Microbenchmark: GPU vs CPU smooth-proxy call cost.

We want to know: how fast is a single smooth-proxy evaluation on A100 vs CPU?
If CUDA is < 10ms while CPU is > 50ms, GPU CD is worth pursuing.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.cd_core import sdf_init
from gpu_smooth_proxy import GPUSmoothProxy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", default="ibm03")
    ap.add_argument("--n_calls", type=int, default=100)
    args = ap.parse_args()

    bench, plc = load_benchmark_from_dir(str(find_benchmark_dir(args.bench)))
    init = sdf_init(bench)
    print(f"=== Bench: {args.bench}  n_macros={bench.num_macros}  n_nets={bench.num_nets} ===", flush=True)

    for device in ("cpu", "cuda"):
        try:
            smooth = GPUSmoothProxy(bench, plc, device=device)
        except RuntimeError as e:
            print(f"{device}: skipped ({e})", flush=True)
            continue
        init_dev = init.to(smooth.device)
        # Warmup
        _ = float(smooth.cost(init_dev).item())
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.n_calls):
            c = smooth.cost(init_dev)
            v = float(c.item())
        if device == "cuda":
            torch.cuda.synchronize()
        wall = time.time() - t0
        per_call = wall / args.n_calls * 1000
        print(f"  {device}: {args.n_calls} calls in {wall:.2f}s = {per_call:.2f}ms/call", flush=True)


if __name__ == "__main__":
    main()

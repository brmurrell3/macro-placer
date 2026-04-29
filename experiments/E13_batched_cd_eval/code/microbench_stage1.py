"""E13 Stage 1: per-call cost of move/revert vs current_cost on CPU vs MPS.

Question we want answered: when the inner CD sweep evaluates one (col, row)
candidate, what fraction of the time is spent in (a) the Python bookkeeping
inside `move()/revert()` (NOT GPU-able without rewrite) vs (b) the pure-tensor
`current_cost()` (GPU-able today)?

If (a) dominates, batching `current_cost()` on MPS cannot speed up the inner
sweep enough to justify the rewrite. Stage 1 falsifies the hypothesis cheaply.

Run on the M3 Max:

    uv run python experiments/E13_batched_cd_eval/code/microbench_stage1.py

No --hypothesis flag here — this is a probe, not a placement run, so it does
not append to results/experiment_log.jsonl.
"""

from __future__ import annotations

import statistics
import time

import torch

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import sdf_init
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir


BENCHMARK = "ibm01"
N_CALLS = 200
SEED = 0


def median_us(samples_s):
    return statistics.median(samples_s) * 1e6


def time_move_cost_revert(ev, candidates):
    samples = []
    for (idx, xy) in candidates:
        t0 = time.perf_counter()
        ev.move(idx, xy)
        ev.current_cost()
        ev.revert()
        samples.append(time.perf_counter() - t0)
    return samples


def time_current_cost_only(ev):
    samples = []
    for _ in range(N_CALLS):
        t0 = time.perf_counter()
        ev.current_cost()
        samples.append(time.perf_counter() - t0)
    return samples


def move_state_to_device(ev, device):
    """Move every tensor field on the evaluator to `device` in-place.

    Best-effort: anything that is a torch.Tensor gets `.to(device)`. Skips
    Python scalars, dicts, lists. The `_smooth` method allocates fresh
    `torch.arange`/`torch.zeros` with no explicit device, so on MPS those
    intermediates may land on CPU — that is itself a finding.
    """
    for name in dir(ev):
        if name.startswith("__"):
            continue
        try:
            val = getattr(ev, name)
        except AttributeError:
            continue
        if isinstance(val, torch.Tensor):
            try:
                setattr(ev, name, val.to(device))
            except Exception as e:
                print(f"  [warn] could not move {name}: {e}")


def main():
    print(f"# E13 Stage 1 microbench — benchmark={BENCHMARK}, N={N_CALLS}")
    print(f"# torch={torch.__version__} mps={torch.backends.mps.is_available()} "
          f"cuda={torch.cuda.is_available()}")

    bench, plc = load_benchmark_from_dir(str(find_benchmark_dir(BENCHMARK)))
    placement = sdf_init(bench)

    # Build evaluator (always on CPU initially — its __init__ hardcodes .cpu()).
    ev = IncrementalProxyEvaluator(bench, plc, placement)

    # Build a list of (idx, xy) candidate moves — pick movable hard macros and
    # nudge them to a grid-bin center near their current position. We do not
    # care if these are *good* moves; we only care about per-call timing.
    rng = torch.Generator().manual_seed(SEED)
    n_hard = len(bench.hard_macro_indices)
    hard_movable = [
        i for i in range(n_hard)
        if not bool(bench.macro_fixed[bench.hard_macro_indices[i]])
    ]
    if not hard_movable:
        raise RuntimeError("no movable hard macros on this benchmark")
    grid_w = float(plc.width) / int(plc.grid_col)
    grid_h = float(plc.height) / int(plc.grid_row)
    candidates = []
    for k in range(N_CALLS):
        idx = hard_movable[k % len(hard_movable)]
        col = int(torch.randint(0, int(plc.grid_col), (1,), generator=rng))
        row = int(torch.randint(0, int(plc.grid_row), (1,), generator=rng))
        candidates.append((idx, ((col + 0.5) * grid_w, (row + 0.5) * grid_h)))

    # ── (a) CPU: move + current_cost + revert ──────────────────────────────
    samples_a = time_move_cost_revert(ev, candidates)
    med_a = median_us(samples_a)
    print(f"\n(a) CPU move+cost+revert : {med_a:8.1f} us/call (median, N={N_CALLS})")

    # ── (b) CPU: current_cost only ─────────────────────────────────────────
    samples_b = time_current_cost_only(ev)
    med_b = median_us(samples_b)
    print(f"(b) CPU current_cost only: {med_b:8.1f} us/call (median, N={N_CALLS})")

    bookkeeping_us = med_a - med_b
    print(f"    → Python bookkeeping (move+revert): {bookkeeping_us:8.1f} us/call "
          f"({100 * bookkeeping_us / med_a:.0f} %)")
    print(f"    → Pure-tensor cost  (current_cost): {med_b:8.1f} us/call "
          f"({100 * med_b / med_a:.0f} %)")

    # ── (c) MPS: current_cost only ─────────────────────────────────────────
    if not torch.backends.mps.is_available():
        print("\n[skip] MPS unavailable — run on the M3 Max for the GPU number.")
        decide(med_a, med_b, mps_med=None)
        return

    print("\nMoving evaluator state to MPS...")
    move_state_to_device(ev, torch.device("mps"))
    # Warmup (first MPS kernel launch on a fresh process is biased).
    for _ in range(10):
        ev.current_cost()
    samples_c = time_current_cost_only(ev)
    med_c = median_us(samples_c)
    print(f"(c) MPS current_cost only: {med_c:8.1f} us/call (median, N={N_CALLS})")

    decide(med_a, med_b, mps_med=med_c)


def decide(cpu_full_us, cpu_cost_us, mps_med):
    print("\n# Decision (kill gates from manifest)")
    bookkeeping_frac = (cpu_full_us - cpu_cost_us) / cpu_full_us
    print(f"  Python bookkeeping fraction of inner-loop time : {bookkeeping_frac:.0%}")
    # Even with INFINITELY fast batched cost, max speedup of inner sweep is:
    max_speedup = 1.0 / (1.0 - bookkeeping_frac)
    print(f"  Max theoretical inner-sweep speedup (Amdahl)   : {max_speedup:.2f}x")
    if max_speedup < 3.0:
        print("  → KILL: Amdahl ceiling < 3x even with perfect GPU batching.")
        print("    The bookkeeping inside move()/revert() is the bottleneck,")
        print("    not the cost kernel. GPU acceleration cannot deliver.")
        return
    if mps_med is None:
        print("  → INCONCLUSIVE: rerun on M3 Max to get the MPS number.")
        return
    if mps_med >= 2.0 * cpu_full_us:
        print(f"  → KILL: MPS per-call ({mps_med:.0f} us) >= 2x CPU full cycle "
              f"({cpu_full_us:.0f} us). Batching can't amortize that much overhead.")
        return
    print("  → PROCEED to Stage 2: write batched current_cost([K, 2]) on MPS.")


if __name__ == "__main__":
    main()

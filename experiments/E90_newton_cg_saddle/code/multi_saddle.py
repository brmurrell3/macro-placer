"""E90 — multi-direction saddle escape.

Reuses E74's `SmoothProxy` + `find_softest_eigenvectors` (read-only — no edits
to E74 territory). The new thing here: instead of perturbing along ONE
softest eigvec at a time, build combinations `v_combo = Σ_i s_i v_i / ||·||`
for sign vectors s ∈ {-1, 0, +1}^K, and run the same project + polish
pipeline E74 uses.

Hypothesis: at the cascade plateau where multiple eigvals are negative,
joint-direction perturbations reach basins single-direction misses.

Falsification: if no rank-≥2 combination produces a lower polished proxy than
the best rank-1 combination, multi-direction adds nothing → C3 is dead.
"""
from __future__ import annotations

import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_E74 = _ROOT / "experiments" / "E74_hessian_saddle" / "code"
if str(_E74) not in sys.path:
    sys.path.insert(0, str(_E74))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hessian_saddle import SmoothProxy, find_softest_eigenvectors  # type: ignore

from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


@dataclass
class Attempt:
    sign_vec: Tuple[int, ...]
    eps: float
    rank: int
    feasible_after_project: bool
    residual_overlaps: int
    polished_proxy: Optional[float]
    polished_overlap: Optional[int]
    wall_seconds: float


def _normalize_to_2d(
    sign_vec: Sequence[int],
    eigvecs_full_2d: np.ndarray,  # [K, n_macros, 2]
) -> Optional[np.ndarray]:
    """Combine eigvecs by sign vector; return unit-RMS 2D direction or None
    if the combo is zero (all signs zero)."""
    combo = np.zeros_like(eigvecs_full_2d[0])
    nonzero = False
    for s, v in zip(sign_vec, eigvecs_full_2d):
        if s == 0:
            continue
        combo = combo + float(s) * v
        nonzero = True
    if not nonzero:
        return None
    n = np.linalg.norm(combo)
    if n < 1e-12:
        return None
    return combo / n


def _enumerate_sign_vectors(
    K: int,
    max_rank: Optional[int] = None,
    rank_one_signs: Tuple[int, ...] = (1, -1),
) -> List[Tuple[int, ...]]:
    """Enumerate sign vectors. Skip {0..0} and skip sign-flips (s and -s)."""
    out: List[Tuple[int, ...]] = []
    seen = set()
    for s in itertools.product((-1, 0, 1), repeat=K):
        rank = sum(1 for x in s if x != 0)
        if rank == 0:
            continue
        if max_rank is not None and rank > max_rank:
            continue
        # Canonicalize: make the first nonzero sign positive.
        first_nonzero = next(x for x in s if x != 0)
        if first_nonzero < 0:
            s = tuple(-x for x in s)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    out.sort(key=lambda s: (sum(1 for x in s if x != 0), s))
    return out


def multi_saddle_escape(
    plateau_state: torch.Tensor,
    benchmark,
    plc,
    *,
    K: int = 4,
    eps_values: Tuple[float, ...] = (0.3, 1.0, 3.0),
    polish_budget_s: float = 60.0,
    max_rank: Optional[int] = None,
    only_rank_at_least: Optional[int] = None,
    total_budget_s: float = 3600.0,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[torch.Tensor, dict]:
    """From plateau_state, find K softest eigvecs and try all (canonical) sign
    combinations × eps_values. Returns best (lowest-proxy, 0-ovl) result.

    Parameters
    ----------
    K : number of eigvecs to compute
    only_rank_at_least : if set, skip sign-vecs with rank < this — useful to
      isolate the multi-direction contribution from the single-direction
      contribution (set to 2 to test only joint combinations).
    """
    if log is None:
        log = lambda s: print(s, flush=True)
    n_macros = plateau_state.shape[0]
    movable_np = (~benchmark.macro_fixed.cpu().numpy())
    fixed = benchmark.macro_fixed.cpu().numpy()
    movable_idx = [i for i in range(n_macros) if not bool(fixed[i])]

    smooth = SmoothProxy(benchmark, plc)
    log(f"[E90] smooth built; benchmark={benchmark.name} n_macros={n_macros}")

    init_proxy = float(compute_proxy_cost(plateau_state, benchmark, plc)["proxy_cost"])
    init_ovl = compute_overlap_metrics(plateau_state, benchmark)["overlap_count"]
    log(f"[E90] init proxy={init_proxy:.5f} ovl={init_ovl}")
    assert init_ovl == 0, "Expected zero-overlap plateau"

    t_eig = time.time()
    eigvals, eigvecs = find_softest_eigenvectors(
        smooth, plateau_state, movable_np, k=K, log=lambda s: log("  " + s),
    )
    log(f"[E90] eigvecs ready in {time.time() - t_eig:.1f}s; "
        f"eigvals = {[f'{v:.4e}' for v in eigvals]}")
    K_actual = eigvecs.shape[1]
    if K_actual == 0:
        return plateau_state.detach().clone(), {
            "init_proxy": init_proxy,
            "best_proxy": init_proxy,
            "eigvals": [],
            "attempts": [],
            "K_actual": 0,
        }

    # Build full-space 2D eigvecs.
    n_dim = n_macros * 2
    mask_flat = np.zeros(n_dim, dtype=bool)
    for i, m in enumerate(movable_np):
        if bool(m):
            mask_flat[2 * i] = True
            mask_flat[2 * i + 1] = True
    eigvecs_2d = np.zeros((K_actual, n_macros, 2), dtype=np.float64)
    for k in range(K_actual):
        v_full = np.zeros(n_dim, dtype=np.float64)
        v_full[mask_flat] = eigvecs[:, k]
        eigvecs_2d[k] = v_full.reshape(n_macros, 2)

    state_np = plateau_state.detach().cpu().numpy().astype(np.float64)
    hw_np = (benchmark.macro_sizes[:, 0] / 2).cpu().numpy()
    hh_np = (benchmark.macro_sizes[:, 1] / 2).cpu().numpy()
    cw, ch = smooth.cw, smooth.ch
    movable_t = torch.tensor(movable_np)

    sign_vecs = _enumerate_sign_vectors(K_actual, max_rank=max_rank)
    if only_rank_at_least is not None:
        sign_vecs = [s for s in sign_vecs if sum(1 for x in s if x != 0) >= only_rank_at_least]
    log(f"[E90] {len(sign_vecs)} sign vectors × {len(eps_values)} eps "
        f"= {len(sign_vecs) * len(eps_values)} attempts; "
        f"polish_budget={polish_budget_s}s")

    attempts: List[Attempt] = []
    best_state = plateau_state.detach().clone()
    best_proxy = init_proxy

    t_start = time.time()
    for sv in sign_vecs:
        rank = sum(1 for x in sv if x != 0)
        v_combo = _normalize_to_2d(sv, eigvecs_2d)
        if v_combo is None:
            continue
        for eps in eps_values:
            elapsed = time.time() - t_start
            if elapsed > total_budget_s - polish_budget_s - 30:
                log(f"[E90] budget bailout (elapsed={elapsed:.0f}s)")
                break
            t_attempt = time.time()
            pp = state_np + eps * v_combo
            pp[:, 0] = np.clip(pp[:, 0], hw_np, cw - hw_np)
            pp[:, 1] = np.clip(pp[:, 1], hh_np, ch - hh_np)
            cand = torch.tensor(pp, dtype=torch.float32)
            cand[~movable_t] = plateau_state[~movable_t]
            cand, _ = project_overlaps(cand, benchmark)
            ovl = compute_overlap_metrics(cand, benchmark)["overlap_count"]
            if ovl > 0:
                attempts.append(Attempt(
                    sign_vec=sv, eps=eps, rank=rank,
                    feasible_after_project=False, residual_overlaps=int(ovl),
                    polished_proxy=None, polished_overlap=None,
                    wall_seconds=time.time() - t_attempt,
                ))
                log(f"  sv={sv} eps={eps:.1f}: infeasible (resid={ovl})")
                continue
            ev = IncrementalProxyEvaluator(benchmark, plc, cand.clone())
            run_cd_adaptive(
                ev, benchmark, plc, movable_idx,
                min_time_s=min(20.0, polish_budget_s / 3),
                hard_cap_s=polish_budget_s,
                patience=3, plateau_threshold=0.001, log_fn=None,
            )
            polished = ev.placement.detach().clone().to(torch.float32)
            pol_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
            pol_ovl = int(compute_overlap_metrics(polished, benchmark)["overlap_count"])
            wall = time.time() - t_attempt
            attempts.append(Attempt(
                sign_vec=sv, eps=eps, rank=rank,
                feasible_after_project=True, residual_overlaps=0,
                polished_proxy=pol_proxy, polished_overlap=pol_ovl,
                wall_seconds=wall,
            ))
            tag = ""
            if pol_ovl == 0 and pol_proxy < best_proxy - 1e-7:
                best_proxy = pol_proxy
                best_state = polished.detach().clone()
                tag = "  NEW BEST"
            log(f"  rank={rank} sv={sv} eps={eps:.1f}: polished={pol_proxy:.5f} "
                f"ovl={pol_ovl} wall={wall:.0f}s{tag}")
        if (time.time() - t_start) > total_budget_s - 30:
            break

    total_wall = time.time() - t_start
    log(f"[E90] done in {total_wall:.0f}s. init={init_proxy:.5f} -> best={best_proxy:.5f} "
        f"(Δ={best_proxy - init_proxy:+.5f} = {100 * (best_proxy - init_proxy) / init_proxy:+.3f}%)")
    return best_state, {
        "init_proxy": init_proxy,
        "best_proxy": best_proxy,
        "improvement": init_proxy - best_proxy,
        "eigvals": eigvals.tolist() if hasattr(eigvals, "tolist") else list(eigvals),
        "K_actual": K_actual,
        "attempts": [a.__dict__ for a in attempts],
        "wall_seconds": total_wall,
    }

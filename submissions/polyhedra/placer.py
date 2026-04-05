"""
Polyhedra Navigation Placer — orchestration entry point.

Decomposes macro placement into:
1. Discrete topology: pairwise L/R/A/B assignment (which polyhedron)
2. Continuous optimization: LP solve within the chosen polyhedron
3. Navigation: search over neighboring polyhedra for better proxy cost

All algorithmic components are in sibling modules:
  assignment.py  — topology extraction
  lp.py          — LP solver (HPWL + min-displacement)
  surrogate.py   — fast proxy cost estimator
  moves.py       — move types + proposers (plug in new ones here)
  navigator.py   — search loop + acceptance strategies
  projection.py  — constraint projection, overlap detection, legalization
"""

import sys
import time
import pathlib
import numpy as np
import torch

# Add this directory to sys.path so sibling modules can be imported
# (the evaluate harness loads this file directly, not as a package)
_this_dir = str(pathlib.Path(__file__).parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from assignment import extract_assignment
from lp import LPSolver
from surrogate import GridSurrogate
from navigator import Navigator, SimulatedAnnealingAcceptance
from moves import DualGuidedProposer, ClusterProposer
from projection import check_overlaps, legalize, refine_density

from macro_place.benchmark import Benchmark


class PolyhedraNavigationPlacer:
    """
    Polyhedra Navigation placer.

    1. Start from a good initial placement (SDF or greedy shelf packing)
    2. Extract pairwise assignment (defines which polyhedron)
    3. Solve LP within that polyhedron (optimal HPWL + duals)
    4. Navigate to neighboring polyhedra using dual-guided search
    5. Optionally refine density within the best polyhedron
    """

    def __init__(self, navigate: bool = True, refine: bool = False,
                 nav_iters: int = 500, nav_time: float = 50.0,
                 density_steps: int = 80, verbose: bool = True,
                 n_restarts: int = 1):
        self.navigate = navigate
        self.refine = refine
        self.nav_iters = nav_iters
        self.nav_time = nav_time
        self.density_steps = density_steps
        self.verbose = verbose
        self.n_restarts = n_restarts

    def _initial_placement(self, benchmark: Benchmark) -> np.ndarray:
        """Generate initial legal placement.

        Tries SDF placer (best known topology) first, falls back to greedy shelf.
        """
        try:
            import importlib.util
            sdf_path = str(
                pathlib.Path(__file__).parent.parent / "sdf_density" / "placer.py"
            )
            spec = importlib.util.spec_from_file_location("sdf_placer", sdf_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sdf = mod.SDFPlacer()
            placement = sdf.place(benchmark)
            return placement.numpy()
        except Exception as e:
            print(f"  [SDF fallback: {e}]", file=sys.stderr)

        # Fallback: greedy shelf packing
        placement = benchmark.macro_positions.clone()
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_indices = torch.where(movable)[0].tolist()
        sizes = benchmark.macro_sizes
        canvas_w = benchmark.canvas_width
        canvas_h = benchmark.canvas_height

        movable_indices.sort(key=lambda i: -sizes[i, 1].item())

        gap = 0.001
        cursor_x = 0.0
        cursor_y = 0.0
        row_height = 0.0

        for idx in movable_indices:
            w = sizes[idx, 0].item()
            h = sizes[idx, 1].item()

            if cursor_x + w > canvas_w:
                cursor_x = 0.0
                cursor_y += row_height + gap
                row_height = 0.0

            if cursor_y + h > canvas_h:
                placement[idx, 0] = w / 2
                placement[idx, 1] = h / 2
                continue

            placement[idx, 0] = cursor_x + w / 2
            placement[idx, 1] = cursor_y + h / 2

            cursor_x += w + gap
            row_height = max(row_height, h)

        return placement.numpy()

    def _perturb_positions(self, positions, benchmark, rng,
                           fraction=0.15, sigma_frac=0.05):
        """Perturb a fraction of movable macros with Gaussian noise, then legalize."""
        pos = positions.copy()
        sizes = benchmark.macro_sizes.numpy()
        n_hard = benchmark.num_hard_macros
        movable_hard = np.arange(n_hard)[
            ~benchmark.macro_fixed[:n_hard].numpy()
        ]
        n_perturb = max(1, int(len(movable_hard) * fraction))
        chosen = rng.choice(movable_hard, size=n_perturb, replace=False)
        sigma = sigma_frac * min(benchmark.canvas_width, benchmark.canvas_height)

        for idx in chosen:
            pos[idx, 0] += rng.normal(0, sigma)
            pos[idx, 1] += rng.normal(0, sigma)
            hw, hh = sizes[idx, 0] / 2, sizes[idx, 1] / 2
            pos[idx, 0] = np.clip(pos[idx, 0], hw, benchmark.canvas_width - hw)
            pos[idx, 1] = np.clip(pos[idx, 1], hh, benchmark.canvas_height - hh)

        movable_mask = ~benchmark.macro_fixed[:n_hard].numpy()
        pos = legalize(pos, sizes, n_hard, movable_mask,
                       benchmark.canvas_width, benchmark.canvas_height)
        return pos

    def place(self, benchmark: Benchmark, plc=None) -> torch.Tensor:
        t_start = time.time()

        if plc is None:
            from macro_place.loader import load_benchmark_from_dir
            _, plc = load_benchmark_from_dir(
                f"external/MacroPlacement/Testcases/ICCAD04/{benchmark.name}"
            )

        if self.verbose:
            print(f"\n=== Polyhedra Navigation: {benchmark.name} ===")
            print(f"  {benchmark.num_hard_macros} hard macros, "
                  f"{benchmark.num_nets} nets, "
                  f"canvas {benchmark.canvas_width:.1f}x{benchmark.canvas_height:.1f}")

        sizes = benchmark.macro_sizes.numpy()
        hard_indices = np.arange(benchmark.num_hard_macros)
        movable_hard = hard_indices[~benchmark.macro_fixed[:benchmark.num_hard_macros].numpy()]
        rng = np.random.default_rng(42)

        # Step 1: Initial legal placement
        t0 = time.time()
        init_pos = self._initial_placement(benchmark)
        if self.verbose:
            print(f"  Initial placement: {time.time() - t0:.2f}s")

        # Build shared components
        lp = LPSolver(benchmark, plc)
        surrogate = GridSurrogate(benchmark, plc)

        # Determine sparse margin based on problem size
        n_pairs_est = len(movable_hard) * (len(movable_hard) - 1) // 2
        if n_pairs_est > 200000:
            sparse_margin = 1.0
        elif n_pairs_est > 50000:
            sparse_margin = 10.0
        else:
            sparse_margin = 0.0
        lp_cap = 15.0 if n_pairs_est < 100000 else 60.0

        # Multi-start loop
        overall_best_positions = init_pos
        overall_best_proxy = float('inf')

        if n_pairs_est > 100000 or not self.navigate:
            n_restarts = 1
        else:
            n_restarts = self.n_restarts

        first_restart_improvements = 0
        for restart in range(n_restarts):
            restart_start = time.time()
            wall_remaining = max(5, self.nav_time + 10 - (restart_start - t_start))

            if restart > 0 and first_restart_improvements >= 3:
                if self.verbose:
                    print(f"\n  Skipping restart {restart} — "
                          f"restart 0 found {first_restart_improvements} improvements")
                continue

            if restart == 0:
                restart_budget = wall_remaining
            else:
                restarts_left = n_restarts - restart
                restart_budget = wall_remaining / restarts_left

            if self.verbose:
                print(f"\n  --- Restart {restart}/{n_restarts} "
                      f"(budget={restart_budget:.1f}s, wall_remaining={wall_remaining:.1f}s) ---")

            # Prepare starting positions
            if restart == 0:
                start_pos = init_pos
            else:
                start_pos = self._perturb_positions(init_pos, benchmark, rng)
                if check_overlaps(start_pos, sizes, benchmark.num_hard_macros, tol=1e-3):
                    if self.verbose:
                        print(f"  Perturbed restart {restart} has overlaps after legalization, skipping")
                    continue
                if self.verbose:
                    print(f"  Perturbed {int(len(movable_hard) * 0.15)} macros")

            # Extract assignment
            assignment = extract_assignment(start_pos, sizes, movable_hard)
            n_pairs = len(assignment)

            # LP solve
            lp_time_limit = min(lp_cap, max(5.0, restart_budget * 0.5))
            result = lp.solve(assignment, time_limit=lp_time_limit,
                              ref_positions=start_pos, sparse_margin=sparse_margin)

            if result["positions"] is None:
                for retry_margin in [3.0, 1.0, 0.5]:
                    if retry_margin >= sparse_margin:
                        continue
                    result = lp.solve(
                        assignment, time_limit=lp_time_limit,
                        ref_positions=start_pos, sparse_margin=retry_margin
                    )
                    if result["positions"] is not None:
                        break
                if result["positions"] is None:
                    if self.verbose:
                        print(f"  LP infeasible for restart {restart}, skipping")
                    continue

            if self.verbose:
                print(f"  LP solve: HPWL={result['hpwl']:.2f}, {result['solve_time']:.2f}s")

            # Navigate
            if self.navigate and n_pairs > 0:
                nav = Navigator(
                    lp_solver=lp,
                    benchmark=benchmark,
                    plc=plc,
                    surrogate=surrogate,
                    move_proposers=[
                        DualGuidedProposer(alpha=1.0, top_k=50),
                        ClusterProposer(n_proposals=20),
                    ],
                    acceptance=SimulatedAnnealingAcceptance(rng=rng),
                )

                nav_budget = max(5, restart_budget - (time.time() - restart_start))

                nav_result = nav.navigate(
                    assignment, result,
                    ref_positions=start_pos,
                    max_iters=self.nav_iters,
                    time_budget=nav_budget,
                    top_k_verify=1,
                    verbose=self.verbose,
                    lp_resolve_cap=3,
                )

                positions = nav_result["positions"]
                if restart == 0:
                    first_restart_improvements = nav_result["improvements"]
                if self.verbose:
                    print(f"  Restart {restart}: {nav_result['improvements']} improvements")
            else:
                positions = start_pos

            # Verify no overlaps
            if check_overlaps(positions, sizes, benchmark.num_hard_macros, tol=1e-3):
                if self.verbose:
                    print(f"  Restart {restart} has overlaps, skipping")
                continue

            # Evaluate with surrogate to compare restarts
            surrogate.init_from_placement(positions)
            surr_proxy = surrogate.get_proxy_cost()

            if surr_proxy < overall_best_proxy:
                overall_best_proxy = surr_proxy
                overall_best_positions = positions.copy()
                if self.verbose:
                    print(f"  New overall best: surr_proxy={surr_proxy:.4f} (restart {restart})")

        if self.verbose:
            print(f"\n  Total time: {time.time() - t_start:.1f}s, "
                  f"best surr_proxy={overall_best_proxy:.4f}")

        return torch.tensor(overall_best_positions, dtype=torch.float32)

"""Xplace-init + cascade saddle Lévy polish.

Pipeline:
  1. Run Xplace as basin generator (subprocess to ~/Xplace) — gets GP-only output
  2. Parse Xplace .pl, convert to torch positions
  3. project_overlaps to remove Xplace residuals
  4. CD-adaptive legalize (E25 CD only, no LNS/SA)
  5. Cascade saddle escape (E84) for final polish

Speculative experiment: Carrotato uses Xplace + Triton + polish (0.9671).
We can't replicate Triton kernels but our Lévy saddle is a powerful polish.
If Xplace basin combines with our polish productively, this is a new
basin lane to add to the ensemble.

Falls back gracefully to cd_lns_sa_cascade_levy behavior if Xplace not
available on the host.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

# E97 Lévy saddle
_E97_CODE = _ROOT / "experiments" / "E97_levy_saddle" / "code"
if str(_E97_CODE) not in sys.path:
    sys.path.insert(0, str(_E97_CODE))
from levy_saddle import levy_saddle_escape

# E25 fallback (when Xplace not available)
_E25_PATH = _ROOT / "submissions" / "cd_lns_sa" / "placer.py"
_E25_SPEC = importlib.util.spec_from_file_location("e25_placer", str(_E25_PATH))
_E25_MOD = importlib.util.module_from_spec(_E25_SPEC)
_E25_SPEC.loader.exec_module(_E25_MOD)
CDLNSSAPlacer = _E25_MOD.CDLNSSAPlacer


def _run_xplace(benchmark: Benchmark, plc, log) -> Optional[torch.Tensor]:
    """Run Xplace subprocess; return GP placement tensor or None on failure."""
    xplace_home = os.environ.get("XPLACE_HOME", "/home/ubuntu/Xplace")
    if not Path(xplace_home).exists():
        log("[xplace] XPLACE_HOME not set / dir missing; skipping")
        return None
    xplace_main = Path(xplace_home) / "main.py"
    if not xplace_main.exists():
        log("[xplace] Xplace main.py missing")
        return None

    venv_py = os.environ.get("XPLACE_PYTHON",
                              "/home/ubuntu/macro-place-challenge-2026/.venv/bin/python")
    if not Path(venv_py).exists():
        venv_py = sys.executable
    log(f"[xplace] using python {venv_py}")

    SCALE = 100
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes.cpu().numpy()
    pos = benchmark.macro_positions.cpu().numpy()
    fixed = benchmark.macro_fixed.cpu().numpy()
    ports = benchmark.port_positions.cpu().numpy() if hasattr(benchmark, 'port_positions') else None
    N = benchmark.num_macros
    P = ports.shape[0] if ports is not None else 0

    with tempfile.TemporaryDirectory(prefix=f"xplace_{benchmark.name}_") as tmp:
        tmp = Path(tmp)
        # Write bookshelf: nodes, pl, scl, nets, wts, aux
        with open(tmp / f"{benchmark.name}.nodes", "w") as f:
            f.write("UCLA nodes 1.0\n\n")
            f.write(f"NumNodes : {N + P}\n")
            f.write(f"NumTerminals : {int(fixed.sum()) + P}\n")
            for i in range(N):
                w = max(1, int(round(sizes[i, 0] * SCALE)))
                h = max(1, int(round(sizes[i, 1] * SCALE)))
                term = " terminal" if bool(fixed[i]) else ""
                f.write(f"  n{i}\t{w}\t{h}{term}\n")
            for k in range(P):
                f.write(f"  n{N + k}\t1\t1 terminal\n")
        with open(tmp / f"{benchmark.name}.pl", "w") as f:
            f.write("UCLA pl 1.0\n\n")
            for i in range(N):
                wh = sizes[i, 0] / 2.0
                hh = sizes[i, 1] / 2.0
                x_ll = int(round((pos[i, 0] - wh) * SCALE))
                y_ll = int(round((pos[i, 1] - hh) * SCALE))
                moved = " /FIXED" if bool(fixed[i]) else ""
                f.write(f"  n{i}\t{x_ll}\t{y_ll} : N{moved}\n")
            if ports is not None:
                for k in range(P):
                    x_ll = int(round(ports[k, 0] * SCALE))
                    y_ll = int(round(ports[k, 1] * SCALE))
                    f.write(f"  n{N + k}\t{x_ll}\t{y_ll} : N /FIXED\n")
        row_h = SCALE; site_w = SCALE
        cw_int = int(round(cw * SCALE)); ch_int = int(round(ch * SCALE))
        num_rows = max(1, ch_int // row_h); num_sites = max(1, cw_int // site_w)
        with open(tmp / f"{benchmark.name}.scl", "w") as f:
            f.write("UCLA scl 1.0\n\n")
            f.write(f"NumRows : {num_rows}\n\n")
            for r in range(num_rows):
                y = r * row_h
                f.write(f"CoreRow Horizontal\n  Coordinate    :  {y}\n  Height        :  {row_h}\n  Sitewidth     :  {site_w}\n  Sitespacing   :  {site_w}\n  Siteorient    :  N\n  Sitesymmetry  :  Y\n  SubrowOrigin  :  0\tNumSites :  {num_sites}\nEnd\n")
        # Nets via E76 helper
        e76 = _ROOT / "experiments" / "E76_dreamplace_integration" / "code"
        if str(e76) not in sys.path:
            sys.path.insert(0, str(e76))
        import tilos_to_bookshelf as bk
        bk._write_nets(benchmark, tmp)
        bk._write_wts(benchmark, tmp)
        with open(tmp / f"{benchmark.name}.aux", "w") as f:
            f.write(f"RowBasedPlacement : {benchmark.name}.nodes  {benchmark.name}.nets  {benchmark.name}.wts  {benchmark.name}.pl  {benchmark.name}.scl\n")

        cmd = [
            venv_py, str(xplace_main),
            "--custom_path", f"aux:{tmp}/{benchmark.name}.aux,design_name:{benchmark.name},benchmark:custom",
            "--load_from_raw", "True",
            "--detail_placement", "False",
            "--legalization", "False",
            "--write_global_placement", "True",
            "--deterministic", "False",
            "--result_dir", str(tmp / "result"),
        ]
        log(f"[xplace] running on {benchmark.name}...")
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=xplace_home)
        except subprocess.TimeoutExpired:
            log("[xplace] timed out 600s")
            return None
        if proc.returncode != 0:
            log(f"[xplace] returncode={proc.returncode}; stderr tail: {proc.stderr[-200:]}")
            return None
        log(f"[xplace] done in {time.time() - t0:.0f}s")

        # Find the output .pl file
        out_dir = tmp / "result"
        pl_files = list(out_dir.glob(f"*/output/placement_{benchmark.name}_gp.pl"))
        if not pl_files:
            log(f"[xplace] no output .pl found at {out_dir}")
            return None
        out_pl = pl_files[0]
        log(f"[xplace] parsing {out_pl}")
        # Xplace output is in μm directly
        placement = benchmark.macro_positions.clone().to(torch.float32)
        with open(out_pl) as f:
            for line in f:
                line = line.strip()
                if line.startswith("n") and ":" in line:
                    parts = line.split()
                    nid = int(parts[0][1:])
                    if nid < N:
                        x_ll = float(parts[1]); y_ll = float(parts[2])
                        w = float(sizes[nid, 0]); h = float(sizes[nid, 1])
                        placement[nid, 0] = x_ll + w / 2
                        placement[nid, 1] = y_ll + h / 2
        return placement


class CDLNSSACascadeXplaceLevyPlacer:
    """Xplace basin + cascade saddle Lévy polish."""

    def __init__(
        self,
        max_iters: int = 5,
        K_eps: int = 3,
        eps_scale: float = 1.0,
        polish_budget: float = 180.0,
        budget_seconds: Optional[float] = 3300.0,
        rng_seed: int = 42,
        verbose: bool = True,
    ):
        self.max_iters = max_iters
        self.K_eps = K_eps
        self.eps_scale = eps_scale
        self.polish_budget = polish_budget
        self.budget_seconds = budget_seconds
        self.rng_seed = rng_seed
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True) if self.verbose else None
        log(f"=== CDLNSSACascadeXplaceLevyPlacer ({benchmark.name}) ===")
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # Phase 1: Xplace basin
        xplace_pos = _run_xplace(benchmark, plc, log)
        xplace_proxy = float("inf")
        if xplace_pos is not None:
            xplace_pos, _ = project_overlaps(xplace_pos, benchmark)
            xplace_ovl = int(compute_overlap_metrics(xplace_pos, benchmark)["overlap_count"])
            xplace_proxy = float(compute_proxy_cost(xplace_pos, benchmark, plc)["proxy_cost"])
            log(f"[xplace] basin proxy={xplace_proxy:.5f} ovl={xplace_ovl}")

        # Phase 2: legalize + CD polish on Xplace basin (using E25's CD)
        legalized = None
        legalized_proxy = float("inf")
        if xplace_pos is not None and int(compute_overlap_metrics(xplace_pos, benchmark)["overlap_count"]) == 0:
            fixed = benchmark.macro_fixed.cpu().numpy()
            movable = [i for i in range(benchmark.num_macros) if not bool(fixed[i])]
            ev = IncrementalProxyEvaluator(benchmark, plc, xplace_pos.clone().to(torch.float64))
            log("[xplace-polish] CD on Xplace basin")
            run_cd_adaptive(
                evaluator=ev, benchmark=benchmark, plc=plc, movable=movable,
                min_time_s=60.0, hard_cap_s=600.0,
                patience=3, plateau_threshold=0.001, log_fn=None,
            )
            legalized = ev.placement.detach().clone().to(torch.float32)
            legalized, _ = project_overlaps(legalized, benchmark)
            legalized_proxy = float(compute_proxy_cost(legalized, benchmark, plc)["proxy_cost"])
            log(f"[xplace-polish] post-CD proxy={legalized_proxy:.5f}")

        # Phase 3: E25 fallback (always run, in case Xplace fails or its basin is bad)
        log("[fallback] E25 SDF+CD+LNS+SA")
        e25 = CDLNSSAPlacer(cd_hard_cap_s=660.0, lns_budget_s=200.0, sa_budget_s=200.0).place(benchmark)
        e25_proxy = float(compute_proxy_cost(e25, benchmark, plc)["proxy_cost"])
        log(f"[fallback] E25 proxy={e25_proxy:.5f}")

        # Phase 4: pick best plateau
        candidates = [(e25_proxy, e25, "E25")]
        if legalized is not None:
            candidates.append((legalized_proxy, legalized, "xplace_cd"))
        candidates.sort(key=lambda c: c[0])
        plateau_proxy, plateau, plateau_label = candidates[0]
        log(f"[plateau] pick: {plateau_label} ({plateau_proxy:.5f})")

        # Phase 5: cascade saddle
        cascade_state, cascade_proxy = plateau, plateau_proxy
        remaining = deadline - time.time() if deadline else 3600.0
        if remaining > 60.0:
            cascade_budget = remaining - 10.0 if deadline else 3600.0
            log(f"[saddle] Lévy cascade budget={cascade_budget:.0f}s")
            try:
                cascade_state, stats = levy_saddle_escape(
                    plateau, benchmark, plc,
                    max_iters=self.max_iters, K_eps=self.K_eps, eps_scale=self.eps_scale,
                    polish_budget=self.polish_budget, total_budget_s=cascade_budget,
                    rng_seed=self.rng_seed, log=lambda s: None,
                )
                cascade_proxy = float(compute_proxy_cost(cascade_state, benchmark, plc)["proxy_cost"])
                log(f"[saddle] done proxy={cascade_proxy:.5f}")
            except Exception as exc:
                log(f"[saddle] failed: {exc}")

        # Phase 6: best of all
        all_candidates = [(e25_proxy, e25, "E25"), (cascade_proxy, cascade_state, "xplace_levy_cascade")]
        if legalized is not None:
            all_candidates.append((legalized_proxy, legalized, "xplace_cd"))
        all_candidates.sort(key=lambda c: c[0])
        for proxy_i, plac_i, name_i in all_candidates:
            ovl_i = compute_overlap_metrics(plac_i, benchmark)["overlap_count"]
            if int(ovl_i) == 0:
                log(f"WINNER: {name_i} proxy={proxy_i:.5f} (wall={time.time()-t0:.0f}s)")
                return plac_i
        raise RuntimeError("xplace_levy pipeline produced no overlap-free candidate")

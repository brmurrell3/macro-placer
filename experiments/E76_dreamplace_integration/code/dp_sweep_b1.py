"""PATH B1: DREAMPlace 5D hyperparameter sweep.

For each bench, run K random-search configs over:
  target_density: [0.5, 0.65, 0.75, 0.85, 0.9, 0.95]
  density_weight: [1e-5, 4e-5, 8e-5, 2e-4, 5e-4, 1e-3]
  iteration:      [1000, 1500, 2000, 2500, 3000]
  learning_rate:  [0.005, 0.01, 0.02, 0.05]
  num_bins:       [512, 1024, 2048]

Each config: run DP → greedy_legalize → measure proxy (no CD polish at sweep
time; cheap evaluation). Save (config, proxy, overlap_count, dp_wall) per
bench. Best-of-K per bench is then a candidate basin for path-B integration.

Decision gate (after sweep): if ≥5/17 IBM benches show best-of-K DP basin
proxy ≤ E25/E41 plateau (~0.89-1.10 range), B is viable. Else kill B.
"""
import os, sys, time, json, tempfile, subprocess, random, hashlib
os.environ["DREAMPLACE_ROOT"] = "/opt/DREAMPlace/install"
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

import torch
from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.cd_core import project_overlaps
from macro_place.objective import compute_proxy_cost, compute_overlap_metrics
import tilos_to_bookshelf as bk_writer
import bookshelf_to_pt as bk_reader
from macro_legalizer import greedy_macro_legalize


SWEEP_GRID = {
    "target_density": [0.5, 0.65, 0.75, 0.85, 0.9, 0.95],
    "density_weight": [1e-5, 4e-5, 8e-5, 2e-4, 5e-4, 1e-3],
    "iteration":      [1000, 1500, 2000, 2500, 3000],
    "learning_rate":  [0.005, 0.01, 0.02, 0.05],
    "num_bins":       [512, 1024, 2048],
}


def sample_config(rng):
    return {k: rng.choice(v) for k, v in SWEEP_GRID.items()}


def cfg_hash(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def run_dp(bench, plc, config, tmp_root="/tmp/dp_sweep"):
    dp_root = "/opt/DREAMPlace/install"
    placer_py = f"{dp_root}/dreamplace/Placer.py"
    SCALE = float(bk_writer.SCALE)
    Path(tmp_root).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"dp_{bench.name}_", dir=tmp_root) as tmp:
        tmp = Path(tmp)
        bk_writer._write_nodes(bench, tmp); bk_writer._write_pl(bench, tmp)
        bk_writer._write_nets(bench, tmp); bk_writer._write_scl(bench, tmp)
        bk_writer._write_wts(bench, tmp); bk_writer._write_aux(bench, tmp)
        nbins = config["num_bins"]
        cfg_path = tmp / "dp.json"
        cfg_path.write_text(json.dumps({
            "aux_input": str(tmp / (bench.name + ".aux")),
            "target_density": config["target_density"],
            "density_weight": config["density_weight"],
            "gpu": 1,
            "global_place_stages": [{
                "num_bins_x": nbins, "num_bins_y": nbins,
                "iteration": config["iteration"],
                "learning_rate": config["learning_rate"],
                "wirelength": "weighted_average", "optimizer": "nesterov",
            }],
            "legalize_flag": 0, "detailed_place_flag": 0,
            "stop_overflow": 0.07, "result_dir": str(tmp),
        }))
        t0 = time.time()
        proc = subprocess.run(["/usr/bin/python3", placer_py, str(cfg_path)],
            cwd=dp_root, capture_output=True, text=True, timeout=300)
        wall = time.time() - t0
        if proc.returncode != 0:
            return None, wall, "dp_failed"
        gp_pl = tmp / f"{bench.name}.gp.pl"
        if not gp_pl.exists():
            for c in tmp.rglob("*.gp.pl"):
                gp_pl = c; break
            if not gp_pl.exists():
                return None, wall, "no_output"
        pl_map = bk_reader.parse_pl(gp_pl)
        n = bench.num_macros
        sizes = bench.macro_sizes.cpu().numpy()
        placement = bench.macro_positions.clone().detach()
        for i in range(n):
            if i in pl_map:
                llx, lly = pl_map[i]
                placement[i, 0] = llx / SCALE + float(sizes[i, 0]) / 2.0
                placement[i, 1] = lly / SCALE + float(sizes[i, 1]) / 2.0
        return placement.to(torch.float32), wall, "ok"


def sweep_bench(bench_name, K=50, seed=42, out_path=None):
    bench, plc = load_benchmark_from_dir(str(find_benchmark_dir(bench_name)))
    rng = random.Random(seed)
    print(f"\n=== {bench_name} (n_macros={bench.num_macros}, K={K}) ===", flush=True)
    results = []
    best_proxy = float("inf")
    best_cfg = None
    seen = set()
    t_total = time.time()
    for i in range(K):
        cfg = sample_config(rng)
        h = cfg_hash(cfg)
        if h in seen:
            continue  # duplicate config from random search
        seen.add(h)
        t0 = time.time()
        out = run_dp(bench, plc, cfg)
        if out[0] is None:
            placement, dp_wall, status = None, out[1], out[2]
            print(f"  [{i+1:2d}/{K}] cfg={h} status={status} dp_wall={dp_wall:.0f}s", flush=True)
            results.append({"cfg": cfg, "cfg_hash": h, "status": status, "dp_wall": dp_wall})
            continue
        placement, dp_wall, _ = out
        # greedy_legalize
        legal, _ = greedy_macro_legalize(placement, bench, step_size_frac=0.01)
        legal, _ = project_overlaps(legal, bench)
        ovl = compute_overlap_metrics(legal, bench)["overlap_count"]
        proxy = float(compute_proxy_cost(legal, bench, plc)["proxy_cost"])
        is_best = proxy < best_proxy and ovl == 0
        if is_best:
            best_proxy = proxy; best_cfg = cfg
        results.append({
            "cfg": cfg, "cfg_hash": h, "status": "ok",
            "dp_wall": dp_wall, "ovl": ovl, "proxy": proxy,
            "is_best_so_far": is_best,
        })
        mark = " *NEW BEST*" if is_best else ""
        print(f"  [{i+1:2d}/{K}] cfg={h} td={cfg['target_density']} dw={cfg['density_weight']:.1e} "
              f"it={cfg['iteration']} lr={cfg['learning_rate']:.3f} bins={cfg['num_bins']} "
              f"→ proxy={proxy:.5f} ovl={ovl} wall={dp_wall:.0f}s{mark}", flush=True)
    total_wall = time.time() - t_total
    summary = {
        "bench": bench_name, "K_attempted": K, "K_unique": len(seen),
        "best_proxy": best_proxy if best_proxy < float("inf") else None,
        "best_cfg": best_cfg, "total_wall": total_wall, "results": results,
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(summary, indent=2))
    print(f"\n=== {bench_name} sweep done: best_proxy={best_proxy:.5f} "
          f"(in {len(seen)} unique cfgs, {total_wall:.0f}s) ===", flush=True)
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"/tmp/dp_sweep_{args.bench}.json"
    sweep_bench(args.bench, K=args.K, seed=args.seed, out_path=out)

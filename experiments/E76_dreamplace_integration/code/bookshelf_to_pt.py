"""Convert DREAMPlace .pl output (Bookshelf placement) back into our format.

After running DREAMPlace on cloud, you'll have a <bench>.gp.pl (global
placement) and possibly <bench>.lg.pl (legalized). Parse the .pl line by
line; each line maps node name → (llx, lly) lower-left coords. Convert
back to centers + 1:1 align with the macro indices used by our placer.

Output: torch.Tensor[num_macros, 2] saved as a .pt file ready for use as
a starting plateau in E74 / E84 / E85.

Usage:
  uv run python experiments/E76_dreamplace_integration/code/bookshelf_to_pt.py \
      <bench> <dreamplace_pl_path> <out_pt_path>
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


_NODE_RE = re.compile(r"^\s*n(\d+)\s+([\-0-9.eE]+)\s+([\-0-9.eE]+)\s+:")


def parse_pl(pl_path: Path) -> dict:
    """Return dict {node_idx: (llx, lly)} from a Bookshelf .pl file."""
    out = {}
    with open(pl_path) as f:
        for line in f:
            m = _NODE_RE.match(line)
            if not m:
                continue
            idx = int(m.group(1))
            llx = float(m.group(2))
            lly = float(m.group(3))
            out[idx] = (llx, lly)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("pl_path")
    ap.add_argument("out_pt")
    ap.add_argument("--no-legalize", action="store_true",
                    help="skip project_overlaps; useful if DREAMPlace output is already legal")
    args = ap.parse_args()

    bench_name = args.bench
    pl_path = Path(args.pl_path)
    out_pt = Path(args.out_pt)

    print(f"[bs2pt] loading {bench_name} benchmark...")
    benchmark, plc = load_benchmark_from_dir(str(find_benchmark_dir(bench_name)))

    print(f"[bs2pt] parsing {pl_path}...")
    pl = parse_pl(pl_path)
    n = benchmark.num_macros
    sizes = benchmark.macro_sizes.cpu().numpy()
    placement = benchmark.macro_positions.clone().detach()  # start from default; overwrite with DREAMPlace output
    missing = []
    for i in range(n):
        if i not in pl:
            missing.append(i)
            continue
        llx, lly = pl[i]
        cx = llx + float(sizes[i, 0]) / 2.0
        cy = lly + float(sizes[i, 1]) / 2.0
        placement[i, 0] = cx
        placement[i, 1] = cy
    if missing:
        print(f"[bs2pt] WARN: {len(missing)} nodes not in .pl, kept at default "
              f"(first few: {missing[:5]})")

    placement = placement.to(torch.float32)

    if not args.no_legalize:
        print(f"[bs2pt] legalizing via project_overlaps...")
        placement, n_iters = project_overlaps(placement, benchmark)
        ovl = compute_overlap_metrics(placement, benchmark)["overlap_count"]
        print(f"[bs2pt] project_overlaps: {n_iters} iters, residual_overlap={ovl}")
        if ovl > 0:
            print(f"[bs2pt] WARN: DREAMPlace output not legalizable in 50 iters — {ovl} residuals")

    proxy = float(compute_proxy_cost(placement, benchmark, plc)["proxy_cost"])
    ovl = compute_overlap_metrics(placement, benchmark)["overlap_count"]
    print(f"[bs2pt] proxy={proxy:.5f}, overlap_count={ovl}")

    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "placement": placement.cpu(),
        "proxy": proxy,
        "overlap_count": ovl,
        "bench_name": bench_name,
        "macro_sizes": benchmark.macro_sizes.cpu(),
        "num_hard_macros": int(benchmark.num_hard_macros),
        "canvas_width": float(benchmark.canvas_width),
        "canvas_height": float(benchmark.canvas_height),
        "source": "dreamplace",
    }, out_pt)
    print(f"[bs2pt] saved -> {out_pt}")


if __name__ == "__main__":
    main()

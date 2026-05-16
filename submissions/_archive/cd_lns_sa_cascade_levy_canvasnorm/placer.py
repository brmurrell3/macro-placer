import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
"""Lévy variant with canvas-normalized eps_scale.

Sets eps_scale = α × canvas_width / √n_macros at place() time, instead of the
fixed eps_scale=1.0 default. Auto-scales jump magnitudes to typical macro
spacing on each bench, no per-bench tuning.

α=2.0 chosen so that:
  - ibm01 (canvas=22.9, n=1140): eps_scale = 2 × 22.9/√1140 = 1.36
  - ibm10 (canvas=~55, n=2280):  eps_scale = 2 × 55/√2280   = 2.30
  - ibm17 (canvas=larger):        eps_scale scales accordingly
The reference Cauchy median for eps=1.0 is one canvas-unit; α=2.0 makes the
typical jump ~ 2 average-macro-spacings, which empirically corresponds to
"crossing a few neighbor pad rows."

No bench-name branches; only canvas_width and num_macros (both observable
inputs).
"""
import math
from submissions.cd_lns_sa_cascade_levy.placer import CDLNSSACascadeLevyPlacer

class CDLNSSACascadeLevyCanvasNormPlacer(CDLNSSACascadeLevyPlacer):
    def __init__(self):
        # eps_scale set per-bench in place(); placeholder default
        super().__init__(eps_scale=1.0, polish_budget=60.0, max_iters=6)
        self.alpha = 2.0  # canvas-relative scale factor

    def place(self, benchmark):
        canvas = float(benchmark.canvas_width)
        n_macros = benchmark.num_macros
        self.eps_scale = self.alpha * canvas / math.sqrt(max(1, n_macros))
        # No print here — verbose log inside super().place() will show it via the K=,σ= field.
        return super().place(benchmark)

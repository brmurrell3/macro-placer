"""The CUDA fine step of ../fine/step.py:CudaFine with ONE launch replaced by a
kernel in the Triton kernel language (kernels.py). Everything else of the step is the CUDA port, unchanged.

  the density factors (site k_dfac3; twin: dfac3 of ../fine/fine.cu)
      TritonDfacFlat       the twin's layout, one element per (lane, macro, grid-side element); the twin's arithmetic
      TritonDfacFlatTl     the same with tl.erf, tl.exp and the operator /
      TritonDfacTile       a tile [macros, grid-side elements] per program; the twin's arithmetic
  the wirelength per net (site k_wl_net; twin: wl_net of ../fine/fine.cu)
      TritonWlLoop         a run-time loop per tile of nets, to the tile's largest pin count, with masks
      TritonWlLoopTl       the same with tl.exp, tl.log and the operator /
      TritonWlLoopOrdered  the same loop, the nets taken through a list sorted by pin count
      TritonWlPad          every net padded to the largest pin count (a power of two), masks, reductions along the pins
      TritonWlBucket       the padded kernel once per bucket of pin counts (2, 4, 8, 16, ...): several launches
  both
      TritonBoth           TritonDfacTile's and TritonWlLoop's launches in one step

A Triton kernel is launched on the torch CUDA tensors of the step (their device pointers; no copies), on torch's
current stream, and keeps nothing between calls. The lists the wirelength forms need beside the twin's (the nets
sorted by pin count, the buckets) are made once in the constructor from the twin's own net list.

Launch options are class attributes: LIBM and IEEE (kernels.py: both True is the arithmetic of the CUDA twin as far
as the language has it; both False is what a Triton kernel gets when nothing is asked for), FUSION (enable_fp_fusion:
contraction of a multiply and an add; True is the default of the installed Triton 3.9.0 and of the CUDA twin's
compiler), WL_PIN (which product of the pin gradient is fused, see kernels.py: 2 is the CUDA twin's compiler's choice,
0 leaves it to Triton's), the tile sizes and num_warps.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / "fine"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
import step as base  # noqa: E402  (../fine/step.py: the CUDA fine step)
import kernels as tk  # noqa: E402


def _pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


class TritonDfacFlat(base.CudaFine):
    name = "triton-dfac3-flat"
    REPLACES = "dfac3"
    LIBM, IEEE, FUSION = True, True, True
    DF_FORM, DF_BLOCK, DF_WARPS, DF_BM = "flat", 256, 8, 2      # flat: one element per thread, 256 threads per program

    def k_dfac3(self, B, s, b):
        N, R, C = self.N, self.R, self.C
        fp = (float(s.sigma_scale), float(s.sigma_floor_x), float(s.sigma_floor_y))
        if self.DF_FORM == "flat":
            total, block = b * N * (R + C), self.DF_BLOCK
            tk.dfac3_flat[((total + block - 1) // block,)](B.fx, B.dx, B.fyt, B.dy, B.fy, B.c, self.half, self.chq, *self.dens, N, R, C, total, *fp,
                                                           BLOCK=block, LIBM=self.LIBM, IEEE=self.IEEE, num_warps=self.DF_WARPS, enable_fp_fusion=self.FUSION)
        else:
            bm = self.DF_BM
            tk.dfac3_tile[((N + bm - 1) // bm, b)](B.fx, B.dx, B.fyt, B.dy, B.fy, B.c, self.half, self.chq, *self.dens, N, R, C, *fp,
                                                   BM=bm, EP=_pow2(max(R, C)), LIBM=self.LIBM, IEEE=self.IEEE, num_warps=self.DF_WARPS, enable_fp_fusion=self.FUSION)


class TritonDfacFlatTl(TritonDfacFlat):
    name = "triton-dfac3-flat-tl"
    LIBM, IEEE = False, False


class TritonDfacTile(TritonDfacFlat):
    name = "triton-dfac3-tile"
    DF_FORM, DF_BM, DF_WARPS = "tile", 2, 1                      # a tile of 2 macros by EP grid-side elements on one warp


class TritonWlLoop(base.CudaFine):
    name = "triton-wl-loop"
    REPLACES = "wl_net"
    LIBM, IEEE, FUSION = True, True, True
    WL_FORM, WL_BN, WL_WARPS, WL_PIN = "loop", 32, 1, 2          # one net per thread, one warp per program
    WL_PAD_ELEMENTS = 256                                        # pin slots per program of the padded kernel (nets = this / DP)

    def __init__(self, static, lanes, device):
        super().__init__(static, lanes, device)
        degree = np.diff(self.net_start.cpu().numpy().astype(np.int64))
        order = np.argsort(degree, kind="stable")
        self.order = torch.from_numpy(order.astype(np.int32)).to(self.device)
        self.DPMAX = max(2, _pow2(int(degree.max())))
        # buckets of the sorted list: pin counts in (DP / 2, DP], and 1 or 2 pins in the first
        self.buckets, sorted_degree, lo, dp = [], degree[order], 0, 2
        while lo < self.NN:
            hi = int(np.searchsorted(sorted_degree, dp, side="right"))
            if hi > lo:
                self.buckets.append((lo, hi - lo, dp))
            lo, dp = hi, 2 * dp

    def _pad_bn(self, dp):
        return max(1, self.WL_PAD_ELEMENTS // dp)

    def k_wl_net(self, B, s, b):
        seg, g = self.seg["wirelength"][0], float(s.gamma)
        lists = (B.pgw, B.val, B.c, self.net_start, self.pin_macro, self.pin_off, self.net_w, self.order)
        opts = {"LIBM": self.LIBM, "IEEE": self.IEEE, "PIN": self.WL_PIN, "num_warps": self.WL_WARPS, "enable_fp_fusion": self.FUSION}
        if self.WL_FORM in ("loop", "loop-ordered"):
            bn = self.WL_BN
            tk.wl_loop[((self.NN + bn - 1) // bn, b)](*lists, self.N, self.NN, self.P, self.VW, seg, g, BN=bn,
                                                      ORDERED=self.WL_FORM == "loop-ordered", **opts)
        elif self.WL_FORM == "pad":
            bn = self._pad_bn(self.DPMAX)
            tk.wl_pad[((self.NN + bn - 1) // bn, b)](*lists, self.N, self.P, self.VW, seg, 0, self.NN, g, BN=bn, DP=self.DPMAX, **opts)
        else:
            for first, count, dp in self.buckets:
                bn = self._pad_bn(dp)
                tk.wl_pad[((count + bn - 1) // bn, b)](*lists, self.N, self.P, self.VW, seg, first, count, g, BN=bn, DP=dp, **opts)


class TritonWlLoopTl(TritonWlLoop):
    name = "triton-wl-loop-tl"
    LIBM, IEEE, WL_PIN = False, False, 0


class TritonWlLoopOrdered(TritonWlLoop):
    name = "triton-wl-loop-ordered"
    WL_FORM = "loop-ordered"


class TritonWlPad(TritonWlLoop):
    name = "triton-wl-pad"
    WL_FORM, WL_PAD_ELEMENTS, WL_WARPS = "pad", 256, 4


class TritonWlBucket(TritonWlLoop):
    name = "triton-wl-bucket"
    WL_FORM, WL_PAD_ELEMENTS, WL_WARPS = "bucket", 64, 1


class TritonBoth(TritonDfacTile, TritonWlLoop):
    """Both Triton kernels in one step: the tile form of the density factors and the loop form of the wirelength."""
    name = "triton-both"
    REPLACES = "dfac3+wl_net"

"""The whole coarse step as CUDA kernels (float32): loss and gradient (entry 1), and the step with Adam (entry 2).

A port of the Metal coarse step (../../metal/coarse.py): its kernels and lists, the routing keys ordered by their
number of pairs, and the cell edges and centres read from the live stock stage on the device. In one class here:

  kernels   coarse.cu, compiled at construction through ../k9cu.py (CuPy, NVIDIA's run-time compiler); the Metal
            kernels translated statement by statement (the header of coarse.cu lists what differs).
  lists     lists.py, `build_lists(static)`: the Metal constructor's lists and key order, on numpy arrays.
  grid      `live_grid`: x_edges, y_edges, x_grid, y_grid of a stock CoarseStage on the device, as on Metal: a macro
            edge exactly on a cell edge takes the stock subgradient only if the kernel compares against the numbers the
            stock stage holds on this device.
  clamp     `stock_clamp_at_bound`: whether the stock clamp_min passes the gradient at exactly its bound is asked of the
            installed torch at construction and compiled into the kernels (torch 2.8.0, the Metal parts: yes; torch
            2.15.0, this port: no). The one place where these kernels do not decide as the Metal kernels do.

One step: 9 kernel launches and 5 torch operations (two batched products, two top-k, one sum), with the Adam update and
the projection in the last kernel (entry 2). Every weighted value goes to VAL [lanes, nets + hard + k + 2 R C]; the loss
is one sum over it. Nothing is carried between calls. Schedule values are kernel arguments.
"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.append(str(HERE.parent))      # cuda/: the launch path
import k9cu  # noqa: E402


def _sibling(name, path):
    """A file of this folder as a module under a name of its own (another folder's lists.py does not collide)."""
    if name not in sys.modules:
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


lists = _sibling("k9c_coarse_lists", HERE / "lists.py")

TERMS = ("wirelength", "density", "routing", "overlap")
BIT = {"wirelength": 1, "density": 2, "routing": 4, "overlap": 8}
GRID_KEYS = ("x_edges", "y_edges", "x_grid", "y_grid")
_LIVE = {}   # (design, device) -> the four arrays of a stock stage on that device


def live_grid(static, device, stage=None):
    """The four grid arrays as the stock coarse stage holds them on the device (float32 numpy)."""
    key = (str(static["design"]), str(device))
    if stage is not None:
        return {k: getattr(stage.proxy, k).detach().cpu().numpy().astype(np.float32) for k in GRID_KEYS}
    if key not in _LIVE:
        import coarsestatic                # the research harness's stock coarse stage; pass `stage` to avoid it
        proxy = coarsestatic.new_coarse(key[0], device).proxy
        _LIVE[key] = {k: getattr(proxy, k).detach().cpu().numpy().astype(np.float32) for k in GRID_KEYS}
    return _LIVE[key]


def stock_clamp_at_bound(device):
    """1 if clamp_min of the installed torch passes the gradient where its input is exactly on the bound (torch 2.8.0,
    the stock code of the Metal parts), 0 if it does not (torch 2.15.0). Asked of the stock operation itself, on the device."""
    with torch.enable_grad():
        x = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
        g, = torch.autograd.grad(x.clamp_min(0).sum(), x)
    return int(float(g) != 0.0)


class CudaCoarse:
    name = "cuda-coarse"
    BLOCK = 32            # threads per block of every launch; BLOCKS overrides per kernel. The Metal thread-group size is 64:
                          # a sweep on the L4 read the step 2 to 5% shorter at 32 (an unrefereed timing)
    BLOCKS = {}
    OPTIONS = ()          # compiler options beyond CuPy's own (-ftz=true; NVRTC defaults otherwise, --fmad=true)
    CHECK = True          # k9cu verifies the argument kinds of every launch
    LIVE_GRID = True
    CLAMP_AT_BOUND = None  # the subgradient of a clamp exactly on its bound: None = what the installed stock torch does; 1 or 0 forces it

    def __init__(self, static, lanes, device, stage=None):
        d = self.device = torch.device(device)
        if d.type != "cuda":
            raise ValueError("CudaCoarse is a set of CUDA kernels: device cuda, float32 only")
        if self.LIVE_GRID:
            live = live_grid(static, device, stage)
            for k in GRID_KEYS:
                assert live[k].shape == static[k].shape, k
            # entries where the device's numbers are not the frozen ones (kept for records)
            self.grid_differs = {k: [int(i) for i in np.flatnonzero(live[k] != static[k].astype(np.float32))] for k in GRID_KEYS}
            static = dict(static)
            static.update(live)
        L = lists.build_lists(static)
        for k in lists.SCALARS:
            setattr(self, k, getattr(L, k))
        for k in lists.ARRAYS:
            setattr(self, k, torch.from_numpy(getattr(L, k)).to(d))
        self.seg = L.seg
        self.w_density, self.w_routing = float(static["weight_density"]), float(static["weight_routing"])
        self.mean_weight = float(static["route_mean_weight"])
        self.lr, self.beta1, self.beta2, self.eps = (float(static[k]) for k in ("adam_lr", "adam_beta1", "adam_beta2", "adam_eps"))
        self.stock_clamp_at_bound = stock_clamp_at_bound(d)
        self.clamp_at_bound = self.stock_clamp_at_bound if self.CLAMP_AT_BOUND is None else int(self.CLAMP_AT_BOUND)
        self.options = tuple(self.OPTIONS) + (f"-DCOARSE_CLAMP_AT_BOUND={self.clamp_at_bound}",)
        self.lib = k9cu.Library([HERE / "coarse.cu"], block=self.BLOCK, blocks=self.BLOCKS, options=self.options, check=self.CHECK)
        self._buf = {}

    def _buffers(self, b):
        if b not in self._buf:
            d, N, NH, R, C = self.device, self.N, self.NH, self.R, self.C
            z = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=d)  # noqa: E731
            s = self.seg
            self._buf[b] = types.SimpleNamespace(
                ip=torch.tensor([N, NH, R, C, b, self.NN, self.KD, self.KR, self.KA, self.KT, self.K2, self.VW, s["wirelength"][0],
                                 s["overlap"][0], s["density"][0], s["routing"][0], self.P, self.E, self.KOFF], dtype=torch.int32, device=d),
                val=z(b, self.VW), pgw=z(b, self.P, 2), fyt=z(b, R, N), fx=z(b, N, C), gd=z(b, N, 2),
                left=z(2 * b, R, self.K2), right=z(2 * b, self.K2, C), cot=z(b, 2 * R * C), kg=z(b, self.KT, 2), pg=z(b, self.E),
                gb=z(b, NH, 2), go=z(b, NH, 2), g=z(b, N, 2), m0=z(b, N, 2), v0=z(b, N, 2))
        return self._buf[b]

    def _product(self, left, right, term):
        """The batched matrix product of the two factors of a term: torch's, as in the Metal step. A method of its
        own so that a variant can make the product in a kernel."""
        return torch.bmm(left, right)

    def _top(self, cells, k, term):
        """(values, indices) of the k largest cells of every lane. A method of its own only so that a diagnostic can hand
        the kernels another selection (the live stock step's own, at a near-tie of a tail mean)."""
        return torch.topk(cells, k, dim=1)

    def _pipeline(self, B, pos, s):
        """Everything up to `combine`: the values in VAL and the per-term gradient buffers."""
        b = pos.shape[0]
        lib, ip, cf = self.lib, B.ip, self.cf
        N, NH, RC = self.N, self.NH, self.RC
        floors = [s.sigma_floor_x, s.sigma_floor_y]
        lib.coarse_wl(B.pgw, B.val, pos, self.net_start, self.pin_macro, self.pin_off, self.net_w, ip, cf, [s.gamma], threads=b * self.NN)
        lib.coarse_dfac(B.fyt, B.fx, pos, self.sig, self.area, self.xe, self.ye, ip, cf, floors, threads=b * N)
        lib.coarse_rfwd(B.left, B.right, pos, self.kmac, self.kwoff, self.kstart, self.eoth, self.eown, self.eooff, self.ew, self.xg, self.yg,
                        ip, cf, threads=b * self.KT)
        lib.coarse_bfac(B.left, B.right, pos, self.half, self.xe, self.ye, ip, cf, threads=b * NH)
        vals, idx = self._top(self._product(B.fyt, B.fx, "density").view(b, RC), self.KD, "density")
        lib.coarse_dgrad(B.gd, B.val, pos, self.sig, self.area, self.xe, self.ye, idx, vals, ip, cf,
                         floors + [self.w_density * 0.5 / self.KD], threads=b * N)
        load = self._product(B.left, B.right, "routing").view(b, 2 * RC)
        top = self._top(load, self.KR, "routing")[1]
        lib.coarse_cot(B.cot, B.val, load, top, ip, [self.w_routing / self.KR, self.w_routing * self.mean_weight / (2 * RC)], threads=b * 2 * RC)
        lib.coarse_rbwd(B.kg, B.pg, B.cot, B.left, B.right, pos, self.kmac, self.kwoff, self.kstart, self.eoth, self.eown, self.eooff, self.ew,
                        self.xg, self.yg, ip, cf, threads=b * self.KT)
        lib.coarse_hard(B.gb, B.go, B.val, B.cot, pos, self.half, self.xe, self.ye, ip, cf, [s.penalty], threads=b * NH)

    def _combine(self, B, out, pos, m, v, select, update):
        self.lib.coarse_combine(out, pos, m, v, B.pgw, self.mstart, self.mentry, B.gd, B.kg, self.kms, self.kme, B.pg, self.exs, self.exe,
                                self.eys, self.eye, B.gb, B.go, self.mov, self.half, self.high, self.fixed, B.ip, [float(select)] + update,
                                threads=pos.shape[0] * self.N)

    def loss_and_grad(self, pos, s):
        with torch.no_grad():
            pos = pos.detach().contiguous()
            B = self._buffers(pos.shape[0])
            self._pipeline(B, pos, s)
            out = torch.empty(pos.shape[0], self.N, 2, dtype=torch.float32, device=self.device)
            self._combine(B, out, pos, B.m0, B.v0, 15, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            loss = B.val.sum(dim=1)
        return loss, out

    def terms(self, pos, s):
        out = {}
        with torch.no_grad():
            pos = pos.detach().contiguous()
            B = self._buffers(pos.shape[0])
            self._pipeline(B, pos, s)
            for t in TERMS:
                grad = torch.empty(pos.shape[0], self.N, 2, dtype=torch.float32, device=self.device)
                self._combine(B, grad, pos, B.m0, B.v0, BIT[t], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                lo, hi = self.seg[t]
                out[t] = (B.val[:, lo:hi].sum(dim=1), grad)
        return out

    def step(self, pos, exp_avg, exp_avg_sq, s):
        with torch.no_grad():
            if not (pos.is_contiguous() and exp_avg.is_contiguous() and exp_avg_sq.is_contiguous()):
                pos, exp_avg, exp_avg_sq = pos.contiguous(), exp_avg.contiguous(), exp_avg_sq.contiguous()
            B = self._buffers(pos.shape[0])
            self._pipeline(B, pos, s)
            t = s.adam_t
            self._combine(B, B.g, pos, exp_avg, exp_avg_sq, 15,
                          [1.0, 1.0 - self.beta1, self.beta2, 1.0 - self.beta2, self.lr / (1.0 - self.beta1 ** t),
                           math.sqrt(1.0 - self.beta2 ** t), self.eps])
            loss = B.val.sum(dim=1)
        return loss, B.g, pos, exp_avg, exp_avg_sq


class CudaCoarseBlock64(CudaCoarse):
    """The same kernels at the Metal thread-group size, 64 threads per block (the size the port started with)."""
    name = "cuda-coarse-block64"
    BLOCK = 64


class CudaCoarseNoFmad(CudaCoarse):
    """The same kernels compiled without contraction of multiply-add (asked for: the result both ways)."""
    name = "cuda-coarse-nofmad"
    OPTIONS = ("--fmad=false",)


class CudaCoarseMetalKinks(CudaCoarse):
    """The clamp choice of the Metal kernels (torch 2.8.0: the gradient passes at exactly the bound), whatever the
    installed torch does. Kept to show the result both ways; on torch 2.15.0 it is not the stock choice."""
    name = "cuda-coarse-metal-kinks"
    CLAMP_AT_BOUND = 1


class CudaCoarseUnchecked(CudaCoarse):
    """The same kernels, the launcher's argument checks off (timings with the checks on and off)."""
    name = "cuda-coarse-unchecked"
    CHECK = False

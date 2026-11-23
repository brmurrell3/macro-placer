"""The fine step of the lane-batched placement descent as CUDA kernels (float32).

A port of the Metal fine step (../../metal/fine.py). The kernels are plain CUDA C in pairs.cu (the pair
accumulation) and fine.cu (everything else), compiled at construction through ../k9cu.py; this file is the driver
code, one class. One step is 20 kernel launches and 5 torch operations (three batched matrix products, one top-k,
one sum), in the Metal step's order:

  the launches that need only the positions
      prep  wl_net  dfac3  ranges  assign_rows  box_c  box_r  bfac  ovl_part
  the grids, the selection and the masks
      bmm (density grid)  bmm (pair product: H and V)  bmm (blockage grids)  tail5  topk  setk5
  the gradients
      dbits2  dgradw2  cot  pair_grad  macro_sum  bgrad_part  hsum  combine  sum (the loss)
  entry 2
      adam

Every launch site is a method (`k_<kernel>` for a kernel, `t_<name>` for a torch operation), so that a subclass can
replace one launch, for example by a Triton twin that writes the same buffers.

What differs from the Metal class: no commits inside the step (torch.mps.event.Event.record() has no CUDA
counterpart: a CUDA launch is submitted when it is made); a Metal thread-group size is a CUDA block size (BLOCKS, BLOCK);
`constant float* fp` arguments are float parameters by value; the scratch buffers of the pair accumulation are
allocated per lane count with the others.
Nothing is carried between calls: every buffer is written in full before it is read inside one call.
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
    sys.path.insert(0, str(HERE.parent))
import k9cu  # noqa: E402  (cuda/k9cu.py: the launch path)

TERMS = ("wirelength", "density", "congestion", "overlap", "boundary")
BIT = {"wirelength": 1, "density": 2, "congestion": 4, "overlap": 8, "boundary": 16}
ALL = sum(BIT.values())
ROWMAX = 64      # a grid side of at most 64 cells: dbits2 and dgradw2 hold a row of the density mask in two 32-bit words
FAR = 5.0        # the density gradient of a macro is gathered within FAR widths (sigma sqrt 2) and one cell of it
OCH = 32         # hard macros per thread of ovl_part
# the Metal thread-group sizes as block sizes: 64 for the kernels whose threads loop, 256 for the others
LOOPING = ("wl_net", "dgradw2", "ovl_part", "bgrad_part", "hsum", "pair_grad", "macro_sum", "combine", "adam", "dbits2", "ranges", "assign_rows")


class Pairs:
    """The static lists of the pair accumulation (numpy, once)."""
    GROUP_MAX = 16            # pairs per row of a grouped factor at most (a pin with more pairs gets several rows)
    MERGE_DUPLICATES = True   # pairs with the same source pin and sink pin become one pair with the summed weight
    TOL_ASSIGN = 1.0e-10      # first omitted soft-assignment weight, relative to the largest weight of the row
    TOL_BOX = 5.0e-9          # soft box (and its derivative) at the first omitted cell

    def __init__(self, static, device):
        d = self.device = torch.device(device)
        self.P = int(static["src_macro"].shape[0])
        self.R, self.C = int(static["grid_rows"]), int(static["grid_cols"])
        self.N = N = int(static["n_macros"])
        src, snk = static["src_macro"].astype(np.int64), static["snk_macro"].astype(np.int64)
        if src.max() > N or snk.max() > N or src.min() < 0 or snk.min() < 0:
            raise ValueError("a macro index outside 0..N")

        # pins: the distinct (macro, offset) among sources and sinks, compared by their float32 bits
        def rows(macro, offset):
            out = np.empty((macro.shape[0], 3), dtype=np.int32)
            out[:, 0] = macro
            out[:, 1:] = np.ascontiguousarray(offset, dtype=np.float32).view(np.int32).reshape(-1, 2)
            return out

        pin_rows, which = np.unique(np.concatenate([rows(src, static["src_offset"]), rows(snk, static["snk_offset"])]), axis=0, return_inverse=True)
        which = which.reshape(-1)
        U = self.U = int(pin_rows.shape[0])
        pin_macro = pin_rows[:, 0].astype(np.int64)
        pin_off = np.ascontiguousarray(pin_rows[:, 1:]).view(np.float32).reshape(-1, 2)
        # merged pairs in the order (source pin, sink pin): the constructor's order, whatever order the pairs came in
        key = which[:self.P].astype(np.int64) * U + which[self.P:]
        weight = np.asarray(static["pair_w"], dtype=np.float64)
        if self.MERGE_DUPLICATES:
            merged, member = np.unique(key, return_inverse=True)
            w = np.bincount(member.reshape(-1), weights=weight, minlength=merged.shape[0])
        else:
            order = np.argsort(key, kind="stable")
            merged, w = key[order], weight[order]
        self.M = M = int(merged.shape[0])
        m_src, m_snk = merged // U, merged % U

        # rows of the grouped factors: the pairs of a pin, cut into runs of at most GROUP_MAX
        def groups(pin_of_pair, order):
            pin_sorted = pin_of_pair[order]
            first = np.flatnonzero(np.r_[True, pin_sorted[1:] != pin_sorted[:-1]])
            length = np.diff(np.r_[first, pin_sorted.shape[0]])
            starts, pins = [], []
            for begin, count, pin in zip(first.tolist(), length.tolist(), pin_sorted[first].tolist()):
                for offset in range(0, count, self.GROUP_MAX):
                    starts.append(begin + offset)
                    pins.append(pin)
            return np.asarray(starts + [pin_sorted.shape[0]], dtype=np.int64), np.asarray(pins, dtype=np.int64)

        by_sink = np.lexsort((m_src, m_snk))                       # pairs in the order (sink pin, source pin)
        gs_start, gs_pin = groups(m_src, np.arange(M))
        gt_start, gt_pin = groups(m_snk, by_sink)
        self.Gs, self.Gt = int(gs_pin.shape[0]), int(gt_pin.shape[0])
        self.G = G = max(self.Gs, self.Gt)

        def padded(start, pin):
            return (np.r_[start, np.full(G - pin.shape[0], start[-1])].astype(np.int32), np.r_[pin, np.full(G - pin.shape[0], -1)].astype(np.int32))

        gs_start, gs_pin = padded(gs_start, gs_pin)
        gt_start, gt_pin = padded(gt_start, gt_pin)
        on = lambda array, kind: torch.from_numpy(np.ascontiguousarray(array, dtype=kind)).to(d)  # noqa: E731
        self.src_pin, self.snk_pin = on(m_src, np.int32), on(m_snk, np.int32)
        self.pin_macro, self.pin_off = on(pin_macro, np.int32), on(pin_off, np.float32)
        self.pair_w = on(w, np.float32)
        if not np.array_equal(self.pair_w.cpu().numpy().astype(np.float64), w):
            raise ValueError("a merged pair weight is not a float32 number")
        self.gs_start, self.gs_pin, self.gt_start, self.gt_pin = on(gs_start, np.int32), on(gs_pin, np.int32), on(gt_start, np.int32), on(gt_pin, np.int32)
        self.gt_order = on(by_sink, np.int32)

        x_ctr, y_ctr = static["x_ctr"].astype(np.float32), static["y_ctr"].astype(np.float32)
        self.x_ctr, self.y_ctr = torch.from_numpy(x_ctr).to(d), torch.from_numpy(y_ctr).to(d)
        sigma_x, sigma_y = float(static["sigma_x"]), float(static["sigma_y"])
        beta, rho = float(static["beta_minmax"]), float(static["beta_range"])
        cell_w = float(np.diff(x_ctr.astype(np.float64)).mean())
        cell_h = float(np.diff(y_ctr.astype(np.float64)).mean())
        # soft assignment: a weight at j cells from the nearest cell is at most exp(-ratio j (j - 1)) of the largest
        ratio = min(cell_w / sigma_x, cell_h / sigma_y) ** 2      # 4 when sigma is half a cell
        wa = 1
        while math.exp(-ratio * wa * (wa + 1)) > self.TOL_ASSIGN:
            wa += 1
        # soft box: at the first omitted cell below the run at most exp(-rho (DL - ln 2 / beta)), above exp(-rho (DH + 1 - ln 2 / beta))
        dl = 1
        while math.exp(-rho * (dl - math.log(2.0) / beta)) > self.TOL_BOX:
            dl += 1
        if wa > 6 or dl > 16:
            raise ValueError(f"windows of {wa} and {dl} cells: sigma or the box edge is too wide for this kernel")
        self.WA, self.DL, self.DH = wa, dl, dl - 1
        self.bound_assign = math.exp(-ratio * wa * (wa + 1))
        self.bound_box = math.exp(-rho * (dl - math.log(2.0) / beta))
        self.fp = torch.tensor([1.0 / sigma_x, 1.0 / sigma_y, beta, 1.0 / beta, rho, 1.0 / cell_w, 1.0 / cell_h,
                                2.0 / sigma_x ** 2, 2.0 / sigma_y ** 2], dtype=torch.float32, device=d)
        # CSR list of the pins of every macro: entry = 4 p (source pin) or 4 p + 2 (sink pin); ports (index N) are dropped
        owner = np.concatenate([pin_macro[m_src], pin_macro[m_snk]])
        entry = np.concatenate([4 * np.arange(M, dtype=np.int64), 4 * np.arange(M, dtype=np.int64) + 2])
        keep = owner < N
        order = np.argsort(owner[keep], kind="stable")
        self.entry = on(entry[keep][order], np.int32)
        self.start = on(np.concatenate([[0], np.cumsum(np.bincount(owner[keep], minlength=N))]), np.int32)


class CudaFine:
    name = "cuda-fine"
    SOURCES = ("pairs.cu", "fine.cu")
    BLOCKS = {k: 64 for k in LOOPING}     # block size of a kernel (Metal: thread-group size)
    BLOCK = 256                           # block size of the kernels BLOCKS does not name
    OPTIONS = ()                          # extra compiler options, e.g. ("--fmad=false",)
    CHECK = True                          # k9cu verifies the argument kinds of every launch

    def __init__(self, static, lanes, device):
        d = self.device = torch.device(device)
        if d.type != "cuda":
            raise ValueError("CudaFine is a set of CUDA kernels: device cuda, float32 only")
        st = self.static = static
        N, NH = int(st["n_macros"]), int(st["n_hard"])
        R, C = int(st["grid_rows"]), int(st["grid_cols"])
        self.N, self.NH, self.R, self.C, self.RC = N, NH, R, C, R * C
        self.KD, self.KC, self.SR = int(st["dens_k"]), int(st["cong_k"]), int(st["smooth_range"])
        self.area = float(st["area"])
        self.w_density, self.w_congestion = float(st["weight_density"]), float(st["weight_congestion"])
        self.w_boundary, self.mean_weight = float(st["weight_boundary"]), float(st["cong_mean_weight"])
        on = lambda array, kind: torch.from_numpy(np.ascontiguousarray(array, dtype=kind)).to(d)  # noqa: E731
        half = np.asarray(st["half"], dtype=np.float32)
        self.half = on(half, np.float32)
        self.hi = on(np.stack([st["x_hi"], st["y_hi"]], axis=1), np.float32)
        self.mov = on(np.asarray(st["movable"]), np.float32)
        self.movable = self.mov.view(1, -1, 1)

        # wirelength: the nets as pin lists, nets with the same pins merged (weights added), CSR of pin slots per macro
        merged = {}
        for g in range(int(st["net_groups"])):
            owners, offsets = st[f"net{g}_owners"], np.ascontiguousarray(st[f"net{g}_offsets"], dtype=np.float32)
            mask, weights = st[f"net{g}_mask"], st[f"net{g}_weights"]
            bits = offsets.view(np.int32)
            for i in range(owners.shape[0]):
                key = tuple(sorted((int(owners[i, k]), int(bits[i, k, 0]), int(bits[i, k, 1])) for k in range(owners.shape[1]) if mask[i, k]))
                merged[key] = merged.get(key, 0.0) + float(weights[i])
        self.nets_given = int(sum(st[f"net{g}_owners"].shape[0] for g in range(int(st["net_groups"]))))
        self.NN = NN = len(merged)
        pins = np.asarray([p for key in merged for p in key], dtype=np.int64).reshape(-1, 3)
        degree = np.asarray([len(key) for key in merged], dtype=np.int64)
        self.P = P = int(pins.shape[0])
        self.net_start = on(np.concatenate([[0], np.cumsum(degree)]), np.int32)
        self.pin_macro = on(pins[:, 0], np.int32)
        self.pin_off = on(np.ascontiguousarray(pins[:, 1:].astype(np.int32)).view(np.float32).reshape(-1, 2), np.float32)
        self.net_w = on(np.asarray(list(merged.values()), dtype=np.float64) / float(st["wl_norm"]), np.float32)
        owner = pins[:, 0]
        keep = np.flatnonzero(owner < N)
        order = keep[np.argsort(owner[keep], kind="stable")]
        self.mentry = on(order, np.int32)
        self.mstart = on(np.concatenate([[0], np.cumsum(np.bincount(owner[keep], minlength=N))]), np.int32)

        # density
        self.chq = on(np.asarray(st["charge"], dtype=np.float64) / float(st["cell_area"]), np.float32)
        self.dens = [on(st[f"dens_{k}"], np.float32) for k in ("x_min", "x_max", "y_min", "y_max")]
        if max(R, C) > ROWMAX:
            raise ValueError(f"grid {R} x {C}: dbits2 and dgradw2 hold a grid side of at most {ROWMAX} cells")
        x_min, x_max = np.asarray(st["dens_x_min"], dtype=np.float64), np.asarray(st["dens_x_max"], dtype=np.float64)
        y_min, y_max = np.asarray(st["dens_y_min"], dtype=np.float64), np.asarray(st["dens_y_max"], dtype=np.float64)
        cw, ch = (x_max[-1] - x_min[0]) / C, (y_max[-1] - y_min[0]) / R
        for lo, hi, size, n in ((x_min, x_max, cw, C), (y_min, y_max, ch, R)):
            if np.abs(lo - (lo[0] + size * np.arange(n))).max() > 1e-3 * size or np.abs(hi - (lo[0] + size * (np.arange(n) + 1))).max() > 1e-3 * size:
                raise ValueError("the density grid is not uniform: the window of dgradw2 assumes it is")
        self.gf = torch.tensor([float(x_min[0]), float(y_min[0]), cw, ch], dtype=torch.float32, device=d)
        self.KM = max(self.KD, self.KC)
        if self.KM > self.RC:
            raise ValueError("the stacked top-k needs k <= R C")

        # congestion outside the pair accumulation
        self.trace = [on(st[f"trace_{k}"], np.float32) for k in ("x_min", "x_max", "y_min", "y_max")]
        v_cap, h_cap = float(st["v_capacity"]), float(st["h_capacity"])
        cell_w, cell_h = float(st["cell_w"]), float(st["cell_h"])
        x_min, y_min = np.asarray(st["trace_x_min"], dtype=np.float64), np.asarray(st["trace_y_min"], dtype=np.float64)
        if (np.abs(x_min - (x_min[0] + cell_w * np.arange(C))).max() > 1e-3 * cell_w
                or np.abs(y_min - (y_min[0] + cell_h * np.arange(R))).max() > 1e-3 * cell_h):
            raise ValueError("the routing grid is not uniform: the window of bgrad_part assumes it is")
        self.cf = torch.tensor([float(st["v_alloc"]) / v_cap, float(st["h_alloc"]) / h_cap, cell_w, cell_h, float(x_min[0]), float(y_min[0])],
                               dtype=torch.float32, device=d)
        self.icv = on(1.0 / (v_cap * np.asarray(st["count_v"], dtype=np.float64)), np.float32)
        self.ich = on(1.0 / (h_cap * np.asarray(st["count_h"], dtype=np.float64)), np.float32)
        self.hot = Pairs(static, d)
        # the short threads of the overlap and of the blockage gradient
        half_y = np.asarray(st["half"], dtype=np.float64)[:NH, 1]
        self.Q = -(-NH // OCH)
        # rows of the blockage gradient's window: floor((y + hy - y0) / h) + 1 - (floor((y - hy - y0) / h) - 1) + 1 <= floor(2 hy / h) + 4; one more for rounding
        self.RQ = min(R, int(math.floor(2.0 * float(half_y.max()) / float(st["cell_h"]))) + 5)

        # the value buffer: one segment per term
        self.seg, at = {}, 0
        for term, width in (("wirelength", NN), ("overlap", NH), ("boundary", N), ("density", self.KD), ("congestion", R * C)):
            self.seg[term] = (at, at + width)
            at += width
        self.VW = at
        self.lr, self.beta1, self.beta2, self.eps = (float(st[k]) for k in ("adam_lr", "adam_beta1", "adam_beta2", "adam_eps"))

        h = self.hot
        self.options = (f"-DWA={h.WA}", f"-DDL={h.DL}", f"-DDH={h.DH}", "-DFARW=%.9ff" % (FAR * math.sqrt(2.0)), f"-DOCH={OCH}") + tuple(self.OPTIONS)
        self.lib = k9cu.Library([HERE / name for name in self.SOURCES], block=self.BLOCK, blocks=self.BLOCKS, options=self.options, check=self.CHECK)
        self._buf = {}

    def _buffers(self, b):
        if b not in self._buf:
            d, N, NH, R, C, RC, h = self.device, self.N, self.NH, self.R, self.C, self.RC, self.hot
            if max(2 * b * h.G * (R + C), 4 * b * h.M, b * N * (R + C), 2 * b * self.P, 4 * b * RC, b * self.VW, 2 * b * self.KM,
                   3 * b * NH * self.Q, 2 * b * NH * max(self.RQ, R, C)) >= 2 ** 31:
                raise ValueError(f"{b} lanes: 32-bit index arithmetic in the kernels")
            z = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=d)  # noqa: E731
            e = lambda *shape: torch.empty(*shape, dtype=torch.float32, device=d)  # noqa: E731
            i32 = lambda values: torch.tensor(values, dtype=torch.int32, device=d)  # noqa: E731
            s = self.seg
            self._buf[b] = types.SimpleNamespace(
                ip=i32([N, NH, R, C, b, self.NN, self.KD, self.KC, self.SR, self.VW, s["wirelength"][0], s["overlap"][0],
                        s["boundary"][0], s["density"][0], s["congestion"][0], self.P]),
                c=z(b, N, 2), val=z(b, self.VW), pgw=z(b, self.P, 2),
                fx=z(b, N, C), dx=z(b, N, C), fyt=z(b, R, N), dy=z(b, N, R), fy=z(b, N, R), gd=z(b, N, 2),
                bl=z(2 * b, R, NH), br=z(2 * b, NH, C), dbx=z(b, NH, C, 2), dby=z(b, NH, R, 2),
                mask=z(b, 2 * RC), gv=z(b, R, C), gh=z(b, R, C), gb=z(b, NH, 2), go=z(b, NH, 2), gc=z(b, N, 2),
                rci=torch.zeros(b, self.KD, 2, dtype=torch.int32, device=d), bits=torch.zeros(b, R, 2, dtype=torch.int32, device=d),
                tk=z(2 * b, 2 * RC), jpb=i32([b, self.KM]), dm=z(b, RC),
                op=z(b, NH, self.Q, 3), bp=z(b, NH, self.RQ, 2), jq=i32([self.Q, self.RQ]),
                # the pair accumulation (v2): its six integers; runs RI, RF [b, M, 4]; grouped factors FR [2 b, G, R] and
                # FC [2 b, G, C] (rows 0..b-1: the H product of each lane, b..2b-1: the V product); pin gradients PG [b, M, 4]
                hip=i32([h.M, N, R, C, h.G, b]),
                ri=torch.empty(b, h.M, 4, dtype=torch.int32, device=d), rf=e(b, h.M, 4), fr=e(2 * b, h.G, R), fc=e(2 * b, h.G, C),
                pg=e(b, h.M, 4))
        return self._buf[b]

    # ---- the launch sites: one method per kernel launch (k_) and per torch operation (t_) -----------------------------
    def k_prep(self, B, pos, b):
        self.lib.prep(B.c, pos, self.half, self.hi, B.ip, threads=b * self.N)

    def k_wl_net(self, B, s, b):
        self.lib.wl_net(B.pgw, B.val, B.c, self.net_start, self.pin_macro, self.pin_off, self.net_w, B.ip, [s.gamma], threads=b * self.NN)

    def k_dfac3(self, B, s, b):
        self.lib.dfac3(B.fx, B.dx, B.fyt, B.dy, B.fy, B.c, self.half, self.chq, *self.dens, B.ip,
                       [s.sigma_scale, s.sigma_floor_x, s.sigma_floor_y], threads=b * self.N * (self.R + self.C))

    def k_ranges(self, B, b):
        h = self.hot
        self.lib.ranges(B.ri, B.rf, B.c, h.src_pin, h.snk_pin, h.pin_macro, h.pin_off, h.x_ctr, h.y_ctr, B.hip, h.fp, threads=b * h.M)

    def k_assign_rows(self, B, b):
        h = self.hot
        self.lib.assign_rows(B.fr, B.fc, B.c, h.gs_pin, h.gt_pin, h.pin_macro, h.pin_off, h.x_ctr, h.y_ctr, B.hip, h.fp, threads=b * h.G)

    def k_box_c(self, B, b):
        h = self.hot
        self.lib.box_c(B.fc, B.ri, B.rf, h.pair_w, h.gs_start, B.hip, h.fp, threads=b * h.G * self.C)

    def k_box_r(self, B, b):
        h = self.hot
        self.lib.box_r(B.fr, B.ri, B.rf, h.pair_w, h.gt_start, h.gt_order, B.hip, h.fp, threads=b * h.G * self.R)

    def k_bfac(self, B, b):
        self.lib.bfac(B.bl, B.br, B.dbx, B.dby, B.c, self.half, *self.trace, B.ip, self.cf, threads=b * self.NH * (self.R + self.C))

    def k_ovl_part(self, B, b):
        self.lib.ovl_part(B.op, B.c, self.half, B.ip, B.jq, threads=b * self.NH * self.Q)

    def t_density_grid(self, B):
        return torch.bmm(B.fyt, B.fx)                         # [b, R, C]

    def t_pair_product(self, B):
        return B.fr.transpose(1, 2).bmm(B.fc)                 # [2 b, R, C]: rows 0..b-1 H, rows b..2b-1 V

    def t_blockage_grid(self, B):
        return torch.bmm(B.bl, B.br)                          # [2 b, R, C]: rows 0..b-1 for V, rows b..2b-1 for H

    def k_tail5(self, B, v, h, mb, dgrid, b):
        self.lib.tail5(B.tk, B.mask, B.dm, v, h, mb, dgrid, self.icv, self.ich, B.ip, self.cf, threads=b * self.RC)

    def t_topk(self, B):
        return torch.topk(B.tk, self.KM, dim=1)               # ONE top-k over the load and the density grid stacked

    def k_setk5(self, B, vals, idx, cd, b):
        self.lib.setk5(B.mask, B.val, B.dm, B.rci, vals, idx, B.ip, B.jpb, cd, threads=2 * b * self.KM)

    def k_dbits2(self, B, b):
        self.lib.dbits2(B.bits, B.dm, B.ip, threads=2 * b * self.R)

    def k_dgradw2(self, B, s, cd, b):
        self.lib.dgradw2(B.gd, B.fx, B.dx, B.fy, B.dy, B.bits, B.c, self.half, B.ip, self.gf,
                         [cd[0], s.sigma_scale, s.sigma_floor_x, s.sigma_floor_y], threads=b * self.N)

    def k_cot(self, B, cc, b):
        self.lib.cot(B.gv, B.gh, B.val, B.tk, B.mask, self.icv, self.ich, B.ip, cc, threads=b * self.RC)

    def k_pair_grad(self, B, b):
        h = self.hot
        self.lib.pair_grad(B.pg, B.gv, B.gh, B.c, h.src_pin, h.snk_pin, h.pin_macro, h.pin_off, h.pair_w, h.x_ctr, h.y_ctr, B.hip, h.fp,
                           threads=b * h.M)

    def k_macro_sum(self, B, b):
        h = self.hot
        self.lib.macro_sum(B.gc, B.pg, h.start, h.entry, B.hip, threads=b * self.N)

    def k_bgrad_part(self, B, cc, b):
        self.lib.bgrad_part(B.bp, B.mask, B.bl, B.br, B.dbx, B.dby, B.c, self.half, B.ip, self.cf, B.jq, cc, threads=b * self.NH * self.RQ)

    def k_hsum(self, B, s, b):
        self.lib.hsum(B.go, B.gb, B.val, B.op, B.bp, B.ip, B.jq, [s.overlap_lambda / self.area], threads=b * self.NH)

    def k_combine(self, out, B, pos, select, b):
        self.lib.combine(out, B.val, pos, self.half, self.hi, self.mov, B.pgw, self.mstart, self.mentry, B.gd, B.gc, B.gb, B.go, B.ip,
                         [float(select), self.w_boundary / self.area], threads=b * self.N)

    def t_loss(self, B):
        return B.val.sum(dim=1)

    def k_adam(self, pos, exp_avg, exp_avg_sq, grad, s):
        t = s.adam_t
        self.lib.adam(pos, exp_avg, exp_avg_sq, grad, [1.0 - self.beta1, self.beta2, 1.0 - self.beta2, self.lr / (1.0 - self.beta1 ** t),
                                                       math.sqrt(1.0 - self.beta2 ** t), self.eps], threads=pos.numel())

    # ---- the step -----------------------------------------------------------------------------------------------------
    def _pipeline(self, pos, s):
        """Everything up to `combine`: the clamped positions, the values in VAL and the per-term gradient buffers."""
        if pos.dtype != torch.float32 or pos.dim() != 3 or pos.shape[1:] != (self.N, 2):
            raise ValueError(f"positions must be float32 [lanes, {self.N}, 2], got {pos.dtype} {tuple(pos.shape)}")
        b = pos.shape[0]
        B = self._buffers(b)
        cd = [self.w_density * 0.5 / self.KD]
        cc = [self.w_congestion / self.KC, self.w_congestion * self.mean_weight / (2 * self.RC)]
        # the launches that need only the positions
        self.k_prep(B, pos, b)
        self.k_wl_net(B, s, b)
        self.k_dfac3(B, s, b)
        self.k_ranges(B, b)
        self.k_assign_rows(B, b)
        self.k_box_c(B, b)
        self.k_box_r(B, b)
        self.k_bfac(B, b)
        self.k_ovl_part(B, b)
        # the three batched products, the load, ONE top-k, the masks; the density grid's selected values go to VAL
        dgrid = self.t_density_grid(B)
        hv = self.t_pair_product(B)
        mb = self.t_blockage_grid(B)
        self.k_tail5(B, hv[b:], hv[:b], mb, dgrid, b)
        vals, idx = self.t_topk(B)
        self.k_setk5(B, vals, idx, cd, b)
        # the gradients of the terms
        self.k_dbits2(B, b)
        self.k_dgradw2(B, s, cd, b)
        self.k_cot(B, cc, b)
        self.k_pair_grad(B, b)
        self.k_macro_sum(B, b)
        self.k_bgrad_part(B, cc, b)
        self.k_hsum(B, s, b)
        return B

    def _combine(self, B, pos, select):
        out = torch.empty(pos.shape[0], self.N, 2, dtype=torch.float32, device=self.device)
        self.k_combine(out, B, pos, select, pos.shape[0])
        return out

    def loss_and_grad(self, pos, s):
        with torch.no_grad():
            pos = pos.detach().contiguous()
            B = self._pipeline(pos, s)
            grad = self._combine(B, pos, ALL)
            loss = self.t_loss(B)
        return loss, grad

    def terms(self, pos, s):
        out = {}
        with torch.no_grad():
            pos = pos.detach().contiguous()
            B = self._pipeline(pos, s)
            grads = {t: self._combine(B, pos, BIT[t]) for t in TERMS}     # boundary's call writes its values
            for t in TERMS:
                lo, hi = self.seg[t]
                out[t] = (B.val[:, lo:hi].sum(dim=1), grads[t])
        return out

    def step(self, pos, exp_avg, exp_avg_sq, s):
        if not (pos.is_contiguous() and exp_avg.is_contiguous() and exp_avg_sq.is_contiguous()):
            pos, exp_avg, exp_avg_sq = pos.contiguous(), exp_avg.contiguous(), exp_avg_sq.contiguous()
        loss, grad = self.loss_and_grad(pos, s)
        self.k_adam(pos, exp_avg, exp_avg_sq, grad, s)
        return loss, grad, pos, exp_avg, exp_avg_sq

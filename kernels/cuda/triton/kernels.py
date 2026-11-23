"""Two kernels of the fine step in the Triton kernel language (triton-lang 3.9.0), float32.

Twins of two CUDA kernels of ../fine/fine.cu, writing the same buffers:

  dfac3 (the density factors; dense and regular)
      dfac3_flat   the twin's layout: one element per (lane, macro, e in C + R), a program holds BLOCK of them
      dfac3_tile   the layout the language suggests: a program holds a tile [BM macros, EP grid elements] of one lane
  wl_net (the wirelength per net; a loop over the net's pins, whose number differs from net to net)
      wl_loop      a program holds BN nets and loops, at run time, to the largest pin count among them, with masks;
                   the sums run in pin order as in the twin. ORDERED: the nets come through a list sorted by pin count.
                   PIN: which product of the pin gradient e_hi i_hi - e_lo i_lo is fused with the subtraction: 0 the
                   compiler's choice under enable_fp_fusion, 1 the first (tl.fma), 2 the second (the choice the CUDA
                   twin's compiler made on this machine: tools/twin.py)
      wl_pad       a program holds a tile [BN nets, DP pin slots] with masks and reduces along the pins; one launch
                   with DP = the largest pin count rounded up to a power of two, or one launch per bucket of pin counts

Arithmetic: the formulas of the twins in the same order. Two switches choose how they are computed.
LIBM True: libdevice's erf, exp and log (triton.language.extra.libdevice; exp and log gave the bits of the CUDA
twin's expf and logf on this machine, erf did not: tools/mathcheck.py). LIBM False: tl.erf, tl.exp (lowered to the
approximate instruction ex2.approx.f32) and tl.log. IEEE True: divisions rounded to nearest
(tl.fdiv(..., ieee_rounding=True), the CUDA twin's division). IEEE False: Triton's default float32 division, the PTX
instruction div.full.f32, which is not rounded to nearest. Both False is what a Triton kernel gets when nothing is
asked for. Contraction of a multiply and an add is a launch option (enable_fp_fusion), set by the caller.

Nothing is kept between calls: a kernel reads its arguments and writes its outputs. The compile cache is Triton's own
(on disk, TRITON_CACHE_DIR; set below to a folder beside this file unless the caller has set it).
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("TRITON_CACHE_DIR", str(HERE / "cache"))

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.language.extra import libdevice  # noqa: E402

_DF_INTS = ["N", "R", "C", "total"]
_DF_PTRS = ["FX", "DX", "FYT", "DY", "FY", "c", "half_", "chq", "xlo", "xhi", "ylo", "yhi"]
_WL_INTS = ["N", "NN", "P", "VW", "V_WL", "first", "count"]
_WL_PTRS = ["PGW", "VAL", "c", "net_start", "pm", "po", "net_w", "order"]


@triton.jit
def _erf(x, LIBM: tl.constexpr):
    if LIBM:
        y = libdevice.erf(x)
    else:
        y = tl.erf(x)
    return y


@triton.jit
def _exp(x, LIBM: tl.constexpr):
    if LIBM:
        y = libdevice.exp(x)
    else:
        y = tl.exp(x)
    return y


@triton.jit
def _log(x, LIBM: tl.constexpr):
    if LIBM:
        y = libdevice.log(x)
    else:
        y = tl.log(x)
    return y



# ---- the density factors ---------------------------------------------------------------------------------------------
# fp0: sigma_scale, fp1: floor_x, fp2: floor_y. FX [B, N, C] mass per column, DX its derivative by x; FYT [B, R, N] =
# q x mass per row (q = charge / cell_area), FY [B, N, R] the same in macro order, DY [B, N, R] = q x its derivative by y.
@triton.jit(do_not_specialize=_DF_INTS, do_not_specialize_on_alignment=_DF_PTRS)
def dfac3_flat(FX, DX, FYT, DY, FY, c, half_, chq, xlo, xhi, ylo, yhi, N, R, C, total, fp0, fp1, fp2,
               BLOCK: tl.constexpr, LIBM: tl.constexpr, IEEE: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = i < total
    E = R + C
    NE = N * E
    lane = i // NE
    rem = i - lane * NE
    n = rem // E
    e = rem - n * E
    ln = lane * N + n
    r = e - C
    mx = live & (e < C)
    my = live & (e >= C)
    ax = tl.where(e < C, 0, 1)
    p = tl.load(c + 2 * ln + ax, mask=live, other=0.0)
    hv = tl.load(half_ + 2 * n + ax, mask=live, other=1.0)
    lo = tl.where(e < C, tl.load(xlo + e, mask=mx, other=0.0), tl.load(ylo + r, mask=my, other=0.0))
    hi = tl.where(e < C, tl.load(xhi + e, mask=mx, other=0.0), tl.load(yhi + r, mask=my, other=0.0))
    sw = fp0 * hv
    fl = tl.where(e < C, fp1, fp2)
    w = tl.where(sw < fl, fl, sw) * 1.41421356237309504880
    t_lo = tl.fdiv(lo - p, w, ieee_rounding=IEEE)
    t_hi = tl.fdiv(hi - p, w, ieee_rounding=IEEE)
    f = 0.5 * (_erf(t_hi, LIBM) - _erf(t_lo, LIBM))
    d = (_exp(-t_lo * t_lo, LIBM) - _exp(-t_hi * t_hi, LIBM)) * tl.fdiv(0.56418958354775628695, w, ieee_rounding=IEEE)
    tl.store(FX + ln * C + e, f, mask=mx)
    tl.store(DX + ln * C + e, d, mask=mx)
    q = tl.load(chq + n, mask=my, other=0.0)
    mass = q * f
    tl.store(FYT + (lane * R + r) * N + n, mass, mask=my)
    tl.store(FY + ln * R + r, mass, mask=my)
    tl.store(DY + ln * R + r, q * d, mask=my)


@triton.jit(do_not_specialize=_DF_INTS, do_not_specialize_on_alignment=_DF_PTRS)
def dfac3_tile(FX, DX, FYT, DY, FY, c, half_, chq, xlo, xhi, ylo, yhi, N, R, C, fp0, fp1, fp2,
               BM: tl.constexpr, EP: tl.constexpr, LIBM: tl.constexpr, IEEE: tl.constexpr):
    lane = tl.program_id(1)
    n = tl.program_id(0) * BM + tl.arange(0, BM)
    nl = n < N
    ln = lane * N + n
    e = tl.arange(0, EP)
    # columns
    mc = e < C
    x = tl.load(c + 2 * ln, mask=nl, other=0.0)
    sx = fp0 * tl.load(half_ + 2 * n, mask=nl, other=1.0)
    wx = tl.where(sx < fp1, fp1, sx) * 1.41421356237309504880
    t_lo = tl.fdiv(tl.load(xlo + e, mask=mc, other=0.0)[None, :] - x[:, None], wx[:, None], ieee_rounding=IEEE)
    t_hi = tl.fdiv(tl.load(xhi + e, mask=mc, other=0.0)[None, :] - x[:, None], wx[:, None], ieee_rounding=IEEE)
    f = 0.5 * (_erf(t_hi, LIBM) - _erf(t_lo, LIBM))
    d = (_exp(-t_lo * t_lo, LIBM) - _exp(-t_hi * t_hi, LIBM)) * tl.fdiv(0.56418958354775628695, wx, ieee_rounding=IEEE)[:, None]
    mask = nl[:, None] & mc[None, :]
    at = (ln * C)[:, None] + e[None, :]
    tl.store(FX + at, f, mask=mask)
    tl.store(DX + at, d, mask=mask)
    # rows
    mr = e < R
    y = tl.load(c + 2 * ln + 1, mask=nl, other=0.0)
    sy = fp0 * tl.load(half_ + 2 * n + 1, mask=nl, other=1.0)
    wy = tl.where(sy < fp2, fp2, sy) * 1.41421356237309504880
    q = tl.load(chq + n, mask=nl, other=0.0)
    u_lo = tl.fdiv(tl.load(ylo + e, mask=mr, other=0.0)[None, :] - y[:, None], wy[:, None], ieee_rounding=IEEE)
    u_hi = tl.fdiv(tl.load(yhi + e, mask=mr, other=0.0)[None, :] - y[:, None], wy[:, None], ieee_rounding=IEEE)
    fy = 0.5 * (_erf(u_hi, LIBM) - _erf(u_lo, LIBM))
    dy = (_exp(-u_lo * u_lo, LIBM) - _exp(-u_hi * u_hi, LIBM)) * tl.fdiv(0.56418958354775628695, wy, ieee_rounding=IEEE)[:, None]
    mask = nl[:, None] & mr[None, :]
    mass = q[:, None] * fy
    at = (ln * R)[:, None] + e[None, :]
    tl.store(FYT + (lane * R + e)[None, :] * N + n[:, None], mass, mask=mask)
    tl.store(FY + at, mass, mask=mask)
    tl.store(DY + at, q[:, None] * dy, mask=mask)


# ---- the wirelength per net ------------------------------------------------------------------------------------------
# g: gamma. Value w (max - min + gamma (log S_hi + log S_lo)) per axis, with S_hi = sum exp((x - max) / gamma),
# S_lo = sum exp((min - x) / gamma); pin gradient w (e_hi / S_hi - e_lo / S_lo). A pin coordinate is the float32 sum
# position + offset together with its rounding error (two-term sum); a port (macro N) sits at its offset.
@triton.jit(do_not_specialize=_WL_INTS, do_not_specialize_on_alignment=_WL_PTRS)
def wl_loop(PGW, VAL, c, net_start, pm, po, net_w, order, N, NN, P, VW, V_WL, g,
            BN: tl.constexpr, ORDERED: tl.constexpr, LIBM: tl.constexpr, IEEE: tl.constexpr, PIN: tl.constexpr):
    lane = tl.program_id(1)
    j = tl.program_id(0) * BN + tl.arange(0, BN)
    live = j < NN
    if ORDERED:
        n = tl.load(order + j, mask=live, other=0)
    else:
        n = j
    k0 = tl.load(net_start + n, mask=live, other=0)
    deg = tl.load(net_start + n + 1, mask=live, other=0) - k0
    most = tl.max(deg, axis=0)
    w = tl.load(net_w + n, mask=live, other=0.0)
    cl = c + 2 * lane * N
    val = tl.zeros([BN], tl.float32)
    for ax in tl.static_range(2):
        smax = tl.full([BN], -3.0e38, tl.float32)
        emax = tl.zeros([BN], tl.float32)
        smin = tl.full([BN], 3.0e38, tl.float32)
        emin = tl.zeros([BN], tl.float32)
        for t in range(most):
            k = k0 + t
            on = t < deg
            m = tl.load(pm + k, mask=on, other=0)
            base = tl.load(cl + 2 * m + ax, mask=on & (m < N), other=0.0)
            off = tl.load(po + 2 * k + ax, mask=on, other=0.0)
            s = base + off
            bb = s - base
            err = (base - (s - bb)) + (off - bb)
            up = on & ((s > smax) | ((s == smax) & (err > emax)))
            smax = tl.where(up, s, smax)
            emax = tl.where(up, err, emax)
            dn = on & ((s < smin) | ((s == smin) & (err < emin)))
            smin = tl.where(dn, s, smin)
            emin = tl.where(dn, err, emin)
        shi = tl.zeros([BN], tl.float32)
        slo = tl.zeros([BN], tl.float32)
        for t in range(most):
            k = k0 + t
            on = t < deg
            m = tl.load(pm + k, mask=on, other=0)
            base = tl.load(cl + 2 * m + ax, mask=on & (m < N), other=0.0)
            off = tl.load(po + 2 * k + ax, mask=on, other=0.0)
            s = base + off
            bb = s - base
            err = (base - (s - bb)) + (off - bb)
            shi += tl.where(on, _exp(tl.fdiv((s - smax) + (err - emax), g, ieee_rounding=IEEE), LIBM), 0.0)
            slo += tl.where(on, _exp(tl.fdiv((smin - s) + (emin - err), g, ieee_rounding=IEEE), LIBM), 0.0)
        val += ((smax - smin) + (emax - emin)) + g * (_log(shi, LIBM) + _log(slo, LIBM))
        ih = tl.fdiv(w, shi, ieee_rounding=IEEE)
        il = tl.fdiv(w, slo, ieee_rounding=IEEE)
        for t in range(most):
            k = k0 + t
            on = t < deg
            m = tl.load(pm + k, mask=on, other=0)
            base = tl.load(cl + 2 * m + ax, mask=on & (m < N), other=0.0)
            off = tl.load(po + 2 * k + ax, mask=on, other=0.0)
            s = base + off
            bb = s - base
            err = (base - (s - bb)) + (off - bb)
            eh = _exp(tl.fdiv((s - smax) + (err - emax), g, ieee_rounding=IEEE), LIBM)
            el = _exp(tl.fdiv((smin - s) + (emin - err), g, ieee_rounding=IEEE), LIBM)
            if PIN == 1:
                out = tl.fma(eh, ih, -(el * il))
            elif PIN == 2:
                out = tl.fma(-el, il, eh * ih)
            else:
                out = eh * ih - el * il
            tl.store(PGW + 2 * (lane * P + k) + ax, out, mask=on)
    tl.store(VAL + lane * VW + V_WL + n, w * val, mask=live)


# The nets order[first .. first + count) of one lane as tiles [BN, DP]; every one of them has at most DP pins.
@triton.jit(do_not_specialize=_WL_INTS, do_not_specialize_on_alignment=_WL_PTRS)
def wl_pad(PGW, VAL, c, net_start, pm, po, net_w, order, N, P, VW, V_WL, first, count, g,
           BN: tl.constexpr, DP: tl.constexpr, LIBM: tl.constexpr, IEEE: tl.constexpr, PIN: tl.constexpr):
    lane = tl.program_id(1)
    j = tl.program_id(0) * BN + tl.arange(0, BN)
    live = j < count
    n = tl.load(order + first + j, mask=live, other=0)
    k0 = tl.load(net_start + n, mask=live, other=0)
    deg = tl.load(net_start + n + 1, mask=live, other=0) - k0
    w = tl.load(net_w + n, mask=live, other=0.0)
    cl = c + 2 * lane * N
    t = tl.arange(0, DP)
    k = k0[:, None] + t[None, :]
    on = t[None, :] < deg[:, None]
    m = tl.load(pm + k, mask=on, other=0)
    val = tl.zeros([BN], tl.float32)
    for ax in tl.static_range(2):
        base = tl.load(cl + 2 * m + ax, mask=on & (m < N), other=0.0)
        off = tl.load(po + 2 * k + ax, mask=on, other=0.0)
        s = base + off
        bb = s - base
        err = (base - (s - bb)) + (off - bb)
        smax = tl.max(tl.where(on, s, -3.0e38), axis=1)
        emax = tl.max(tl.where(on & (s == smax[:, None]), err, -3.0e38), axis=1)
        smin = tl.min(tl.where(on, s, 3.0e38), axis=1)
        emin = tl.min(tl.where(on & (s == smin[:, None]), err, 3.0e38), axis=1)
        eh = tl.where(on, _exp(tl.fdiv((s - smax[:, None]) + (err - emax[:, None]), g, ieee_rounding=IEEE), LIBM), 0.0)
        el = tl.where(on, _exp(tl.fdiv((smin[:, None] - s) + (emin[:, None] - err), g, ieee_rounding=IEEE), LIBM), 0.0)
        shi = tl.sum(eh, axis=1)
        slo = tl.sum(el, axis=1)
        val += ((smax - smin) + (emax - emin)) + g * (_log(shi, LIBM) + _log(slo, LIBM))
        ih = tl.fdiv(w, shi, ieee_rounding=IEEE)
        il = tl.fdiv(w, slo, ieee_rounding=IEEE)
        if PIN == 1:
            out = tl.fma(eh, ih[:, None], -(el * il[:, None]))
        elif PIN == 2:
            out = tl.fma(-el, il[:, None], eh * ih[:, None])
        else:
            out = eh * ih[:, None] - el * il[:, None]
        tl.store(PGW + 2 * (lane * P + k) + ax, out, mask=on)
    tl.store(VAL + lane * VW + V_WL + n, w * val, mask=live)

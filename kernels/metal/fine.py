"""The fine step of the lane-batched placement descent as Metal kernels (float32): loss, gradient and Adam update.

The descent optimises b independent placements ("lanes") of the same design at once. One fine step takes the macro
centres pos [b, N, 2] (microns, float32 on the MPS device) and returns, per lane, the loss and its gradient; `step`
also makes one Adam update. With c = clamp(pos) (each coordinate clamped to [half, hi] of its macro):

    loss = wirelength(c) + 0.5 density(c) + 0.5 congestion(c) + overlap_lambda overlap(c) / area + 50 boundary(pos) / area

  wirelength  per net and axis, gamma (log sum exp(x / gamma) + log sum exp(-x / gamma)) over the net's pins
              (pin = macro centre + offset; a port is a fixed pin), weighted, divided by wl_norm.
  density     every macro spreads its area over the R x C density grid as a separable Gaussian (erf masses per
              column and per row, widths max(sigma_scale half, floor)); the term is 0.5 x the mean of the dens_k
              fullest cells.
  congestion  routing demand from every source-sink pin pair (a Gaussian soft assignment of a pin to rows or
              columns, times a soft box between the two pins along the other axis), box-smoothed over
              2 smooth_range + 1 cells and divided by capacity, plus the blockage of the hard macros (the area of a
              macro inside each cell); the term is the mean of the cong_k largest of the 2 R C loads plus 0.5 x the
              mean of all.
  overlap     over pairs of hard macros, the product over both axes of max(half_i + half_j - |c_i - c_j|, 0).
  boundary    squared distance of pos outside [half, hi], per coordinate.

Inputs. `static` is a dict of numpy arrays and scalars describing the design (macro sizes and clamp bounds, the nets
grouped by degree, the grids, the source-sink pairs, the loss weights and the Adam constants; the keys are read in
`__init__`). `sched` is a named tuple of the values that change with the step index: gamma, sigma_scale,
sigma_floor_x, sigma_floor_y, overlap_lambda and adam_t (the 1-based number of the Adam update). Lanes are
independent; nothing is carried from one call to the next (every scratch buffer is written in full before it is read
inside a call).

Gradients are derived by hand, not by autograd: each kernel writes a term's weighted value and its gradient with
respect to the clamped positions directly, in gather form (one thread per output, no atomics, no colliding writes),
and `combine` sums them per macro, applies the clamp's gradient (1 inside and on a bound, 0 outside), adds the
boundary gradient and multiplies by the movable mask. Where the loss has a kink the kernels take the subgradients of
the torch code they replace (min and max split an exact tie, a clamp passes at its bound, |x| has gradient 0 at 0, a
top-k mean gives 1/k to the selected cells).

Precision. Everything is float32. The kernels are gated against a float64 CPU reference of the same loss: the
gradient error is held to 1e-5 of the largest gradient entry (entries at a kink, where float32 and float64 can take
different sides, are judged apart). Float32 care: positions plus offsets are two-sums (value and rounding error),
log-sum-exps are taken relative to the extreme pin, per-macro sums are compensated (Kahan), and the soft box is
written without the cancellation of 1 - 1.

`loss_and_grad` is 19 kernel launches and 5 torch operations; `step` adds the Adam update:

  needs only the positions   prep  wl_net  dfac3  ranges  assign_rows  box_c  box_r  bfac  ovl_part
  grids and selection        bmm (density grid)  bmm (pair product)  bmm (blockage grids)  tail5  topk  setk5
  gradients                  dbits2  dgradw2  cot  pair_grad  macro_sum  bgrad_part  hsum  combine  sum (loss)
  update (step only)         adam

Every weighted value goes to one buffer VAL [b, nets + hard + N + dens_k + R C]; the loss is one sum over it.
"""
from __future__ import annotations

import functools
import math
import types

import numpy as np
import torch
import torch.mps.event

TERMS = ("wirelength", "density", "congestion", "overlap", "boundary")
BIT = {"wirelength": 1, "density": 2, "congestion": 4, "overlap": 8, "boundary": 16}
ALL = sum(BIT.values())

ROWMAX = 64      # a grid side of at most 64 cells: a row of the density selection is held in two 32-bit words
FAR = 5.0        # the density gradient of a macro is gathered within FAR widths (sigma sqrt 2) plus one cell
OCH = 32         # hard macros per thread of ovl_part
GROUP_MAX = 16   # pairs per row of a grouped pair factor at most (a pin with more pairs gets several rows)
TOL_ASSIGN = 1.0e-10   # first omitted soft-assignment weight, relative to the largest weight of the row
TOL_BOX = 5.0e-9       # soft box (and its derivative) at the first omitted cell

# Threadgroup sizes. A kernel lasts as long as its slowest thread; kernels whose threads loop run faster in groups
# of 64 than in torch's default group of 1,024. The size does not change any thread's arithmetic.
LOOPING = ("wl_net", "dgradw2", "ovl_part", "bgrad_part", "hsum", "pair_grad", "macro_sum", "combine", "adam", "dbits2",
           "ranges", "assign_rows")
GROUP_LOOPING, GROUP_OTHER = 64, 256

# ---- everything except the pair accumulation -------------------------------------------------------------------------
# ip: 0 N, 1 NH, 2 R, 3 C, 4 B, 5 NN (merged nets), 6 KD, 7 KC, 8 SR, 9 VW (width of VAL), 10 V_WL, 11 V_OV, 12 V_BD,
#     13 V_DN, 14 V_CG, 15 P (pin slots)
# cf: 0 v_alloc / v_capacity, 1 h_alloc / h_capacity, 2 cell_w, 3 cell_h, 4 x of the first cell edge, 5 y of it
STEP_SHADER = r"""
#include <metal_stdlib>
#include <c10/metal/special_math.h>
using namespace metal;
#define FARW @FARW@
#define OCH @OCH@

static inline float2 two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    return float2(s, (a - (s - bb)) + (b - bb));
}

// the clamped positions, a thread per (lane, macro)
kernel void prep(device float* c [[buffer(0)]], device const float* pos [[buffer(1)]], device const float* lo [[buffer(2)]],
                 device const float* hi [[buffer(3)]], device const int* ip [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint m = i % uint(ip[0]);
    c[2 * i] = min(max(pos[2 * i], lo[2 * m]), hi[2 * m]);
    c[2 * i + 1] = min(max(pos[2 * i + 1], lo[2 * m + 1]), hi[2 * m + 1]);
}

// one coordinate of pin slot k: float32 sum position + offset and its rounding error; a port (macro N) sits at its offset
static inline float2 pin(device const float* c, device const int* pm, device const float* po, int lane, int k, int N, int ax) {
    int m = pm[k];
    float base = 0.0f;
    if (m < N) { base = c[2 * (lane * N + m) + ax]; }
    return two_sum(base, po[2 * k + ax]);
}

// wirelength, a thread per (lane, net). fp: gamma. Value w (max - min + gamma (log S_hi + log S_lo)) per axis, with
// S_hi = sum exp((x - max) / gamma), S_lo = sum exp((min - x) / gamma); pin gradient w (e_hi / S_hi - e_lo / S_lo).
// Each pin is (value, rounding error), so differences are taken on both parts before the division by gamma.
kernel void wl_net(device float* PGW [[buffer(0)]], device float* VAL [[buffer(1)]], device const float* c [[buffer(2)]],
                   device const int* net_start [[buffer(3)]], device const int* pm [[buffer(4)]], device const float* po [[buffer(5)]],
                   device const float* net_w [[buffer(6)]], device const int* ip [[buffer(7)]], constant float* fp [[buffer(8)]],
                   uint i [[thread_position_in_grid]]) {
    int N = ip[0], NN = ip[5], P = ip[15];
    int lane = int(i) / NN, n = int(i) - lane * NN;
    int k0 = net_start[n], k1 = net_start[n + 1];
    float g = fp[0], w = net_w[n], val = 0.0f;
    for (int ax = 0; ax < 2; ax++) {
        float smax = -3.0e38f, emax = 0.0f, smin = 3.0e38f, emin = 0.0f;
        for (int k = k0; k < k1; k++) {
            float2 t = pin(c, pm, po, lane, k, N, ax);
            if (t.x > smax || (t.x == smax && t.y > emax)) { smax = t.x; emax = t.y; }
            if (t.x < smin || (t.x == smin && t.y < emin)) { smin = t.x; emin = t.y; }
        }
        float shi = 0.0f, slo = 0.0f;
        for (int k = k0; k < k1; k++) {
            float2 t = pin(c, pm, po, lane, k, N, ax);
            shi += exp(((t.x - smax) + (t.y - emax)) / g);
            slo += exp(((smin - t.x) + (emin - t.y)) / g);
        }
        val += ((smax - smin) + (emax - emin)) + g * (log(shi) + log(slo));
        float ih = w / shi, il = w / slo;
        for (int k = k0; k < k1; k++) {
            float2 t = pin(c, pm, po, lane, k, N, ax);
            PGW[2 * (lane * P + k) + ax] = exp(((t.x - smax) + (t.y - emax)) / g) * ih - exp(((smin - t.x) + (emin - t.y)) / g) * il;
        }
    }
    VAL[lane * ip[9] + ip[10] + n] = w * val;
}

// density factors, a thread per (lane, macro, e in C + R). fp: sigma_scale, floor_x, floor_y.
// FX [B, N, C] mass per column, DX its derivative by x; with q = charge / cell_area: FYT [B, R, N] = q x mass per row
// (the left factor of the grid product), FY [B, N, R] the same in macro order, DY [B, N, R] = q x its derivative by y.
kernel void dfac3(device float* FX [[buffer(0)]], device float* DX [[buffer(1)]], device float* FYT [[buffer(2)]],
                  device float* DY [[buffer(3)]], device float* FY [[buffer(4)]], device const float* c [[buffer(5)]],
                  device const float* half_ [[buffer(6)]], device const float* chq [[buffer(7)]], device const float* xlo [[buffer(8)]],
                  device const float* xhi [[buffer(9)]], device const float* ylo [[buffer(10)]], device const float* yhi [[buffer(11)]],
                  device const int* ip [[buffer(12)]], constant float* fp [[buffer(13)]], uint i [[thread_position_in_grid]]) {
    int N = ip[0], R = ip[2], C = ip[3];
    int E = R + C;
    int lane = int(i) / (N * E), rem = int(i) - lane * (N * E);
    int n = rem / E, e = rem - n * E;
    int ln = lane * N + n;
    if (e < C) {
        float x = c[2 * ln];
        float w = max(fp[0] * half_[2 * n], fp[1]) * 1.41421356237309504880f;
        float tl = (xlo[e] - x) / w, th = (xhi[e] - x) / w;
        FX[ln * C + e] = 0.5f * (c10::metal::erf(th) - c10::metal::erf(tl));
        DX[ln * C + e] = (exp(-tl * tl) - exp(-th * th)) * (0.56418958354775628695f / w);
    } else {
        int r = e - C;
        float y = c[2 * ln + 1], q = chq[n];
        float w = max(fp[0] * half_[2 * n + 1], fp[2]) * 1.41421356237309504880f;
        float tl = (ylo[r] - y) / w, th = (yhi[r] - y) / w;
        float mass = q * (0.5f * (c10::metal::erf(th) - c10::metal::erf(tl)));
        FYT[(lane * R + r) * N + n] = mass;
        FY[ln * R + r] = mass;
        DY[ln * R + r] = q * ((exp(-tl * tl) - exp(-th * th)) * (0.56418958354775628695f / w));
    }
}

// macro blockage factors, a thread per (lane, hard macro, e in C + R), in the order of operations of the torch code
// and with its subgradients (min and max split a tie, clamps pass on the bound), so that a macro edge exactly on a
// cell edge takes the same decision.
// BR [2 B, NH, C]: lanes 0..B-1 x_ol (for V), B..2B-1 x_present (for H). BL [2 B, R, NH]: y_present (V), y_ol (H).
// DBX [B, NH, C, 2] = d(x_ol, x_present)/dx. DBY [B, NH, R, 2] = d(y_present, y_ol)/dy.
kernel void bfac(device float* BL [[buffer(0)]], device float* BR [[buffer(1)]], device float* DBX [[buffer(2)]],
                 device float* DBY [[buffer(3)]], device const float* c [[buffer(4)]], device const float* half_ [[buffer(5)]],
                 device const float* xlo [[buffer(6)]], device const float* xhi [[buffer(7)]], device const float* ylo [[buffer(8)]],
                 device const float* yhi [[buffer(9)]], device const int* ip [[buffer(10)]], device const float* cf [[buffer(11)]],
                 uint i [[thread_position_in_grid]]) {
    int N = ip[0], NH = ip[1], R = ip[2], C = ip[3], B = ip[4];
    int E = R + C;
    int lane = int(i) / (NH * E), rem = int(i) - lane * (NH * E);
    int n = rem / E, e = rem - n * E;
    bool isx = e < C;
    int k = isx ? e : e - C;
    float u = c[2 * (lane * N + n) + (isx ? 0 : 1)], h = half_[2 * n + (isx ? 0 : 1)];
    float cell = isx ? cf[2] : cf[3];
    float emax = isx ? xhi[k] : yhi[k], emin = isx ? xlo[k] : ylo[k];
    float a = u + h, b = u - h;
    float top = min(a, emax), bot = max(b, emin);
    float dtop = a < emax ? 1.0f : (a == emax ? 0.5f : 0.0f);
    float dbot = b > emin ? 1.0f : (b == emin ? 0.5f : 0.0f);
    float d = top - bot;
    float ol = max(d, 0.0f), dol = d >= 0.0f ? dtop - dbot : 0.0f;
    float t = ol / cell;
    float p = min(t, 1.0f), dp = t <= 1.0f ? dol / cell : 0.0f;
    if (isx) {
        BR[(lane * NH + n) * C + k] = ol; BR[((B + lane) * NH + n) * C + k] = p;
        DBX[2 * ((lane * NH + n) * C + k)] = dol; DBX[2 * ((lane * NH + n) * C + k) + 1] = dp;
    } else {
        BL[(lane * R + k) * NH + n] = p; BL[((B + lane) * R + k) * NH + n] = ol;
        DBY[2 * ((lane * NH + n) * R + k)] = dp; DBY[2 * ((lane * NH + n) * R + k) + 1] = dol;
    }
}

// overlap, partial sums: a thread per (lane, hard macro a, chunk q of OCH other hard macros). One thread per macro
// would loop over all NH others; short threads finish together. Value over j > a only, gradient over all j.
// OP [B, NH, Q, 3] = (gx, gy, value). jq: 0 Q (chunks), 1 RQ (rows of the largest blockage window)
kernel void ovl_part(device float* OP [[buffer(0)]], device const float* c [[buffer(1)]], device const float* half_ [[buffer(2)]],
                     device const int* ip [[buffer(3)]], device const int* jq [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    int N = ip[0], NH = ip[1], Q = jq[0];
    int t = int(i) / Q, q = int(i) - t * Q;
    int lane = t / NH, a = t - lane * NH;
    int row = 2 * lane * N;
    float xi = c[row + 2 * a], yi = c[row + 2 * a + 1], hxi = half_[2 * a], hyi = half_[2 * a + 1];
    float gx = 0.0f, gy = 0.0f, val = 0.0f;
    int j1 = min(NH, (q + 1) * OCH);
    for (int j = q * OCH; j < j1; j++) {
        if (j == a) { continue; }
        float dx = c[row + 2 * j] - xi, dy = c[row + 2 * j + 1] - yi;
        float px = (hxi + half_[2 * j]) - fabs(dx), py = (hyi + half_[2 * j + 1]) - fabs(dy);
        float ox = max(px, 0.0f), oy = max(py, 0.0f);
        if (j > a) { val += ox * oy; }
        if (px >= 0.0f) { gx += oy * sign(dx); }
        if (py >= 0.0f) { gy += ox * sign(dy); }
    }
    OP[3 * i] = gx; OP[3 * i + 1] = gy; OP[3 * i + 2] = val;
}

// congestion load, a thread per (lane, cell), written into the stacked top-k input TK [2 B, 2 R C]: rows 0..B-1 the
// load (vertical then horizontal), rows B..2B-1 the density grid and then -3e38, so that one torch.topk selects both
// the congestion cells and the density cells. Also zeroes the congestion mask and the density mask DM [B, R C].
kernel void tail5(device float* TK [[buffer(0)]], device float* MASK [[buffer(1)]], device float* DM [[buffer(2)]],
                  device const float* V [[buffer(3)]], device const float* H [[buffer(4)]], device const float* MB [[buffer(5)]],
                  device const float* D [[buffer(6)]], device const float* icv [[buffer(7)]], device const float* ich [[buffer(8)]],
                  device const int* ip [[buffer(9)]], device const float* cf [[buffer(10)]], uint i [[thread_position_in_grid]]) {
    int R = ip[2], C = ip[3], B = ip[4], SR = ip[8];
    int RC = R * C;
    int lane = int(i) / RC, cell = int(i) - lane * RC;
    int r = cell / C, cc = cell - r * C;
    float lv = 0.0f, lh = 0.0f;
    for (int s = -SR; s <= SR; s++) {
        int c2 = cc + s, r2 = r + s;
        if (c2 >= 0 && c2 < C) { lv += V[lane * RC + r * C + c2] * icv[r * C + c2]; }
        if (r2 >= 0 && r2 < R) { lh += H[lane * RC + r2 * C + cc] * ich[r2 * C + cc]; }
    }
    TK[lane * 2 * RC + cell] = lv + cf[0] * MB[lane * RC + cell];
    TK[lane * 2 * RC + RC + cell] = lh + cf[1] * MB[(B + lane) * RC + cell];
    TK[(B + lane) * 2 * RC + cell] = D[i];
    TK[(B + lane) * 2 * RC + RC + cell] = -3.0e38f;
    MASK[lane * 2 * RC + cell] = 0.0f;
    MASK[lane * 2 * RC + RC + cell] = 0.0f;
    DM[i] = 0.0f;
}

// after the stacked top-k, a thread per (row of TK, j < KM): a load row sets the congestion mask of its j-th cell
// (j < KC); a density row writes its j-th value into VAL and sets the density mask (j < KD).
// jp: 0 B, 1 KM (width of a row of idx). fp: density weight / KD
kernel void setk5(device float* MASK [[buffer(0)]], device float* VAL [[buffer(1)]], device float* DM [[buffer(2)]],
                  device const float* vals [[buffer(3)]], device const long* idx [[buffer(4)]], device const int* ip [[buffer(5)]],
                  device const int* jp [[buffer(6)]], constant float* fp [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    int R = ip[2], C = ip[3], B = ip[4], KD = ip[6], KC = ip[7], KM = jp[1];
    int row = int(i) / KM, j = int(i) - row * KM;
    int cell = int(idx[i]);
    if (row < B) {
        if (j < KC) { MASK[uint(row) * uint(2 * R * C) + uint(cell)] = 1.0f; }
    } else if (j < KD) {
        int lane = row - B;
        VAL[lane * ip[9] + ip[13] + j] = fp[0] * vals[i];
        DM[lane * R * C + cell] = 1.0f;
    }
}

// the selected density cells of a (lane, row) as bits, a thread per (lane, row, 32-column word)
kernel void dbits2(device uint* BITS [[buffer(0)]], device const float* DM [[buffer(1)]], device const int* ip [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    int C = ip[3];
    int w = int(i & 1u), t = int(i >> 1u);
    int c0 = 32 * w, c1 = min(C, c0 + 32);
    uint bits = 0u;
    for (int cc = c0; cc < c1; cc++) {
        if (DM[t * C + cc] != 0.0f) { bits |= 1u << uint(cc - c0); }
    }
    BITS[i] = bits;
}

static inline uint range32(int a, int b) {        // bits a..b of a 32-bit word, empty when the range misses the word
    a = max(a, 0); b = min(b, 31);
    if (b < a) { return 0u; }
    uint upto = b == 31 ? 0xFFFFFFFFu : ((1u << uint(b + 1)) - 1u);
    return upto & ~((1u << uint(a)) - 1u);
}

// the density gradient of one macro, gathered from the selected cells inside its window only: a cell further than
// FAR widths from the macro on an axis carries a mass below 1e-12 and a derivative below exp(-FAR^2) of its peak. The
// window's rows are visited, and in each row only the set bits inside the window's columns (count trailing zeros), so
// the longest thread is the window's rows plus the selected cells, not the window's area. Compensated sum.
// fp: 0 weight / KD, 1 sigma_scale, 2 floor_x, 3 floor_y.  gf: x and y of the first cell edge, cell width, cell height
static inline float2 dens_grad2(device const float* FX, device const float* DX, device const float* FY, device const float* DY,
                                device const uint* BITS, device const float* c, device const float* half_, device const int* ip,
                                device const float* gf, constant float* fp, int i) {
    int N = ip[0], R = ip[2], C = ip[3];
    int lane = i / N, n = i - lane * N;
    float x = c[2 * i], y = c[2 * i + 1];
    float wx = max(fp[1] * half_[2 * n], fp[2]) * FARW, wy = max(fp[1] * half_[2 * n + 1], fp[3]) * FARW;
    int c_lo = clamp(int(floor((x - wx - gf[0]) / gf[2])) - 1, 0, C - 1), c_hi = clamp(int(floor((x + wx - gf[0]) / gf[2])) + 1, 0, C - 1);
    int r_lo = clamp(int(floor((y - wy - gf[1]) / gf[3])) - 1, 0, R - 1), r_hi = clamp(int(floor((y + wy - gf[1]) / gf[3])) + 1, 0, R - 1);
    uint mlo = range32(c_lo, c_hi), mhi = range32(c_lo - 32, c_hi - 32);
    int fb = i * C, rb = i * R;
    float2 s = float2(0.0f), comp = float2(0.0f);
    for (int r = r_lo; r <= r_hi; r++) {
        uint lo = BITS[2 * (lane * R + r)] & mlo, hi = BITS[2 * (lane * R + r) + 1] & mhi;
        if ((lo | hi) == 0u) { continue; }
        float fy = FY[rb + r], dy = DY[rb + r];
        while (lo != 0u) {
            int cc = int(ctz(lo));
            lo &= lo - 1u;
            float2 v = float2(fy * DX[fb + cc], dy * FX[fb + cc]) - comp;
            float2 t = s + v;
            comp = (t - s) - v;
            s = t;
        }
        while (hi != 0u) {
            int cc = 32 + int(ctz(hi));
            hi &= hi - 1u;
            float2 v = float2(fy * DX[fb + cc], dy * FX[fb + cc]) - comp;
            float2 t = s + v;
            comp = (t - s) - v;
            s = t;
        }
    }
    return fp[0] * s;
}

// density gradient, a thread per (lane, macro)
kernel void dgradw2(device float* GD [[buffer(0)]], device const float* FX [[buffer(1)]], device const float* DX [[buffer(2)]],
                    device const float* FY [[buffer(3)]], device const float* DY [[buffer(4)]], device const uint* BITS [[buffer(5)]],
                    device const float* c [[buffer(6)]], device const float* half_ [[buffer(7)]], device const int* ip [[buffer(8)]],
                    device const float* gf [[buffer(9)]], constant float* fp [[buffer(10)]], uint i [[thread_position_in_grid]]) {
    float2 g = dens_grad2(FX, DX, FY, DY, BITS, c, half_, ip, gf, fp, int(i));
    GD[2 * i] = g.x; GD[2 * i + 1] = g.y;
}

// congestion cotangents, a thread per (lane, cell). fp: ct = weight / KC, cm = weight x mean weight / (2 R C).
// d(term)/d(load) = ct mask + cm; GV, GH are its transposed box sums over the capacity (inputs of pair_grad).
// Also writes the cell's share of the congestion value.
kernel void cot(device float* GV [[buffer(0)]], device float* GH [[buffer(1)]], device float* VAL [[buffer(2)]],
                device const float* LOAD [[buffer(3)]], device const float* MASK [[buffer(4)]], device const float* icv [[buffer(5)]],
                device const float* ich [[buffer(6)]], device const int* ip [[buffer(7)]], constant float* fp [[buffer(8)]],
                uint i [[thread_position_in_grid]]) {
    int R = ip[2], C = ip[3], SR = ip[8];
    int RC = R * C;
    int lane = int(i) / RC, cell = int(i) - lane * RC;
    int r = cell / C, cc = cell - r * C;
    int base = lane * 2 * RC;
    float ct = fp[0], cm = fp[1], gv = 0.0f, gh = 0.0f;
    for (int s = -SR; s <= SR; s++) {
        int c2 = cc + s, r2 = r + s;
        if (c2 >= 0 && c2 < C) { gv += ct * MASK[base + r * C + c2] + cm; }
        if (r2 >= 0 && r2 < R) { gh += ct * MASK[base + RC + r2 * C + cc] + cm; }
    }
    GV[i] = gv * icv[cell]; GH[i] = gh * ich[cell];
    float lv = LOAD[base + cell], lh = LOAD[base + RC + cell];
    VAL[lane * ip[9] + ip[14] + cell] = ct * (MASK[base + cell] * lv + MASK[base + RC + cell] * lh) + cm * (lv + lh);
}

// blockage gradient, partial sums: a thread per (lane, hard macro, row q of the macro's window of cells, one cell of
// margin), row r_lo + q; rows past the window write zero. BP [B, NH, RQ, 2]. fp as cot: ct, cm
kernel void bgrad_part(device float* BP [[buffer(0)]], device const float* MASK [[buffer(1)]], device const float* BL [[buffer(2)]],
                       device const float* BR [[buffer(3)]], device const float* DBX [[buffer(4)]], device const float* DBY [[buffer(5)]],
                       device const float* c [[buffer(6)]], device const float* half_ [[buffer(7)]], device const int* ip [[buffer(8)]],
                       device const float* cf [[buffer(9)]], device const int* jq [[buffer(10)]], constant float* fp [[buffer(11)]],
                       uint i [[thread_position_in_grid]]) {
    int N = ip[0], NH = ip[1], R = ip[2], C = ip[3], B = ip[4], RQ = jq[1];
    int RC = R * C;
    int t = int(i) / RQ, q = int(i) - t * RQ;
    int lane = t / NH, n = t - lane * NH;
    float x = c[2 * (lane * N + n)], y = c[2 * (lane * N + n) + 1], hx = half_[2 * n], hy = half_[2 * n + 1];
    int c_lo = clamp(int(floor((x - hx - cf[4]) / cf[2])) - 1, 0, C - 1), c_hi = clamp(int(floor((x + hx - cf[4]) / cf[2])) + 1, 0, C - 1);
    int r_lo = clamp(int(floor((y - hy - cf[5]) / cf[3])) - 1, 0, R - 1), r_hi = clamp(int(floor((y + hy - cf[5]) / cf[3])) + 1, 0, R - 1);
    float ct = fp[0], cm = fp[1], kv = cf[0], kh = cf[1];
    int base = lane * 2 * RC, ln = lane * NH + n;
    float gx = 0.0f, gy = 0.0f;
    int r = r_lo + q;
    if (r <= r_hi) {
        float yp = BL[(lane * R + r) * NH + n], yol = BL[((B + lane) * R + r) * NH + n];
        float dyp = DBY[2 * (ln * R + r)], dyol = DBY[2 * (ln * R + r) + 1];
        for (int cc = c_lo; cc <= c_hi; cc++) {
            float lv = kv * (ct * MASK[base + r * C + cc] + cm), lh = kh * (ct * MASK[base + RC + r * C + cc] + cm);
            float xol = BR[ln * C + cc], xp = BR[((B + lane) * NH + n) * C + cc];
            float dxol = DBX[2 * (ln * C + cc)], dxp = DBX[2 * (ln * C + cc) + 1];
            gx += lv * yp * dxol + lh * yol * dxp;
            gy += lv * dyp * xol + lh * dyol * xp;
        }
    }
    BP[2 * i] = gx; BP[2 * i + 1] = gy;
}

// the partial sums of a hard macro, a thread per (lane, hard macro): overlap value and gradient, blockage gradient.
// fp: overlap_lambda / area
kernel void hsum(device float* GO [[buffer(0)]], device float* GB [[buffer(1)]], device float* VAL [[buffer(2)]],
                 device const float* OP [[buffer(3)]], device const float* BP [[buffer(4)]], device const int* ip [[buffer(5)]],
                 device const int* jq [[buffer(6)]], constant float* fp [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    int NH = ip[1], Q = jq[0], RQ = jq[1];
    int lane = int(i) / NH, a = int(i) - lane * NH;
    float gx = 0.0f, gy = 0.0f, val = 0.0f;
    for (int q = 0; q < Q; q++) { gx += OP[3 * (int(i) * Q + q)]; gy += OP[3 * (int(i) * Q + q) + 1]; val += OP[3 * (int(i) * Q + q) + 2]; }
    GO[2 * i] = fp[0] * gx; GO[2 * i + 1] = fp[0] * gy;
    VAL[lane * ip[9] + ip[11] + a] = fp[0] * val;
    float bx = 0.0f, by = 0.0f;
    for (int q = 0; q < RQ; q++) { bx += BP[2 * (int(i) * RQ + q)]; by += BP[2 * (int(i) * RQ + q) + 1]; }
    GB[2 * i] = bx; GB[2 * i + 1] = by;
}

// the gradient, a thread per (lane, macro): compensated CSR sum of the macro's wirelength pin gradients, plus the
// density, congestion, blockage and overlap gradients, times the clamp's gradient, plus the boundary gradient, times
// the movable mask. fp: 0 the terms to include (bits: 1 wirelength, 2 density, 4 congestion, 8 overlap,
// 16 boundary), 1 weight_boundary / area. With bit 16 it also writes the boundary values.
kernel void combine(device float* G [[buffer(0)]], device float* VAL [[buffer(1)]], device const float* pos [[buffer(2)]],
                    device const float* lo [[buffer(3)]], device const float* hi [[buffer(4)]], device const float* mov [[buffer(5)]],
                    device const float* PGW [[buffer(6)]], device const int* mstart [[buffer(7)]], device const int* mentry [[buffer(8)]],
                    device const float* GD [[buffer(9)]], device const float* GC [[buffer(10)]], device const float* GB [[buffer(11)]],
                    device const float* GO [[buffer(12)]], device const int* ip [[buffer(13)]], constant float* fp [[buffer(14)]],
                    uint i [[thread_position_in_grid]]) {
    int N = ip[0], NH = ip[1], P = ip[15];
    int lane = int(i) / N, m = int(i) - lane * N;
    int sel = int(fp[0]);
    float2 g = float2(0.0f);
    if (sel & 1) {
        float2 s = float2(0.0f), comp = float2(0.0f);
        uint base = 2u * uint(lane) * uint(P);
        for (int k = mstart[m]; k < mstart[m + 1]; k++) {
            uint e = base + 2u * uint(mentry[k]);
            float2 y = float2(PGW[e], PGW[e + 1]) - comp;
            float2 t = s + y;
            comp = (t - s) - y;
            s = t;
        }
        g = s;
    }
    if (sel & 2) { g += float2(GD[2 * i], GD[2 * i + 1]); }
    if (sel & 4) {
        g += float2(GC[2 * i], GC[2 * i + 1]);
        if (m < NH) { g += float2(GB[2 * (lane * NH + m)], GB[2 * (lane * NH + m) + 1]); }
    }
    if ((sel & 8) && m < NH) { g += float2(GO[2 * (lane * NH + m)], GO[2 * (lane * NH + m) + 1]); }
    float2 p = float2(pos[2 * i], pos[2 * i + 1]);
    float2 l = float2(lo[2 * m], lo[2 * m + 1]), h = float2(hi[2 * m], hi[2 * m + 1]);
    g.x = (p.x >= l.x && p.x <= h.x) ? g.x : 0.0f;
    g.y = (p.y >= l.y && p.y <= h.y) ? g.y : 0.0f;
    if (sel & 16) {
        float2 under = max(l - p, float2(0.0f)), over = max(p - h, float2(0.0f));
        g += (2.0f * fp[1]) * (over - under);
        VAL[lane * ip[9] + ip[12] + m] = fp[1] * (under.x * under.x + over.x * over.x + under.y * under.y + over.y * over.y);
    }
    G[2 * i] = g.x * mov[m]; G[2 * i + 1] = g.y * mov[m];
}

// the Adam update, a thread per coordinate, in place.
// fp: 1 - beta1, beta2, 1 - beta2, lr / (1 - beta1^t), sqrt(1 - beta2^t), eps
kernel void adam(device float* pos [[buffer(0)]], device float* m [[buffer(1)]], device float* v [[buffer(2)]],
                 device const float* g [[buffer(3)]], constant float* fp [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    float gi = g[i];
    float mi = m[i] + fp[0] * (gi - m[i]);
    float vi = fp[1] * v[i] + fp[2] * (gi * gi);
    m[i] = mi; v[i] = vi;
    pos[i] = pos[i] - fp[3] * (mi / (sqrt(vi) / fp[4] + fp[5]));
}
"""

# ---- the pair accumulation of the congestion term ---------------------------------------------------------------------
# For each source-sink pair p with weight w_p (a pin = macro centre + offset):
#     V[r, c] = sum_p w_p box_rows_p[r] A_x(sink_p)[c],    H[r, c] = sum_p w_p A_y(source_p)[r] box_cols_p[c]
# A_x, A_y: Gaussian soft assignment of a pin coordinate to the cell centres; box: sigmoid(low edge) - sigmoid(high
# edge) of the run between the pair's two soft-assigned expected indices (soft min and max).
# Three facts keep it cheap. Locality: a soft assignment is above float32 round-off on 2 WA + 1 cells around the pin,
# a box on its run plus DL cells below and DH above; every sum over a grid side runs over that window only.
# Repetition: pairs with the same source pin and sink pin are merged (weights added), and pairs that share a source
# (sink) pin share its soft-assignment row, so H (V) is ONE batched matrix product over pin groups ("group rows"),
# not over pairs. Gather: the backward recomputes each pair's scalars and gathers the two cotangent grids inside
# the pair's windows; nothing dense is stored by the forward.
#
# ip: M (merged pairs), N, R, C, G (rows of the grouped factors), B (lanes of this call)
# fp: 1/sigma_x, 1/sigma_y, beta, 1/beta, rho, 1/cell_w, 1/cell_h, 2/sigma_x^2, 2/sigma_y^2
PAIR_SHADER = r"""
#include <metal_stdlib>
using namespace metal;
#define WA @WA@
#define NW (2 * WA + 1)
#define DL @DL@
#define DH @DH@

// one pin coordinate against a row of cell centres
struct Soft {
    int k0;          // nearest cell
    float f;         // expected index minus k0
    float ckm;       // sum_k A_k (k - k0) (dm_k - md): covariance of index and centre under A
    float md;        // sum_k A_k dm_k
    float a[NW];     // A_k for k = k0 - WA .. k0 + WA (0 outside the grid)
    float dm[NW];    // centre_k - centre_k0
};

// the weights are formed relative to the nearest cell (exp(z_k - z_k0)), so the largest is 1 and none overflows
static inline Soft soft_assign(float u, float ue, device const float* ctr, int K, float inv_cell, float inv_sigma) {
    Soft s;
    int k0 = clamp(int(floor((u - ctr[0]) * inv_cell + 0.5f)), 0, K - 1);
    float d0 = fabs((u - ctr[k0]) + ue);
    if (k0 + 1 < K && fabs((u - ctr[k0 + 1]) + ue) < d0) { k0 += 1; }
    else if (k0 > 0 && fabs((u - ctr[k0 - 1]) + ue) < d0) { k0 -= 1; }
    float m0 = ctr[k0];
    float t0 = ((u - m0) + ue) * inv_sigma;
    float sum = 0.0f;
    for (int j = 0; j < NW; j++) {
        int k = k0 + j - WA;
        float e = 0.0f, dm = 0.0f;
        if (k >= 0 && k < K) {
            dm = ctr[k] - m0;
            float a = dm * inv_sigma;             // t0 - t_k
            e = exp(a * (2.0f * t0 - a));         // exp(t0^2 - t_k^2) = exp(z_k - z_k0)
        }
        s.a[j] = e; s.dm[j] = dm; sum += e;
    }
    float inv = 1.0f / sum;
    float f = 0.0f, md = 0.0f;
    for (int j = 0; j < NW; j++) { s.a[j] *= inv; f += s.a[j] * float(j - WA); md += s.a[j] * s.dm[j]; }
    float ckm = 0.0f;
    for (int j = 0; j < NW; j++) { ckm += s.a[j] * float(j - WA) * (s.dm[j] - md); }
    s.k0 = k0; s.f = f; s.md = md; s.ckm = ckm;
    return s;
}

// soft minimum and maximum of two expected indices, each given as (integer cell, fraction): the integer parts are
// kept apart so that the sigmoid arguments do not lose the fraction to the magnitude of the cell index
struct Range {
    int k_lo; float off_lo;   // sigmoid argument of the low edge at cell c: rho * (float(c - k_lo) + off_lo)
    int k_hi; float off_hi;
    float q;                  // d(min)/d(first) = d(max)/d(second); 1 - q the other two
    float q1;                 // 1 - q, computed without cancellation
};

static inline Range soft_range(int ka, float fa, int kb, float fb, float beta, float inv_beta) {
    Range r;
    float delta = float(kb - ka) + (fb - fa);              // second minus first
    float a = exp(-beta * fabs(delta));
    float corr = (a < 1.0e-4f ? a * (1.0f - 0.5f * a) : log(1.0f + a)) * inv_beta;
    bool first_low = delta >= 0.0f;
    r.k_lo = first_low ? ka : kb; r.k_hi = first_low ? kb : ka;
    r.off_lo = (0.5f - (first_low ? fa : fb)) + corr;
    r.off_hi = (0.5f - (first_low ? fb : fa)) - corr;
    float big = 1.0f / (1.0f + a), small = a * big;
    r.q = first_low ? big : small; r.q1 = first_low ? small : big;
    return r;
}

// sigmoid(x1) - sigmoid(x2) for x1 > x2, without the cancellation of 1 - 1
static inline float box(float x1, float x2) {
    float a1 = exp(-fabs(x1)), a2 = exp(-fabs(x2));
    float num = x2 >= 0.0f ? a2 - a1 : (x1 <= 0.0f ? a1 - a2 : 1.0f - a1 * a2);
    return num / ((1.0f + a1) * (1.0f + a2));
}

// the box and the derivatives of its two edges, sigmoid'(x) = a / (1 + a)^2 with a = exp(-|x|), from two exponentials
struct Edge { float bx, dlo, dhi; };

static inline Edge edges(float x1, float x2) {
    float a1 = exp(-fabs(x1)), a2 = exp(-fabs(x2));
    float i1 = 1.0f / (1.0f + a1), i2 = 1.0f / (1.0f + a2);
    float num = x2 >= 0.0f ? a2 - a1 : (x1 <= 0.0f ? a1 - a2 : 1.0f - a1 * a2);
    Edge e;
    e.bx = num * i1 * i2; e.dlo = a1 * i1 * i1; e.dhi = a2 * i2 * i2;
    return e;
}

struct Pins { float sx, sxe, sy, sye, tx, txe, ty, tye; };

static inline float2 two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    return float2(s, (a - (s - bb)) + (b - bb));
}

// one coordinate (axis 0: x, 1: y) of pin u in a lane: the float32 sum position + offset and its rounding error.
// A pin of a port (macro index N) sits at its offset.
static inline float2 pin_coord(device const float* pos, device const int* pin_macro, device const float* pin_off,
                               int lane, int u, int N, int axis) {
    int m = pin_macro[u];
    float base = 0.0f;
    if (m < N) { base = pos[2u * (uint(lane) * uint(N) + uint(m)) + uint(axis)]; }
    return two_sum(base, pin_off[2 * u + axis]);
}

static inline Pins pins(device const float* pos, device const int* src_pin, device const int* snk_pin,
                        device const int* pin_macro, device const float* pin_off, int lane, int p, int N) {
    int us = src_pin[p], ut = snk_pin[p];
    float2 a = pin_coord(pos, pin_macro, pin_off, lane, us, N, 0), b = pin_coord(pos, pin_macro, pin_off, lane, us, N, 1);
    float2 c = pin_coord(pos, pin_macro, pin_off, lane, ut, N, 0), d = pin_coord(pos, pin_macro, pin_off, lane, ut, N, 1);
    Pins r; r.sx = a.x; r.sxe = a.y; r.sy = b.x; r.sye = b.y; r.tx = c.x; r.txe = c.y; r.ty = d.x; r.tye = d.y;
    return r;
}

// forward 1, a thread per (lane, pair): the two runs of the pair as (integer end cells, sigmoid offsets).
// RI, RF [B, M, 4]: columns low, high, then rows low, high.
kernel void ranges(device int* RI [[buffer(0)]], device float* RF [[buffer(1)]], device const float* pos [[buffer(2)]],
                   device const int* src_pin [[buffer(3)]], device const int* snk_pin [[buffer(4)]],
                   device const int* pin_macro [[buffer(5)]], device const float* pin_off [[buffer(6)]],
                   device const float* xc [[buffer(7)]], device const float* yc [[buffer(8)]],
                   device const int* ip [[buffer(9)]], device const float* fp [[buffer(10)]], uint i [[thread_position_in_grid]]) {
    int M = ip[0], N = ip[1], R = ip[2], C = ip[3];
    int lane = int(i) / M, p = int(i) - lane * M;
    Pins z = pins(pos, src_pin, snk_pin, pin_macro, pin_off, lane, p, N);
    Soft xs = soft_assign(z.sx, z.sxe, xc, C, fp[5], fp[0]);
    Soft xt = soft_assign(z.tx, z.txe, xc, C, fp[5], fp[0]);
    Soft ys = soft_assign(z.sy, z.sye, yc, R, fp[6], fp[1]);
    Soft yt = soft_assign(z.ty, z.tye, yc, R, fp[6], fp[1]);
    Range rc = soft_range(xs.k0, xs.f, xt.k0, xt.f, fp[2], fp[3]);
    Range rr = soft_range(ys.k0, ys.f, yt.k0, yt.f, fp[2], fp[3]);
    RI[4 * i] = rc.k_lo; RI[4 * i + 1] = rc.k_hi; RI[4 * i + 2] = rr.k_lo; RI[4 * i + 3] = rr.k_hi;
    RF[4 * i] = rc.off_lo; RF[4 * i + 1] = rc.off_hi; RF[4 * i + 2] = rr.off_lo; RF[4 * i + 3] = rr.off_hi;
}

// forward 2, a thread per (lane, group row g): the soft assignment of the group's pin as a dense row.
// FR[lane, g, :] = A_y of the source pin of source group g (for H); FC[B + lane, g, :] = A_x of the sink pin of sink group g (for V).
kernel void assign_rows(device float* FR [[buffer(0)]], device float* FC [[buffer(1)]], device const float* pos [[buffer(2)]],
                        device const int* gs_pin [[buffer(3)]], device const int* gt_pin [[buffer(4)]],
                        device const int* pin_macro [[buffer(5)]], device const float* pin_off [[buffer(6)]],
                        device const float* xc [[buffer(7)]], device const float* yc [[buffer(8)]],
                        device const int* ip [[buffer(9)]], device const float* fp [[buffer(10)]], uint i [[thread_position_in_grid]]) {
    int N = ip[1], R = ip[2], C = ip[3], G = ip[4], B = ip[5];
    int lane = int(i) / G, g = int(i) - lane * G;
    int us = gs_pin[g], ut = gt_pin[g];                    // -1: a padding row
    uint r0 = (uint(lane) * uint(G) + uint(g)) * uint(R), c1 = (uint(B + lane) * uint(G) + uint(g)) * uint(C);
    if (us >= 0) {
        float2 y = pin_coord(pos, pin_macro, pin_off, lane, us, N, 1);
        Soft ys = soft_assign(y.x, y.y, yc, R, fp[6], fp[1]);
        for (int r = 0; r < R; r++) { int j = r - ys.k0 + WA; FR[r0 + uint(r)] = (j >= 0 && j < NW) ? ys.a[j] : 0.0f; }
    } else {
        for (int r = 0; r < R; r++) { FR[r0 + uint(r)] = 0.0f; }
    }
    if (ut >= 0) {
        float2 x = pin_coord(pos, pin_macro, pin_off, lane, ut, N, 0);
        Soft xt = soft_assign(x.x, x.y, xc, C, fp[5], fp[0]);
        for (int c = 0; c < C; c++) { int j = c - xt.k0 + WA; FC[c1 + uint(c)] = (j >= 0 && j < NW) ? xt.a[j] : 0.0f; }
    } else {
        for (int c = 0; c < C; c++) { FC[c1 + uint(c)] = 0.0f; }
    }
}

// forward 3a, a thread per (lane, source group row g, column e): FC[lane, g, e] = sum over the pairs of the group of
// w box_columns(e), the pairs added in list order (a static list: no atomics, no sort per step). A pair is skipped
// where its box is below the truncation bound (outside [k_lo - DL, k_hi + DH]).
kernel void box_c(device float* FC [[buffer(0)]], device const int* RI [[buffer(1)]], device const float* RF [[buffer(2)]],
                  device const float* pw [[buffer(3)]], device const int* gs_start [[buffer(4)]], device const int* ip [[buffer(5)]],
                  device const float* fp [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    int M = ip[0], C = ip[3], G = ip[4];
    int lane = int(i) / (G * C), rem = int(i) - lane * (G * C);
    int g = rem / C, e = rem - g * C;
    float rho = fp[4], s = 0.0f;
    uint pbase = uint(lane) * uint(M);
    for (int k = gs_start[g]; k < gs_start[g + 1]; k++) {
        uint q = 4u * (pbase + uint(k));
        int k_lo = RI[q], k_hi = RI[q + 1];
        if (e >= k_lo - DL && e <= k_hi + DH) {
            s += pw[k] * box(rho * (float(e - k_lo) + RF[q]), rho * (float(e - k_hi) + RF[q + 1]));
        }
    }
    FC[i] = s;
}

// forward 3b, a thread per (lane, sink group row g, row r): FR[B + lane, g, r] = sum over the pairs of the group of
// w box_rows(r), the pairs in the group's list order (gt_order: the pairs sorted by sink pin).
kernel void box_r(device float* FR [[buffer(0)]], device const int* RI [[buffer(1)]], device const float* RF [[buffer(2)]],
                  device const float* pw [[buffer(3)]], device const int* gt_start [[buffer(4)]], device const int* gt_order [[buffer(5)]],
                  device const int* ip [[buffer(6)]], device const float* fp [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    int M = ip[0], R = ip[2], G = ip[4], B = ip[5];
    int lane = int(i) / (G * R), rem = int(i) - lane * (G * R);
    int g = rem / R, r = rem - g * R;
    float rho = fp[4], s = 0.0f;
    uint pbase = uint(lane) * uint(M);
    for (int k = gt_start[g]; k < gt_start[g + 1]; k++) {
        int p = gt_order[k];
        uint q = 4u * (pbase + uint(p));
        int k_lo = RI[q + 2], k_hi = RI[q + 3];
        if (r >= k_lo - DL && r <= k_hi + DH) {
            s += pw[p] * box(rho * (float(r - k_lo) + RF[q + 2]), rho * (float(r - k_hi) + RF[q + 3]));
        }
    }
    FR[uint(B) * uint(G) * uint(R) + i] = s;
}

// backward 1, a thread per (lane, pair): the gradient of the pair's four pin coordinates from the cotangent grids GV,
// GH, gathered inside the pair's windows. Writes (g_sx, g_sy, g_tx, g_ty).
kernel void pair_grad(device float* PG [[buffer(0)]], device const float* GV [[buffer(1)]], device const float* GH [[buffer(2)]],
                      device const float* pos [[buffer(3)]], device const int* src_pin [[buffer(4)]],
                      device const int* snk_pin [[buffer(5)]], device const int* pin_macro [[buffer(6)]],
                      device const float* pin_off [[buffer(7)]], device const float* pw [[buffer(8)]],
                      device const float* xc [[buffer(9)]], device const float* yc [[buffer(10)]],
                      device const int* ip [[buffer(11)]], device const float* fp [[buffer(12)]], uint i [[thread_position_in_grid]]) {
    int M = ip[0], N = ip[1], R = ip[2], C = ip[3];
    int lane = int(i) / M, p = int(i) - lane * M;
    Pins z = pins(pos, src_pin, snk_pin, pin_macro, pin_off, lane, p, N);
    float w = pw[p], rho = fp[4];
    Soft xs = soft_assign(z.sx, z.sxe, xc, C, fp[5], fp[0]);
    Soft xt = soft_assign(z.tx, z.txe, xc, C, fp[5], fp[0]);
    Soft ys = soft_assign(z.sy, z.sye, yc, R, fp[6], fp[1]);
    Soft yt = soft_assign(z.ty, z.tye, yc, R, fp[6], fp[1]);
    Range rc = soft_range(xs.k0, xs.f, xt.k0, xt.f, fp[2], fp[3]);
    Range rr = soft_range(ys.k0, ys.f, yt.k0, yt.f, fp[2], fp[3]);
    uint g0 = uint(lane) * uint(R * C);
    // H: g_hr[c] = sum_j A_ys[j] G_H[r_j, c] feeds the two box edges; dA[j] = sum_c (G_H[r_j, c] - G_H[r_k0, c]) box[c]
    // is the gradient of A_ys[j] minus that of A_ys[k0]. The difference is taken per cell, so the constant part of
    // G_H cancels before it enters a sum.
    float cmin = 0.0f, cmax = 0.0f, dA[NW];
    for (int j = 0; j < NW; j++) { dA[j] = 0.0f; }
    int c_end = min(C - 1, rc.k_hi + DH);
    for (int c = max(0, rc.k_lo - DL); c <= c_end; c++) {
        Edge e = edges(rho * (float(c - rc.k_lo) + rc.off_lo), rho * (float(c - rc.k_hi) + rc.off_hi));
        float g_k0 = GH[g0 + uint(ys.k0 * C + c)], gcol = 0.0f;
        for (int j = 0; j < NW; j++) {
            int r = ys.k0 + j - WA;
            if (r >= 0 && r < R) {
                float g = GH[g0 + uint(r * C + c)];
                gcol += ys.a[j] * g;
                dA[j] += (g - g_k0) * e.bx;
            }
        }
        cmin += gcol * e.dlo; cmax += gcol * e.dhi;
    }
    // V: the same with rows and columns exchanged, on the sink column
    float rmin = 0.0f, rmax = 0.0f, dB[NW];
    for (int j = 0; j < NW; j++) { dB[j] = 0.0f; }
    int r_end = min(R - 1, rr.k_hi + DH);
    for (int r = max(0, rr.k_lo - DL); r <= r_end; r++) {
        Edge e = edges(rho * (float(r - rr.k_lo) + rr.off_lo), rho * (float(r - rr.k_hi) + rr.off_hi));
        uint row = g0 + uint(r * C);
        float g_k0 = GV[row + uint(xt.k0)], grow = 0.0f;
        for (int j = 0; j < NW; j++) {
            int c = xt.k0 + j - WA;
            if (c >= 0 && c < C) {
                float g = GV[row + uint(c)];
                grow += xt.a[j] * g;
                dB[j] += (g - g_k0) * e.bx;
            }
        }
        rmin += grow * e.dlo; rmax += grow * e.dhi;
    }
    float g_cmin = -rho * w * cmin, g_cmax = rho * w * cmax, g_rmin = -rho * w * rmin, g_rmax = rho * w * rmax;
    // through the soft min and max (first = source, second = sink)
    float g_cs = rc.q * g_cmin + rc.q1 * g_cmax, g_ct = rc.q1 * g_cmin + rc.q * g_cmax;
    float g_rs = rr.q * g_rmin + rr.q1 * g_rmax, g_rt = rr.q1 * g_rmin + rr.q * g_rmax;
    // through the soft assignments: a covariance under A, centred on the nearest cell
    float cov_ys = 0.0f, cov_xt = 0.0f;
    for (int j = 0; j < NW; j++) {
        cov_ys += ys.a[j] * dA[j] * (ys.dm[j] - ys.md);
        cov_xt += xt.a[j] * dB[j] * (xt.dm[j] - xt.md);
    }
    PG[4 * i] = fp[7] * (g_cs * xs.ckm);
    PG[4 * i + 1] = fp[8] * (w * cov_ys + g_rs * ys.ckm);
    PG[4 * i + 2] = fp[7] * (w * cov_xt + g_ct * xt.ckm);
    PG[4 * i + 3] = fp[8] * (g_rt * yt.ckm);
}

// backward 2: a thread per (lane, macro) sums the pin gradients of its pairs in list order (compensated)
kernel void macro_sum(device float* out [[buffer(0)]], device const float* PG [[buffer(1)]], device const int* start [[buffer(2)]],
                      device const int* entry [[buffer(3)]], device const int* ip [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    int M = ip[0], N = ip[1];
    int lane = int(i) / N, m = int(i) - lane * N;
    uint base = uint(lane) * 4u * uint(M);
    float2 s = float2(0.0f), comp = float2(0.0f);
    for (int k = start[m]; k < start[m + 1]; k++) {
        uint e = base + uint(entry[k]);
        float2 y = float2(PG[e], PG[e + 1]) - comp;
        float2 t = s + y;
        comp = (t - s) - y;
        s = t;
    }
    out[2 * i] = s.x; out[2 * i + 1] = s.y;
}
"""


def _pair_lists(static, device):
    """The static lists of the pair accumulation: merged pairs, pins, group rows, CSR of pins per macro, windows."""
    d = device
    on = lambda array, kind: torch.from_numpy(np.ascontiguousarray(array, dtype=kind)).to(d)  # noqa: E731
    h = types.SimpleNamespace()
    P = int(static["src_macro"].shape[0])
    N = int(static["n_macros"])
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
    U = int(pin_rows.shape[0])
    pin_macro = pin_rows[:, 0].astype(np.int64)
    pin_off = np.ascontiguousarray(pin_rows[:, 1:]).view(np.float32).reshape(-1, 2)
    # merged pairs in the order (source pin, sink pin), whatever order the pairs came in
    key = which[:P].astype(np.int64) * U + which[P:]
    merged, member = np.unique(key, return_inverse=True)
    w = np.bincount(member.reshape(-1), weights=np.asarray(static["pair_w"], dtype=np.float64), minlength=merged.shape[0])
    h.M = M = int(merged.shape[0])
    m_src, m_snk = merged // U, merged % U

    # rows of the grouped factors: the pairs of a pin, cut into runs of at most GROUP_MAX
    def groups(pin_of_pair, order):
        pin_sorted = pin_of_pair[order]
        first = np.flatnonzero(np.r_[True, pin_sorted[1:] != pin_sorted[:-1]])
        length = np.diff(np.r_[first, pin_sorted.shape[0]])
        starts, pins = [], []
        for begin, count, pin in zip(first.tolist(), length.tolist(), pin_sorted[first].tolist()):
            for offset in range(0, count, GROUP_MAX):
                starts.append(begin + offset)
                pins.append(pin)
        return np.asarray(starts + [pin_sorted.shape[0]], dtype=np.int64), np.asarray(pins, dtype=np.int64)

    by_sink = np.lexsort((m_src, m_snk))                       # pairs in the order (sink pin, source pin)
    gs_start, gs_pin = groups(m_src, np.arange(M))
    gt_start, gt_pin = groups(m_snk, by_sink)
    h.G = G = max(gs_pin.shape[0], gt_pin.shape[0])

    def padded(start, pin):                                    # padding rows have pin -1 and no pairs
        return (np.r_[start, np.full(G - pin.shape[0], start[-1])].astype(np.int32), np.r_[pin, np.full(G - pin.shape[0], -1)].astype(np.int32))

    gs_start, gs_pin = padded(gs_start, gs_pin)
    gt_start, gt_pin = padded(gt_start, gt_pin)
    h.src_pin, h.snk_pin = on(m_src, np.int32), on(m_snk, np.int32)
    h.pin_macro, h.pin_off = on(pin_macro, np.int32), on(pin_off, np.float32)
    h.pair_w = on(w, np.float32)
    if not np.array_equal(h.pair_w.cpu().numpy().astype(np.float64), w):
        raise ValueError("a merged pair weight is not a float32 number")
    h.gs_start, h.gs_pin, h.gt_start, h.gt_pin = on(gs_start, np.int32), on(gs_pin, np.int32), on(gt_start, np.int32), on(gt_pin, np.int32)
    h.gt_order = on(by_sink, np.int32)

    x_ctr, y_ctr = static["x_ctr"].astype(np.float32), static["y_ctr"].astype(np.float32)
    h.x_ctr, h.y_ctr = torch.from_numpy(x_ctr).to(d), torch.from_numpy(y_ctr).to(d)
    sigma_x, sigma_y = float(static["sigma_x"]), float(static["sigma_y"])
    beta, rho = float(static["beta_minmax"]), float(static["beta_range"])
    cell_w = float(np.diff(x_ctr.astype(np.float64)).mean())
    cell_h = float(np.diff(y_ctr.astype(np.float64)).mean())
    # soft assignment: a weight at j cells from the nearest cell is at most exp(-ratio j (j - 1)) of the largest
    ratio = min(cell_w / sigma_x, cell_h / sigma_y) ** 2      # 4 when sigma is half a cell
    wa = 1
    while math.exp(-ratio * wa * (wa + 1)) > TOL_ASSIGN:
        wa += 1
    # soft box: at the first omitted cell below the run at most exp(-rho (DL - ln 2 / beta)), above exp(-rho (DH + 1 - ln 2 / beta))
    dl = 1
    while math.exp(-rho * (dl - math.log(2.0) / beta)) > TOL_BOX:
        dl += 1
    if wa > 6 or dl > 16:
        raise ValueError(f"windows of {wa} and {dl} cells: sigma or the box edge is too wide for this kernel")
    h.WA, h.DL, h.DH = wa, dl, dl - 1
    h.fp = torch.tensor([1.0 / sigma_x, 1.0 / sigma_y, beta, 1.0 / beta, rho, 1.0 / cell_w, 1.0 / cell_h,
                         2.0 / sigma_x ** 2, 2.0 / sigma_y ** 2], dtype=torch.float32, device=d)
    # CSR list of the pins of every macro: entry = 4 p (source pin) or 4 p + 2 (sink pin); ports (index N) are dropped
    owner = np.concatenate([pin_macro[m_src], pin_macro[m_snk]])
    entry = np.concatenate([4 * np.arange(M, dtype=np.int64), 4 * np.arange(M, dtype=np.int64) + 2])
    keep = owner < N
    order = np.argsort(owner[keep], kind="stable")
    h.entry = on(entry[keep][order], np.int32)
    h.start = on(np.concatenate([[0], np.cumsum(np.bincount(owner[keep], minlength=N))]), np.int32)
    return h


def _bind(lib, names):
    """The kernels of a compiled library, each called with its threadgroup size."""
    return {n: functools.partial(getattr(lib, n), group_size=GROUP_LOOPING if n in LOOPING else GROUP_OTHER) for n in names}


class MetalFineStep:
    name = "metal-fine-step"

    def __init__(self, static, lanes, device):
        d = self.device = torch.device(device)
        if d.type != "mps":
            raise ValueError("MetalFineStep is a set of Metal kernels: device mps, float32 only")
        st = static
        N, NH = int(st["n_macros"]), int(st["n_hard"])
        R, C = int(st["grid_rows"]), int(st["grid_cols"])
        self.N, self.NH, self.R, self.C, self.RC = N, NH, R, C, R * C
        if max(R, C) > ROWMAX:
            raise ValueError(f"grid {R} x {C}: a grid side of at most {ROWMAX} cells")
        self.KD, self.KC, self.SR = int(st["dens_k"]), int(st["cong_k"]), int(st["smooth_range"])
        self.KM = max(self.KD, self.KC)
        if self.KM > self.RC:
            raise ValueError("the stacked top-k needs k <= R C")
        self.area = float(st["area"])
        self.w_density, self.w_congestion = float(st["weight_density"]), float(st["weight_congestion"])
        self.w_boundary, self.mean_weight = float(st["weight_boundary"]), float(st["cong_mean_weight"])
        self.lr, self.beta1, self.beta2, self.eps = (float(st[k]) for k in ("adam_lr", "adam_beta1", "adam_beta2", "adam_eps"))
        on = lambda array, kind: torch.from_numpy(np.ascontiguousarray(array, dtype=kind)).to(d)  # noqa: E731
        self.half = on(np.asarray(st["half"], dtype=np.float32), np.float32)
        self.hi = on(np.stack([st["x_hi"], st["y_hi"]], axis=1), np.float32)
        self.mov = on(np.asarray(st["movable"]), np.float32)

        # wirelength: the nets as pin lists, nets with the same pins merged (weights added), CSR of pin slots per macro
        merged = {}
        for g in range(int(st["net_groups"])):
            owners, offsets = st[f"net{g}_owners"], np.ascontiguousarray(st[f"net{g}_offsets"], dtype=np.float32)
            mask, weights = st[f"net{g}_mask"], st[f"net{g}_weights"]
            bits = offsets.view(np.int32)
            for i in range(owners.shape[0]):
                key = tuple(sorted((int(owners[i, k]), int(bits[i, k, 0]), int(bits[i, k, 1])) for k in range(owners.shape[1]) if mask[i, k]))
                merged[key] = merged.get(key, 0.0) + float(weights[i])
        self.NN = NN = len(merged)
        pins = np.asarray([p for key in merged for p in key], dtype=np.int64).reshape(-1, 3)
        degree = np.asarray([len(key) for key in merged], dtype=np.int64)
        self.P = int(pins.shape[0])
        self.net_start = on(np.concatenate([[0], np.cumsum(degree)]), np.int32)
        self.pin_macro = on(pins[:, 0], np.int32)
        self.pin_off = on(np.ascontiguousarray(pins[:, 1:].astype(np.int32)).view(np.float32).reshape(-1, 2), np.float32)
        self.net_w = on(np.asarray(list(merged.values()), dtype=np.float64) / float(st["wl_norm"]), np.float32)
        owner = pins[:, 0]
        keep = np.flatnonzero(owner < N)
        order = keep[np.argsort(owner[keep], kind="stable")]
        self.mentry = on(order, np.int32)
        self.mstart = on(np.concatenate([[0], np.cumsum(np.bincount(owner[keep], minlength=N))]), np.int32)

        # density; the gradient's window assumes a uniform grid
        self.chq = on(np.asarray(st["charge"], dtype=np.float64) / float(st["cell_area"]), np.float32)
        self.dens = [on(st[f"dens_{k}"], np.float32) for k in ("x_min", "x_max", "y_min", "y_max")]
        x_min, x_max = np.asarray(st["dens_x_min"], dtype=np.float64), np.asarray(st["dens_x_max"], dtype=np.float64)
        y_min, y_max = np.asarray(st["dens_y_min"], dtype=np.float64), np.asarray(st["dens_y_max"], dtype=np.float64)
        cw, ch = (x_max[-1] - x_min[0]) / C, (y_max[-1] - y_min[0]) / R
        for lo, hi, size, n in ((x_min, x_max, cw, C), (y_min, y_max, ch, R)):
            if np.abs(lo - (lo[0] + size * np.arange(n))).max() > 1e-3 * size or np.abs(hi - (lo[0] + size * (np.arange(n) + 1))).max() > 1e-3 * size:
                raise ValueError("the density grid is not uniform: the window of dgradw2 assumes it is")
        self.gf = torch.tensor([float(x_min[0]), float(y_min[0]), cw, ch], dtype=torch.float32, device=d)

        # congestion outside the pair accumulation; the blockage gradient's window assumes a uniform grid
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
        self.pairs = _pair_lists(st, d)

        # overlap and blockage gradient as partial sums: Q chunks of OCH hard macros, RQ rows of the largest window
        half_y = np.asarray(st["half"], dtype=np.float64)[:NH, 1]
        self.Q = -(-NH // OCH)
        # rows of the window: floor((y + hy - y0) / h) + 1 - (floor((y - hy - y0) / h) - 1) + 1 <= floor(2 hy / h) + 4; one more for rounding
        self.RQ = min(R, int(math.floor(2.0 * float(half_y.max()) / float(st["cell_h"]))) + 5)

        # the value buffer: one segment per term
        self.seg, at = {}, 0
        for term, width in (("wirelength", NN), ("overlap", NH), ("boundary", N), ("density", self.KD), ("congestion", R * C)):
            self.seg[term] = (at, at + width)
            at += width
        self.VW = at

        h = self.pairs
        step_lib = torch.mps.compile_shader(STEP_SHADER.replace("@FARW@", "%.9ff" % (FAR * math.sqrt(2.0))).replace("@OCH@", str(OCH)))
        pair_lib = torch.mps.compile_shader(PAIR_SHADER.replace("@WA@", str(h.WA)).replace("@DL@", str(h.DL)).replace("@DH@", str(h.DH)))
        self.k = types.SimpleNamespace(
            **_bind(step_lib, ("prep", "wl_net", "dfac3", "bfac", "ovl_part", "tail5", "setk5", "dbits2", "dgradw2", "cot",
                               "bgrad_part", "hsum", "combine", "adam")),
            **_bind(pair_lib, ("ranges", "assign_rows", "box_c", "box_r", "pair_grad", "macro_sum")))
        # Nothing is sent to the GPU until the command buffer is committed. Recording an event commits it, so the GPU
        # starts on the work already enqueued while the CPU encodes the rest of the step. No effect on the results.
        self._event = torch.mps.event.Event()
        self._buf = {}

    def _buffers(self, b):
        if b not in self._buf:
            d, N, NH, R, C, RC, h = self.device, self.N, self.NH, self.R, self.C, self.RC, self.pairs
            if 2 * b * h.G * (R + C) >= 2 ** 31 or 4 * b * h.M >= 2 ** 31:
                raise ValueError(f"{b} lanes: 32-bit index arithmetic in the kernels")
            z = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=d)  # noqa: E731
            i32 = lambda values: torch.tensor(values, dtype=torch.int32, device=d)  # noqa: E731
            s = self.seg
            self._buf[b] = types.SimpleNamespace(
                ip=i32([N, NH, R, C, b, self.NN, self.KD, self.KC, self.SR, self.VW, s["wirelength"][0], s["overlap"][0],
                        s["boundary"][0], s["density"][0], s["congestion"][0], self.P]),
                c=z(b, N, 2), val=z(b, self.VW), pgw=z(b, self.P, 2),
                fx=z(b, N, C), dx=z(b, N, C), fyt=z(b, R, N), dy=z(b, N, R), fy=z(b, N, R), gd=z(b, N, 2),
                bl=z(2 * b, R, NH), br=z(2 * b, NH, C), dbx=z(b, NH, C, 2), dby=z(b, NH, R, 2),
                mask=z(b, 2 * RC), gv=z(b, R, C), gh=z(b, R, C), gb=z(b, NH, 2), go=z(b, NH, 2), gc=z(b, N, 2),
                bits=torch.zeros(b, R, 2, dtype=torch.int32, device=d), tk=z(2 * b, 2 * RC), jpb=i32([b, self.KM]), dm=z(b, RC),
                op=z(b, NH, self.Q, 3), bp=z(b, NH, self.RQ, 2), jq=i32([self.Q, self.RQ]),
                # the pair accumulation: its six integers; runs RI, RF [b, M, 4]; grouped factors FR [2 b, G, R] and
                # FC [2 b, G, C] (rows 0..b-1: the H product of each lane, b..2b-1: the V product); pin gradients PG [b, M, 4]
                hip=i32([h.M, N, R, C, h.G, b]),
                ri=torch.empty(b, h.M, 4, dtype=torch.int32, device=d), rf=torch.empty(b, h.M, 4, device=d),
                fr=torch.empty(2 * b, h.G, R, device=d), fc=torch.empty(2 * b, h.G, C, device=d), pg=torch.empty(b, h.M, 4, device=d))
        return self._buf[b]

    def _pipeline(self, pos, s):
        """Everything up to `combine`: the clamped positions, the values in VAL and the per-term gradient buffers."""
        if pos.dtype != torch.float32 or pos.dim() != 3 or pos.shape[1:] != (self.N, 2):
            raise ValueError(f"positions must be float32 [lanes, {self.N}, 2], got {pos.dtype} {tuple(pos.shape)}")
        b = pos.shape[0]
        B, k, h = self._buffers(b), self.k, self.pairs
        N, NH, R, C, RC = self.N, self.NH, self.R, self.C, self.RC
        ip, hip = B.ip, B.hip
        cd = [self.w_density * 0.5 / self.KD]
        cc = [self.w_congestion / self.KC, self.w_congestion * self.mean_weight / (2 * RC)]

        # the launches that need only the positions
        k.prep(B.c, pos, self.half, self.hi, ip, threads=b * N)
        k.wl_net(B.pgw, B.val, B.c, self.net_start, self.pin_macro, self.pin_off, self.net_w, ip, [s.gamma], threads=b * self.NN)
        k.dfac3(B.fx, B.dx, B.fyt, B.dy, B.fy, B.c, self.half, self.chq, *self.dens, ip,
                [s.sigma_scale, s.sigma_floor_x, s.sigma_floor_y], threads=b * N * (R + C))
        k.ranges(B.ri, B.rf, B.c, h.src_pin, h.snk_pin, h.pin_macro, h.pin_off, h.x_ctr, h.y_ctr, hip, h.fp, threads=b * h.M)
        k.assign_rows(B.fr, B.fc, B.c, h.gs_pin, h.gt_pin, h.pin_macro, h.pin_off, h.x_ctr, h.y_ctr, hip, h.fp, threads=b * h.G)
        k.box_c(B.fc, B.ri, B.rf, h.pair_w, h.gs_start, hip, h.fp, threads=b * h.G * C)
        k.box_r(B.fr, B.ri, B.rf, h.pair_w, h.gt_start, h.gt_order, hip, h.fp, threads=b * h.G * R)
        k.bfac(B.bl, B.br, B.dbx, B.dby, B.c, self.half, *self.trace, ip, self.cf, threads=b * NH * (R + C))
        k.ovl_part(B.op, B.c, self.half, ip, B.jq, threads=b * NH * self.Q)
        self._event.record()

        # the three batched products, the load, ONE top-k, the masks; the selected density values go to VAL
        dgrid = torch.bmm(B.fyt, B.fx)                     # [b, R, C]
        hv = B.fr.transpose(1, 2).bmm(B.fc)                # [2 b, R, C]: rows 0..b-1 H, rows b..2b-1 V
        mb = torch.bmm(B.bl, B.br)                         # [2 b, R, C]: rows 0..b-1 blockage for V, b..2b-1 for H
        k.tail5(B.tk, B.mask, B.dm, hv[b:], hv[:b], mb, dgrid, self.icv, self.ich, ip, self.cf, threads=b * RC)
        vals, idx = torch.topk(B.tk, self.KM, dim=1)
        k.setk5(B.mask, B.val, B.dm, vals, idx, ip, B.jpb, cd, threads=2 * b * self.KM)

        # the gradients of the terms
        k.dbits2(B.bits, B.dm, ip, threads=2 * b * R)
        k.dgradw2(B.gd, B.fx, B.dx, B.fy, B.dy, B.bits, B.c, self.half, ip, self.gf,
                  [cd[0], s.sigma_scale, s.sigma_floor_x, s.sigma_floor_y], threads=b * N)
        k.cot(B.gv, B.gh, B.val, B.tk, B.mask, self.icv, self.ich, ip, cc, threads=b * RC)
        k.pair_grad(B.pg, B.gv, B.gh, B.c, h.src_pin, h.snk_pin, h.pin_macro, h.pin_off, h.pair_w, h.x_ctr, h.y_ctr, hip, h.fp, threads=b * h.M)
        self._event.record()
        k.macro_sum(B.gc, B.pg, h.start, h.entry, hip, threads=b * N)
        k.bgrad_part(B.bp, B.mask, B.bl, B.br, B.dbx, B.dby, B.c, self.half, ip, self.cf, B.jq, cc, threads=b * NH * self.RQ)
        k.hsum(B.go, B.gb, B.val, B.op, B.bp, ip, B.jq, [s.overlap_lambda / self.area], threads=b * NH)
        return B

    def _combine(self, B, pos, select):
        out = torch.empty(pos.shape[0], self.N, 2, dtype=torch.float32, device=self.device)
        self.k.combine(out, B.val, pos, self.half, self.hi, self.mov, B.pgw, self.mstart, self.mentry, B.gd, B.gc, B.gb, B.go, B.ip,
                       [float(select), self.w_boundary / self.area], threads=pos.shape[0] * self.N)
        return out

    def loss_and_grad(self, pos, s):
        """pos [b, N, 2] float32 (not written). Returns loss [b] and the masked gradient [b, N, 2]."""
        with torch.no_grad():
            pos = pos.detach().contiguous()
            B = self._pipeline(pos, s)
            grad = self._combine(B, pos, ALL)
            loss = B.val.sum(dim=1)
        return loss, grad

    def terms(self, pos, s):
        """{term: (value [b], gradient [b, N, 2])} for the five terms, each with its weight in the loss."""
        out = {}
        with torch.no_grad():
            pos = pos.detach().contiguous()
            B = self._pipeline(pos, s)
            grads = {t: self._combine(B, pos, BIT[t]) for t in TERMS}     # the boundary call writes the boundary values
            for t in TERMS:
                lo, hi = self.seg[t]
                out[t] = (B.val[:, lo:hi].sum(dim=1), grads[t])
        return out

    def step(self, pos, exp_avg, exp_avg_sq, s):
        """One step with torch's Adam (single tensor, no weight decay), in place on the three tensors handed in.
        Returns (loss, grad, pos, exp_avg, exp_avg_sq)."""
        if not (pos.is_contiguous() and exp_avg.is_contiguous() and exp_avg_sq.is_contiguous()):
            pos, exp_avg, exp_avg_sq = pos.contiguous(), exp_avg.contiguous(), exp_avg_sq.contiguous()
        loss, grad = self.loss_and_grad(pos, s)
        t = s.adam_t
        self.k.adam(pos, exp_avg, exp_avg_sq, grad, [1.0 - self.beta1, self.beta2, 1.0 - self.beta2, self.lr / (1.0 - self.beta1 ** t),
                                                     math.sqrt(1.0 - self.beta2 ** t), self.eps], threads=pos.numel())
        return loss, grad, pos, exp_avg, exp_avg_sq

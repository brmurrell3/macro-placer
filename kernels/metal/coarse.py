"""The coarse step of the lane-batched placement descent as Metal kernels (float32): loss, gradient and Adam update.

The coarse stage runs before the fine stage on a coarser grid (at most 31 cells a side), in units of the canvas:
pos = microns / [canvas_width, canvas_height], every coordinate in [half, high] of its macro. One step takes
pos [b, N, 2] (float32 on the MPS device, b independent lanes) and returns, per lane, the loss and its gradient;
`step` also makes one Adam update and the projection pos = where(movable, max(half, min(high, pos)), fixed_ref).

    loss = wirelength(pos) + 0.5 density(pos) + 0.5 routing(pos) + penalty overlap(pos)

  wirelength  per net and axis, gamma (log sum exp(x / gamma) + log sum exp(-x / gamma)) over the net's pins
              (pin = macro centre + offset; a port is a fixed pin), times wl_scale of the axis, weighted.
  density     every macro spreads its area over the R x C grid as a separable Gaussian (erf masses per column and
              per row, clamped at 0 and normalised to sum 1, widths max(size / sqrt 12, floor)); the term is the mean
              of the dens_k fullest cells, times 0.5.
  routing     every source-sink pair adds demand: a horizontal pair spreads over rows by a softmax around the source
              y and over columns by |sigmoid(3 C (x - source x)) - sigmoid(3 C (x - sink x))|, a vertical pair the
              same with the axes exchanged; plus the blockage of the hard macros (the area of a macro inside each
              cell). The term is the mean of the route_k largest of the 2 R C loads plus 0.5 x the mean of all.
  overlap     over pairs of hard macros, the product over both axes of max(half_i + half_j - |pos_i - pos_j|, 0).

Inputs. `static` is a dict of numpy arrays and scalars describing the design (macro sizes, projection bounds `half`
and `high`, `fixed_ref`, the nets grouped by degree, the source-sink pairs, the grid, the loss weights and the Adam
constants; the keys are read in `__init__`). The grid is given by four keys that must hold the float32 numbers the
torch code being replaced holds on the same device:

  x_edges [C + 1], y_edges [R + 1]   cell edges, linspace(0, 1)
  x_grid [C], y_grid [R]             cell centres

torch.linspace on MPS and on the CPU differ in the last bit at a few entries. A macro edge exactly on a cell edge then
takes a different subgradient against one set of numbers than against the other, so pass the numbers of the device
the descent runs on. `sched` is a named tuple of the values that change with the step index: gamma, sigma_floor_x,
sigma_floor_y, penalty and adam_t (the 1-based number of the Adam update). Lanes are independent; nothing is carried
from one call to the next (every buffer element that is read was written earlier in the same call; the padding
columns of the two product factors are zero from construction and never written).

Gradients are derived by hand, not by autograd: each kernel writes a term's weighted value and its gradient in gather
form (one thread per output, no atomics, no colliding writes), and `combine` sums them per macro over static CSR lists
(compensated) and multiplies by the movable mask; in `step` the same kernel also makes the Adam update and the
projection. Where the loss has a kink the kernels take the subgradients of the torch code they replace (min and max
split an exact tie, a clamp passes at its bound, |x| has gradient 0 at 0, a top-k mean gives 1/k to the selected
cells).

Routing is computed in key form. A horizontal key is (source macro, source y offset): all pairs of a key share one row
softmax, so the key's demand is that softmax times one span vector (the sum of the key's pair spans). A vertical key
is (sink macro, sink x offset) with the axes exchanged. The keys' two vectors and the blockage covers are the columns
of two factors, and ONE batched matrix product gives the 2 R C loads of every lane. The keys of each grid are ordered
by their number of pairs, longest first, so that neighbouring threads of rfwd and rbwd loop for similar lengths.

Precision. Everything is float32. The kernels are gated against a float64 CPU reference of the same loss: the
gradient error is held to 1e-5 of the largest gradient entry (entries of hard macros with an edge on a cell edge,
where float32 and float64 can take different sides, are judged apart). Float32 care: positions plus offsets are
two-sums, log-sum-exps are taken relative to the extreme pin and softmaxes relative to their largest argument, and
per-macro sums are compensated (Kahan).

One step is 9 kernel launches and 5 torch operations (in `step`, combine also makes the update):

  wl_net  dfac  rfwd  bfac  bmm (density grid)  topk  dgrad  bmm (loads)  topk  cot  rbwd  hard  combine  sum (loss)

Every weighted value goes to one buffer VAL [b, nets + hard + dens_k + 2 R C]; the loss is one sum over it.
"""
from __future__ import annotations

import math
import types

import numpy as np
import torch

TERMS = ("wirelength", "density", "routing", "overlap")
BIT = {"wirelength": 1, "density": 2, "routing": 4, "overlap": 8}
ALL = sum(BIT.values())
GROUP = 64       # threadgroup size: the threads of every kernel loop, and groups of 64 finish sooner than torch's 1,024

# ip: 0 N, 1 NH, 2 R, 3 C, 4 B, 5 NN (merged nets), 6 KD, 7 KR, 8 KA (horizontal keys), 9 KT (all keys), 10 K2 (width
#     of the product), 11 VW, 12 V_WL, 13 V_OV, 14 V_DN, 15 V_RT, 16 P (wirelength pin slots), 17 E (pair entries),
#     18 KOFF (first blockage column)
# cf: 0 wl_scale x, 1 wl_scale y, 2 R / 0.6, 3 C / 0.6, 4 3 R, 5 3 C, 6 h_blockage R C, 7 v_blockage R C, 8 R C
SHADER = r"""
#include <metal_stdlib>
#include <c10/metal/special_math.h>
using namespace metal;
#define GMAX 32          // thread-local rows: a grid side plus one edge
#define WCUT 1.0e-10f    // softmax weights below this are skipped in rbwd

static inline float2 two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    return float2(s, (a - (s - bb)) + (b - bb));
}

// the argument is clamped so that exp cannot overflow; at +-40 the sigmoid is 0 or 1 in float32
static inline float sigm(float t) { return 1.0f / (1.0f + exp(-clamp(t, -40.0f, 40.0f))); }

static inline float dsigm(float t) {
    float e = exp(-min(fabs(t), 40.0f));
    float q = 1.0f + e;
    return e / (q * q);
}

static inline float2 pin(device const float* pos, device const int* pm, device const float* po, int lane, int k, int N, int ax) {
    int m = pm[k];
    float base = 0.0f;
    if (m < N) { base = pos[2 * (lane * N + m) + ax]; }
    return two_sum(base, po[2 * k + ax]);
}

// wirelength, a thread per (lane, net). fp: gamma. Per axis wl_scale (max - min + gamma (log S_hi + log S_lo)) with
// S_hi = sum exp((x - max) / gamma), S_lo = sum exp((min - x) / gamma); pin gradient w wl_scale (e_hi / S_hi - e_lo / S_lo).
kernel void wl_net(device float* PGW [[buffer(0)]], device float* VAL [[buffer(1)]], device const float* pos [[buffer(2)]],
                   device const int* net_start [[buffer(3)]], device const int* pm [[buffer(4)]], device const float* po [[buffer(5)]],
                   device const float* net_w [[buffer(6)]], device const int* ip [[buffer(7)]], device const float* cf [[buffer(8)]],
                   constant float* fp [[buffer(9)]], uint i [[thread_position_in_grid]]) {
    int N = ip[0], NN = ip[5], P = ip[16];
    int lane = int(i) / NN, n = int(i) - lane * NN;
    int k0 = net_start[n], k1 = net_start[n + 1];
    float g = fp[0], w = net_w[n], val = 0.0f;
    for (int ax = 0; ax < 2; ax++) {
        float smax = -3.0e38f, emax = 0.0f, smin = 3.0e38f, emin = 0.0f;
        for (int k = k0; k < k1; k++) {
            float2 t = pin(pos, pm, po, lane, k, N, ax);
            if (t.x > smax || (t.x == smax && t.y > emax)) { smax = t.x; emax = t.y; }
            if (t.x < smin || (t.x == smin && t.y < emin)) { smin = t.x; emin = t.y; }
        }
        float shi = 0.0f, slo = 0.0f;
        for (int k = k0; k < k1; k++) {
            float2 t = pin(pos, pm, po, lane, k, N, ax);
            shi += exp(((t.x - smax) + (t.y - emax)) / g);
            slo += exp(((smin - t.x) + (emin - t.y)) / g);
        }
        float sc = cf[ax];
        val += sc * (((smax - smin) + (emax - emin)) + g * (log(shi) + log(slo)));
        float ih = w * sc / shi, il = w * sc / slo;
        for (int k = k0; k < k1; k++) {
            float2 t = pin(pos, pm, po, lane, k, N, ax);
            PGW[2 * (lane * P + k) + ax] = exp(((t.x - smax) + (t.y - emax)) / g) * ih - exp(((smin - t.x) + (emin - t.y)) / g) * il;
        }
    }
    VAL[lane * ip[11] + ip[12] + n] = w * val;
}

// density factors, a thread per (lane, macro). fp: floor_x, floor_y.
// FX [B, N, C] = normalised mass per column; FYT [B, R, N] = area R C x normalised mass per row.
kernel void dfac(device float* FYT [[buffer(0)]], device float* FX [[buffer(1)]], device const float* pos [[buffer(2)]],
                 device const float* sig [[buffer(3)]], device const float* area [[buffer(4)]], device const float* xe [[buffer(5)]],
                 device const float* ye [[buffer(6)]], device const int* ip [[buffer(7)]], device const float* cf [[buffer(8)]],
                 constant float* fp [[buffer(9)]], uint i [[thread_position_in_grid]]) {
    int N = ip[0], R = ip[2], C = ip[3];
    int lane = int(i) / N, n = int(i) - lane * N;
    float m[GMAX];
    for (int ax = 0; ax < 2; ax++) {
        int cnt = ax == 0 ? C : R;
        float u = pos[2 * i + ax];
        float w = max(sig[2 * n + ax], fp[ax]) * 1.41421356237309504880f;
        float S = 0.0f;
        float prev = c10::metal::erf(((ax == 0 ? xe[0] : ye[0]) - u) / w);
        for (int j = 0; j < cnt; j++) {
            float cur = c10::metal::erf(((ax == 0 ? xe[j + 1] : ye[j + 1]) - u) / w);
            m[j] = max(cur - prev, 0.0f);
            S += m[j];
            prev = cur;
        }
        float inv = 1.0f / max(S, 1.0e-8f);
        if (ax == 0) {
            for (int j = 0; j < cnt; j++) { FX[int(i) * C + j] = m[j] * inv; }
        } else {
            float q = area[n] * cf[8] * inv;
            for (int j = 0; j < cnt; j++) { FYT[(lane * R + j) * N + n] = q * m[j]; }
        }
    }
}

// density gradient, a thread per (lane, macro): the masses and their derivatives again, in thread-local arrays, the
// derivative of the normalisation included; the gradient gathered from the KD selected cells (compensated). The first
// KD threads of a lane also write the selected values. fp: floor_x, floor_y, weight x 0.5 / KD.
kernel void dgrad(device float* GD [[buffer(0)]], device float* VAL [[buffer(1)]], device const float* pos [[buffer(2)]],
                  device const float* sig [[buffer(3)]], device const float* area [[buffer(4)]], device const float* xe [[buffer(5)]],
                  device const float* ye [[buffer(6)]], device const long* idx [[buffer(7)]], device const float* vals [[buffer(8)]],
                  device const int* ip [[buffer(9)]], device const float* cf [[buffer(10)]], constant float* fp [[buffer(11)]],
                  uint i [[thread_position_in_grid]]) {
    int N = ip[0], R = ip[2], C = ip[3], KD = ip[6];
    int lane = int(i) / N, n = int(i) - lane * N;
    float p[2][GMAX], dp[2][GMAX];
    for (int ax = 0; ax < 2; ax++) {
        int cnt = ax == 0 ? C : R;
        float u = pos[2 * i + ax];
        float w = max(sig[2 * n + ax], fp[ax]) * 1.41421356237309504880f;
        float S = 0.0f, dS = 0.0f;
        float tp = ((ax == 0 ? xe[0] : ye[0]) - u) / w;
        float prev = c10::metal::erf(tp), gp = exp(-tp * tp);
        for (int j = 0; j < cnt; j++) {
            float tc = ((ax == 0 ? xe[j + 1] : ye[j + 1]) - u) / w;
            float cur = c10::metal::erf(tc), gc = exp(-tc * tc);
            float d = cur - prev;
            p[ax][j] = max(d, 0.0f);
            dp[ax][j] = d >= 0.0f ? (gp - gc) * (1.12837916709551257390f / w) : 0.0f;
            S += p[ax][j]; dS += dp[ax][j];
            prev = cur; gp = gc;
        }
        float Sc = max(S, 1.0e-8f);
        if (S < 1.0e-8f) { dS = 0.0f; }
        float inv = 1.0f / Sc;
        for (int j = 0; j < cnt; j++) {
            float q = p[ax][j] * inv;
            dp[ax][j] = (dp[ax][j] - q * dS) * inv;
            p[ax][j] = q;
        }
    }
    float2 s = float2(0.0f), comp = float2(0.0f);
    for (int j = 0; j < KD; j++) {
        int cell = int(idx[lane * KD + j]);
        int r = cell / C, cc = cell - r * C;
        float2 y = float2(p[1][r] * dp[0][cc], dp[1][r] * p[0][cc]) - comp;
        float2 t = s + y;
        comp = (t - s) - y;
        s = t;
    }
    float q = fp[2] * area[n] * cf[8];
    GD[2 * i] = q * s.x; GD[2 * i + 1] = q * s.y;
    if (n < KD) { VAL[lane * ip[11] + ip[14] + n] = fp[2] * vals[lane * KD + n]; }
}

// routing forward, a thread per (lane, key): the key's softmax and its span vector into the two product factors
// LEFT [2 b, R, K2] and RIGHT [2 b, K2, C] (lane 2 b: horizontal loads, 2 b + 1: vertical). Horizontal key k < KA:
// weight axis y (R cells), span axis x (C cells). The span of a pair is |sigmoid edge of one pin - sigmoid edge of
// the other|, which equals the soft box between the hard min and max of the two pins.
kernel void rfwd(device float* LEFT [[buffer(0)]], device float* RIGHT [[buffer(1)]], device const float* pos [[buffer(2)]],
                 device const int* kmac [[buffer(3)]], device const float* kwoff [[buffer(4)]], device const int* kstart [[buffer(5)]],
                 device const int* eoth [[buffer(6)]], device const float* eown [[buffer(7)]], device const float* eooff [[buffer(8)]],
                 device const float* ew [[buffer(9)]], device const float* xg [[buffer(10)]], device const float* yg [[buffer(11)]],
                 device const int* ip [[buffer(12)]], device const float* cf [[buffer(13)]], uint i [[thread_position_in_grid]]) {
    int N = ip[0], R = ip[2], C = ip[3], KA = ip[8], KT = ip[9], K2 = ip[10];
    int lane = int(i) / KT, k = int(i) - lane * KT;
    bool H = k < KA;
    int nW = H ? R : C, nS = H ? C : R, wax = H ? 1 : 0, sax = H ? 0 : 1;
    float kap = H ? cf[2] : cf[3], ks = H ? cf[5] : cf[4];
    int m = kmac[k];
    float bw = 0.0f, bs = 0.0f;
    if (m < N) { bw = pos[2 * (lane * N + m) + wax]; bs = pos[2 * (lane * N + m) + sax]; }
    float2 u = two_sum(bw, kwoff[k]);
    float wv[GMAX], acc[GMAX], gs[GMAX];
    float lmax = -3.0e38f;
    for (int j = 0; j < nW; j++) {
        float z = ((u.x - (H ? yg[j] : xg[j])) + u.y) * kap;
        wv[j] = -0.5f * z * z;
        lmax = max(lmax, wv[j]);
    }
    float Z = 0.0f;
    for (int j = 0; j < nW; j++) { wv[j] = exp(wv[j] - lmax); Z += wv[j]; }
    float iz = 1.0f / Z;
    for (int j = 0; j < nS; j++) { acc[j] = 0.0f; gs[j] = H ? xg[j] : yg[j]; }
    for (int e = kstart[k]; e < kstart[k + 1]; e++) {
        float2 a = two_sum(bs, eown[e]);
        int mo = eoth[e];
        float bo = 0.0f;
        if (mo < N) { bo = pos[2 * (lane * N + mo) + sax]; }
        float2 b = two_sum(bo, eooff[e]);
        float w = ew[e];
        for (int j = 0; j < nS; j++) {
            float sa = sigm(ks * ((gs[j] - a.x) - a.y));
            float sb = sigm(ks * ((gs[j] - b.x) - b.y));
            acc[j] += w * fabs(sa - sb);
        }
    }
    if (H) {
        int base = 2 * lane;
        for (int r = 0; r < R; r++) { LEFT[(base * R + r) * K2 + k] = wv[r] * iz; }
        for (int c = 0; c < C; c++) { RIGHT[(base * K2 + k) * C + c] = acc[c]; }
    } else {
        int base = 2 * lane + 1, kb = k - KA;
        for (int r = 0; r < R; r++) { LEFT[(base * R + r) * K2 + kb] = acc[r]; }
        for (int c = 0; c < C; c++) { RIGHT[(base * K2 + kb) * C + c] = wv[c] * iz; }
    }
}

// blockage covers, a thread per (lane, hard macro): columns KOFF + n of both grids' factors.
kernel void bfac(device float* LEFT [[buffer(0)]], device float* RIGHT [[buffer(1)]], device const float* pos [[buffer(2)]],
                 device const float* half_ [[buffer(3)]], device const float* xe [[buffer(4)]], device const float* ye [[buffer(5)]],
                 device const int* ip [[buffer(6)]], device const float* cf [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    int N = ip[0], NH = ip[1], R = ip[2], C = ip[3], K2 = ip[10], KOFF = ip[18];
    int lane = int(i) / NH, n = int(i) - lane * NH;
    int col = KOFF + n;
    for (int ax = 0; ax < 2; ax++) {
        int cnt = ax == 0 ? C : R;
        float u = pos[2 * (lane * N + n) + ax], h = half_[2 * n + ax];
        float a = u + h, b = u - h;
        for (int j = 0; j < cnt; j++) {
            float emin = ax == 0 ? xe[j] : ye[j], emax = ax == 0 ? xe[j + 1] : ye[j + 1];
            float cov = max(min(a, emax) - max(b, emin), 0.0f);
            if (ax == 0) {
                RIGHT[((2 * lane) * K2 + col) * C + j] = cov;
                RIGHT[((2 * lane + 1) * K2 + col) * C + j] = cov;
            } else {
                LEFT[((2 * lane) * R + j) * K2 + col] = cov * cf[6];
                LEFT[((2 * lane + 1) * R + j) * K2 + col] = cov * cf[7];
            }
        }
    }
}

// cotangent of the loads and the routing value, a thread per (lane, load cell). fp: ct = weight / KR, cm = weight x
// mean weight / (2 R C).
kernel void cot(device float* COT [[buffer(0)]], device float* VAL [[buffer(1)]], device const float* LOAD [[buffer(2)]],
                device const long* idx [[buffer(3)]], device const int* ip [[buffer(4)]], constant float* fp [[buffer(5)]],
                uint i [[thread_position_in_grid]]) {
    int W = 2 * ip[2] * ip[3], KR = ip[7];
    int lane = int(i) / W, cell = int(i) - lane * W;
    float sel = 0.0f;
    for (int j = 0; j < KR; j++) { if (int(idx[lane * KR + j]) == cell) { sel = 1.0f; } }
    float co = fp[0] * sel + fp[1];
    COT[i] = co;
    VAL[lane * ip[11] + ip[15] + cell] = co * LOAD[i];
}

// routing backward, a thread per (lane, key). KG [B, KT, 2] = gradient of the key's macro (x, y) through the key's
// softmax and through its own span pins; PG [B, E] = gradient of each pair's other pin on the span axis (a slot per
// pair, summed per macro in combine).
kernel void rbwd(device float* KG [[buffer(0)]], device float* PG [[buffer(1)]], device const float* COT [[buffer(2)]],
                 device const float* LEFT [[buffer(3)]], device const float* RIGHT [[buffer(4)]], device const float* pos [[buffer(5)]],
                 device const int* kmac [[buffer(6)]], device const float* kwoff [[buffer(7)]], device const int* kstart [[buffer(8)]],
                 device const int* eoth [[buffer(9)]], device const float* eown [[buffer(10)]], device const float* eooff [[buffer(11)]],
                 device const float* ew [[buffer(12)]], device const float* xg [[buffer(13)]], device const float* yg [[buffer(14)]],
                 device const int* ip [[buffer(15)]], device const float* cf [[buffer(16)]], uint i [[thread_position_in_grid]]) {
    int N = ip[0], R = ip[2], C = ip[3], KA = ip[8], KT = ip[9], K2 = ip[10], E = ip[17];
    int lane = int(i) / KT, k = int(i) - lane * KT;
    bool H = k < KA;
    int nW = H ? R : C, nS = H ? C : R, wax = H ? 1 : 0, sax = H ? 0 : 1;
    float kap = H ? cf[2] : cf[3], ks = H ? cf[5] : cf[4];
    int m = kmac[k];
    float bw = 0.0f, bs = 0.0f;
    if (m < N) { bw = pos[2 * (lane * N + m) + wax]; bs = pos[2 * (lane * N + m) + sax]; }
    float2 u = two_sum(bw, kwoff[k]);
    float wv[GMAX], zc[GMAX], acc[GMAX], gs[GMAX], U[GMAX];
    float lmax = -3.0e38f;
    for (int j = 0; j < nW; j++) {
        float z = ((u.x - (H ? yg[j] : xg[j])) + u.y) * kap;
        zc[j] = z;
        wv[j] = -0.5f * z * z;
        lmax = max(lmax, wv[j]);
    }
    float Z = 0.0f;
    for (int j = 0; j < nW; j++) { wv[j] = exp(wv[j] - lmax); Z += wv[j]; }
    float iz = 1.0f / Z, zbar = 0.0f;
    for (int j = 0; j < nW; j++) { wv[j] *= iz; zbar += wv[j] * zc[j]; }
    int g0 = lane * 2 * R * C + (H ? 0 : R * C);
    int kb = k - KA;
    // the key's span vector, as rfwd wrote it into the product factors
    for (int s = 0; s < nS; s++) {
        U[s] = 0.0f;
        gs[s] = H ? xg[s] : yg[s];
        acc[s] = H ? RIGHT[((2 * lane) * K2 + k) * C + s] : LEFT[((2 * lane + 1) * R + s) * K2 + kb];
    }
    // softmax gradient: sum_j w_j (z_j - zbar) t_j with t_j = cotangent row j . span; U = cotangent rows under the softmax
    float gw = 0.0f;
    for (int j = 0; j < nW; j++) {
        if (wv[j] < WCUT) { continue; }
        float t = 0.0f;
        for (int s = 0; s < nS; s++) {
            float g = H ? COT[g0 + j * C + s] : COT[g0 + s * C + j];
            U[s] += wv[j] * g;
            t += g * acc[s];
        }
        gw += wv[j] * (zc[j] - zbar) * t;
    }
    gw *= -kap;
    // span gradient per pair: d|sa - sb| = sign(a - b) (dsa - dsb); the sign of the hard comparison is 0 at a tie
    float gown = 0.0f;
    for (int e = kstart[k]; e < kstart[k + 1]; e++) {
        float2 a = two_sum(bs, eown[e]);
        int mo = eoth[e];
        float bo = 0.0f;
        if (mo < N) { bo = pos[2 * (lane * N + mo) + sax]; }
        float2 b = two_sum(bo, eooff[e]);
        float sg = a.x < b.x ? -1.0f : (a.x > b.x ? 1.0f : 0.0f);
        float da = 0.0f, db = 0.0f;
        if (sg != 0.0f) {
            for (int s = 0; s < nS; s++) {
                da += U[s] * dsigm(ks * ((gs[s] - a.x) - a.y));
                db += U[s] * dsigm(ks * ((gs[s] - b.x) - b.y));
            }
        }
        float w = ew[e] * ks * sg;
        gown += w * da;
        PG[lane * E + e] = -w * db;
    }
    KG[2 * i + sax] = gown;
    KG[2 * i + wax] = gw;
}

// a thread per (lane, hard macro): the blockage gradient with the subgradients of the torch code (min and max split an
// exact tie, the clamp passes at 0) and the overlap (value over j > n, gradient over all j). fp: penalty.
kernel void hard(device float* GB [[buffer(0)]], device float* GO [[buffer(1)]], device float* VAL [[buffer(2)]],
                 device const float* COT [[buffer(3)]], device const float* pos [[buffer(4)]], device const float* half_ [[buffer(5)]],
                 device const float* xe [[buffer(6)]], device const float* ye [[buffer(7)]], device const int* ip [[buffer(8)]],
                 device const float* cf [[buffer(9)]], constant float* fp [[buffer(10)]], uint i [[thread_position_in_grid]]) {
    int N = ip[0], NH = ip[1], R = ip[2], C = ip[3];
    int RC = R * C;
    int lane = int(i) / NH, n = int(i) - lane * NH;
    int row = 2 * lane * N;
    float cv[2][GMAX], dv[2][GMAX];
    for (int ax = 0; ax < 2; ax++) {
        int cnt = ax == 0 ? C : R;
        float u = pos[row + 2 * n + ax], h = half_[2 * n + ax];
        float a = u + h, b = u - h;
        for (int j = 0; j < cnt; j++) {
            float emin = ax == 0 ? xe[j] : ye[j], emax = ax == 0 ? xe[j + 1] : ye[j + 1];
            float dtop = a < emax ? 1.0f : (a == emax ? 0.5f : 0.0f);
            float dbot = b > emin ? 1.0f : (b == emin ? 0.5f : 0.0f);
            float d = min(a, emax) - max(b, emin);
            cv[ax][j] = max(d, 0.0f);
            dv[ax][j] = d >= 0.0f ? dtop - dbot : 0.0f;
        }
    }
    float gx = 0.0f, gy = 0.0f;
    int base = lane * 2 * RC;
    for (int r = 0; r < R; r++) {
        float yc = cv[1][r], yd = dv[1][r];
        if (yc == 0.0f && yd == 0.0f) { continue; }
        for (int c = 0; c < C; c++) {
            float xc = cv[0][c], xd = dv[0][c];
            if (xc == 0.0f && xd == 0.0f) { continue; }
            float g = cf[6] * COT[base + r * C + c] + cf[7] * COT[base + RC + r * C + c];
            gx += g * yc * xd;
            gy += g * yd * xc;
        }
    }
    GB[2 * i] = gx; GB[2 * i + 1] = gy;
    float xi = pos[row + 2 * n], yi = pos[row + 2 * n + 1], hxi = half_[2 * n], hyi = half_[2 * n + 1];
    float ox_ = 0.0f, oy_ = 0.0f, val = 0.0f;
    for (int j = 0; j < NH; j++) {
        if (j == n) { continue; }
        float dx = pos[row + 2 * j] - xi, dy = pos[row + 2 * j + 1] - yi;
        float px = (hxi + half_[2 * j]) - fabs(dx), py = (hyi + half_[2 * j + 1]) - fabs(dy);
        float ox = max(px, 0.0f), oy = max(py, 0.0f);
        if (j > n) { val += ox * oy; }
        if (px >= 0.0f) { ox_ += oy * sign(dx); }
        if (py >= 0.0f) { oy_ += ox * sign(dy); }
    }
    GO[2 * i] = fp[0] * ox_; GO[2 * i + 1] = fp[0] * oy_;
    VAL[lane * ip[11] + ip[13] + n] = fp[0] * val;
}

// the gradient, a thread per (lane, macro): compensated CSR sums of the wirelength pin slots, of the routing keys and
// of the pair slots, plus density, blockage and overlap, times the movable mask; with fp[1] > 0.5 also the Adam update
// and the projection, in place.
// fp: 0 terms to include (bits 1 wirelength, 2 density, 4 routing, 8 overlap), 1 update flag, 2 1 - beta1, 3 beta2,
//     4 1 - beta2, 5 lr / (1 - beta1^t), 6 sqrt(1 - beta2^t), 7 eps
kernel void combine(device float* G [[buffer(0)]], device float* pos [[buffer(1)]], device float* M [[buffer(2)]],
                    device float* V [[buffer(3)]], device const float* PGW [[buffer(4)]], device const int* mstart [[buffer(5)]],
                    device const int* mentry [[buffer(6)]], device const float* GD [[buffer(7)]], device const float* KG [[buffer(8)]],
                    device const int* kms [[buffer(9)]], device const int* kme [[buffer(10)]], device const float* PG [[buffer(11)]],
                    device const int* exs [[buffer(12)]], device const int* exe [[buffer(13)]], device const int* eys [[buffer(14)]],
                    device const int* eye [[buffer(15)]], device const float* GB [[buffer(16)]], device const float* GO [[buffer(17)]],
                    device const float* mov [[buffer(18)]], device const float* half_ [[buffer(19)]], device const float* high [[buffer(20)]],
                    device const float* fixedp [[buffer(21)]], device const int* ip [[buffer(22)]], constant float* fp [[buffer(23)]],
                    uint i [[thread_position_in_grid]]) {
    int N = ip[0], NH = ip[1], KT = ip[9], P = ip[16], E = ip[17];
    int lane = int(i) / N, m = int(i) - lane * N;
    int sel = int(fp[0]);
    float2 g = float2(0.0f);
    if (sel & 1) {
        float2 s = float2(0.0f), comp = float2(0.0f);
        int base = 2 * lane * P;
        for (int k = mstart[m]; k < mstart[m + 1]; k++) {
            int e = base + 2 * mentry[k];
            float2 y = float2(PGW[e], PGW[e + 1]) - comp;
            float2 t = s + y;
            comp = (t - s) - y;
            s = t;
        }
        g = s;
    }
    if (sel & 2) { g += float2(GD[2 * i], GD[2 * i + 1]); }
    if (sel & 4) {
        float2 s = float2(0.0f), comp = float2(0.0f);
        for (int k = kms[m]; k < kms[m + 1]; k++) {
            int e = 2 * (lane * KT + kme[k]);
            float2 y = float2(KG[e], KG[e + 1]) - comp;
            float2 t = s + y;
            comp = (t - s) - y;
            s = t;
        }
        for (int k = exs[m]; k < exs[m + 1]; k++) {
            float2 y = float2(PG[lane * E + exe[k]], 0.0f) - comp;
            float2 t = s + y;
            comp = (t - s) - y;
            s = t;
        }
        for (int k = eys[m]; k < eys[m + 1]; k++) {
            float2 y = float2(0.0f, PG[lane * E + eye[k]]) - comp;
            float2 t = s + y;
            comp = (t - s) - y;
            s = t;
        }
        g += s;
        if (m < NH) { g += float2(GB[2 * (lane * NH + m)], GB[2 * (lane * NH + m) + 1]); }
    }
    if ((sel & 8) && m < NH) { g += float2(GO[2 * (lane * NH + m)], GO[2 * (lane * NH + m) + 1]); }
    g *= mov[m];
    G[2 * i] = g.x; G[2 * i + 1] = g.y;
    if (fp[1] > 0.5f) {
        for (int ax = 0; ax < 2; ax++) {
            int e = 2 * int(i) + ax;
            float gi = ax == 0 ? g.x : g.y;
            float mi = M[e] + fp[2] * (gi - M[e]);
            float vi = fp[3] * V[e] + fp[4] * (gi * gi);
            M[e] = mi; V[e] = vi;
            float p = pos[e] - fp[5] * (mi / (sqrt(vi) / fp[6] + fp[7]));
            p = max(half_[2 * m + ax], min(high[2 * m + ax], p));
            pos[e] = mov[m] > 0.5f ? p : fixedp[2 * m + ax];
        }
    }
}
"""


def _bits_to_float(column):
    return np.ascontiguousarray(np.asarray(column).astype(np.int32)).view(np.float32)


def _csr(owner, count):
    """Entries grouped by owner (owners >= count dropped): (start [count + 1], entry indices)."""
    keep = np.flatnonzero(owner < count)
    order = keep[np.argsort(owner[keep], kind="stable")]
    start = np.concatenate([[0], np.cumsum(np.bincount(owner[keep], minlength=count))])
    return start, order


class MetalCoarseStep:
    name = "metal-coarse-step"

    def __init__(self, static, lanes, device):
        d = self.device = torch.device(device)
        if d.type != "mps":
            raise ValueError("MetalCoarseStep is a set of Metal kernels: device mps, float32 only")
        st = static
        N, NH = int(st["n_macros"]), int(st["n_hard"])
        R, C = int(st["grid_rows"]), int(st["grid_cols"])
        self.N, self.NH, self.R, self.C, self.RC = N, NH, R, C, R * C
        self.KD, self.KR = int(st["dens_k"]), int(st["route_k"])
        if max(R, C) + 1 > 32 or N < self.KD:
            raise ValueError("grid side above 31 cells or fewer macros than selected density cells")
        self.w_density, self.w_routing = float(st["weight_density"]), float(st["weight_routing"])
        self.mean_weight = float(st["route_mean_weight"])
        self.lr, self.beta1, self.beta2, self.eps = (float(st[k]) for k in ("adam_lr", "adam_beta1", "adam_beta2", "adam_eps"))
        on = lambda array, kind: torch.from_numpy(np.ascontiguousarray(array, dtype=kind)).to(d)  # noqa: E731
        self.half, self.high = on(st["half"], np.float32), on(st["high"], np.float32)
        self.fixed = on(st["fixed_ref"], np.float32)
        self.mov = on(np.asarray(st["movable"]), np.float32)
        self.xe, self.ye = on(st["x_edges"], np.float32), on(st["y_edges"], np.float32)
        self.xg, self.yg = on(st["x_grid"], np.float32), on(st["y_grid"], np.float32)

        # wirelength: nets as pin lists, nets with the same pins merged (weights added), CSR of pin slots per macro
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
        self.pin_off = on(_bits_to_float(pins[:, 1:]).reshape(-1, 2), np.float32)
        self.net_w = on(np.asarray(list(merged.values()), dtype=np.float64), np.float32)
        start, order = _csr(pins[:, 0], N)
        self.mstart, self.mentry = on(start, np.int32), on(order, np.int32)

        # density
        self.sig = on(np.asarray(st["sizes"], dtype=np.float64) / math.sqrt(12.0), np.float32)
        self.area = on(st["macro_area"], np.float32)

        # routing: merged pairs (same source pin and sink pin), then the two key lists
        so, to = (np.ascontiguousarray(st[k], dtype=np.float32).view(np.int32) for k in ("src_offset", "sink_offset"))
        cols = np.stack([st["src"].astype(np.int64), so[:, 0], so[:, 1], st["sink"].astype(np.int64), to[:, 0], to[:, 1]], axis=1).astype(np.int64)
        uniq, inverse = np.unique(cols, axis=0, return_inverse=True)
        weight = np.zeros(uniq.shape[0], dtype=np.float64)
        np.add.at(weight, inverse.ravel(), np.asarray(st["route_weight"], dtype=np.float64))
        M = int(uniq.shape[0])
        h_cap, v_cap = float(st["h_capacity"]), float(st["v_capacity"])
        parts = []
        # (key columns, own span offset, other macro, other span offset, capacity)
        for key_cols, own, other, other_off, cap in (((0, 2), 1, 3, 4, h_cap), ((3, 4), 5, 0, 2, v_cap)):
            keys, which = np.unique(uniq[:, list(key_cols)], axis=0, return_inverse=True)
            which = which.ravel()
            count = np.bincount(which, minlength=keys.shape[0])
            # the keys in descending order of their number of pairs; the pairs of a key stay in their merged order
            by_count = np.argsort(-count, kind="stable")                     # new key -> old key
            rank = np.empty_like(by_count)
            rank[by_count] = np.arange(by_count.shape[0])
            order = np.argsort(rank[which], kind="stable")
            parts.append(types.SimpleNamespace(keys=keys[by_count], count=count[by_count], own=_bits_to_float(uniq[order, own]),
                                               other=uniq[order, other], other_off=_bits_to_float(uniq[order, other_off]),
                                               w=weight[order] / cap))
        ph, pv = parts
        self.KA, KB = int(ph.keys.shape[0]), int(pv.keys.shape[0])
        self.KT = self.KA + KB
        self.E = 2 * M                                  # every merged pair is one horizontal and one vertical entry
        self.KOFF = max(self.KA, KB)                    # the blockage columns follow the widest key block
        self.K2 = self.KOFF + NH
        key_macro = np.concatenate([ph.keys[:, 0], pv.keys[:, 0]])
        self.kmac = on(key_macro, np.int32)
        self.kwoff = on(np.concatenate([_bits_to_float(ph.keys[:, 1]), _bits_to_float(pv.keys[:, 1])]), np.float32)
        self.kstart = on(np.concatenate([[0], np.cumsum(np.concatenate([ph.count, pv.count]))]), np.int32)
        ent_other = np.concatenate([ph.other, pv.other])
        self.eoth = on(ent_other, np.int32)
        self.eown = on(np.concatenate([ph.own, pv.own]), np.float32)
        self.eooff = on(np.concatenate([ph.other_off, pv.other_off]), np.float32)
        self.ew = on(np.concatenate([ph.w, pv.w]), np.float32)
        start, order = _csr(key_macro, N)
        self.kms, self.kme = on(start, np.int32), on(order, np.int32)
        is_h = np.arange(2 * M) < M
        start, order = _csr(np.where(is_h, ent_other, N), N)          # horizontal entries: the other pin's x
        self.exs, self.exe = on(start, np.int32), on(order, np.int32)
        start, order = _csr(np.where(is_h, N, ent_other), N)          # vertical entries: the other pin's y
        self.eys, self.eye = on(start, np.int32), on(order, np.int32)
        ws = np.asarray(st["wl_scale"], dtype=np.float64)
        slope, assign = float(st["window_slope_per_cell"]), float(st["assign_sigma_cells"])
        self.cf = torch.tensor([ws[0], ws[1], R / assign, C / assign, slope * R, slope * C, float(st["h_blockage"]) * R * C,
                                float(st["v_blockage"]) * R * C, float(R * C)], dtype=torch.float32, device=d)

        # the value buffer: one segment per term
        self.seg, at = {}, 0
        for term, width in (("wirelength", NN), ("overlap", NH), ("density", self.KD), ("routing", 2 * R * C)):
            self.seg[term] = (at, at + width)
            at += width
        self.VW = at
        self.lib = torch.mps.compile_shader(SHADER)
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

    def _pipeline(self, B, pos, s):
        """Everything up to `combine`: the values in VAL and the per-term gradient buffers."""
        b = pos.shape[0]
        lib, ip, cf = self.lib, B.ip, self.cf
        N, NH, RC = self.N, self.NH, self.RC
        floors = [s.sigma_floor_x, s.sigma_floor_y]
        lib.wl_net(B.pgw, B.val, pos, self.net_start, self.pin_macro, self.pin_off, self.net_w, ip, cf, [s.gamma], threads=b * self.NN, group_size=GROUP)
        lib.dfac(B.fyt, B.fx, pos, self.sig, self.area, self.xe, self.ye, ip, cf, floors, threads=b * N, group_size=GROUP)
        lib.rfwd(B.left, B.right, pos, self.kmac, self.kwoff, self.kstart, self.eoth, self.eown, self.eooff, self.ew, self.xg, self.yg,
                 ip, cf, threads=b * self.KT, group_size=GROUP)
        lib.bfac(B.left, B.right, pos, self.half, self.xe, self.ye, ip, cf, threads=b * NH, group_size=GROUP)
        vals, idx = torch.topk(torch.bmm(B.fyt, B.fx).view(b, RC), self.KD, dim=1)
        lib.dgrad(B.gd, B.val, pos, self.sig, self.area, self.xe, self.ye, idx, vals, ip, cf,
                  floors + [self.w_density * 0.5 / self.KD], threads=b * N, group_size=GROUP)
        load = torch.bmm(B.left, B.right).view(b, 2 * RC)       # per lane: horizontal loads, then vertical
        top = torch.topk(load, self.KR, dim=1)[1]
        lib.cot(B.cot, B.val, load, top, ip, [self.w_routing / self.KR, self.w_routing * self.mean_weight / (2 * RC)],
                threads=b * 2 * RC, group_size=GROUP)
        lib.rbwd(B.kg, B.pg, B.cot, B.left, B.right, pos, self.kmac, self.kwoff, self.kstart, self.eoth, self.eown, self.eooff, self.ew,
                 self.xg, self.yg, ip, cf, threads=b * self.KT, group_size=GROUP)
        lib.hard(B.gb, B.go, B.val, B.cot, pos, self.half, self.xe, self.ye, ip, cf, [s.penalty], threads=b * NH, group_size=GROUP)

    def _combine(self, B, out, pos, m, v, select, update):
        self.lib.combine(out, pos, m, v, B.pgw, self.mstart, self.mentry, B.gd, B.kg, self.kms, self.kme, B.pg, self.exs, self.exe,
                         self.eys, self.eye, B.gb, B.go, self.mov, self.half, self.high, self.fixed, B.ip, [float(select)] + update,
                         threads=pos.shape[0] * self.N, group_size=GROUP)

    def loss_and_grad(self, pos, s):
        """pos [b, N, 2] float32 in canvas units (not written). Returns loss [b] and the masked gradient [b, N, 2]."""
        with torch.no_grad():
            pos = pos.detach().contiguous()
            B = self._buffers(pos.shape[0])
            self._pipeline(B, pos, s)
            out = torch.empty(pos.shape[0], self.N, 2, dtype=torch.float32, device=self.device)
            self._combine(B, out, pos, B.m0, B.v0, ALL, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            loss = B.val.sum(dim=1)
        return loss, out

    def terms(self, pos, s):
        """{term: (value [b], gradient [b, N, 2])} for the four terms, each with its weight in the loss."""
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
        """One step with torch's Adam (single tensor, no weight decay) and the projection, in place on the three tensors
        handed in. Returns (loss, grad, pos, exp_avg, exp_avg_sq); grad is a buffer reused by the next call."""
        with torch.no_grad():
            if not (pos.is_contiguous() and exp_avg.is_contiguous() and exp_avg_sq.is_contiguous()):
                pos, exp_avg, exp_avg_sq = pos.contiguous(), exp_avg.contiguous(), exp_avg_sq.contiguous()
            B = self._buffers(pos.shape[0])
            self._pipeline(B, pos, s)
            t = s.adam_t
            self._combine(B, B.g, pos, exp_avg, exp_avg_sq, ALL,
                          [1.0, 1.0 - self.beta1, self.beta2, 1.0 - self.beta2, self.lr / (1.0 - self.beta1 ** t),
                           math.sqrt(1.0 - self.beta2 ** t), self.eps])
            loss = B.val.sum(dim=1)
        return loss, B.g, pos, exp_avg, exp_avg_sq

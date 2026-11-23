// pairs.cu: the pair accumulation of the congestion term and its transpose (float32).
// A translation of the Metal kernels of ../../metal/fine.py (ranges, assign_rows, pair_grad, macro_sum and their
// helpers, box_c, box_r): the same formulas in the same order.
// k9.h is prepended by the launcher. Compile with -DWA=<n> -DDL=<n> -DDH=<n>: the truncation windows the constructor
// derives from the static data (WA cells each side of the nearest cell for a soft assignment; DL, DH cells kept below
// and above a run).
#if !defined(WA) || !defined(DL) || !defined(DH)
#error "pairs.cu: compile with -DWA=<n> -DDL=<n> -DDH=<n>"
#endif
#define NW (2 * WA + 1)

// one pin coordinate against a row of cell centres
struct Soft {
    int k0;          // nearest cell
    float f;         // expected index minus k0
    float ckm;       // sum_k A_k (k - k0) (dm_k - md): covariance of index and centre under A
    float md;        // sum_k A_k dm_k
    float a[NW];     // A_k for k = k0 - WA .. k0 + WA (0 outside the grid)
    float dm[NW];    // centre_k - centre_k0
};

K9_INLINE Soft soft_assign(float u, float ue, const float* ctr, int K, float inv_cell, float inv_sigma) {
    Soft s;
    int k0 = k9_clampi((int)floorf((u - ctr[0]) * inv_cell + 0.5f), 0, K - 1);
    float d0 = fabsf((u - ctr[k0]) + ue);
    if (k0 + 1 < K && fabsf((u - ctr[k0 + 1]) + ue) < d0) { k0 += 1; }
    else if (k0 > 0 && fabsf((u - ctr[k0 - 1]) + ue) < d0) { k0 -= 1; }
    float m0 = ctr[k0];
    float t0 = ((u - m0) + ue) * inv_sigma;
    float sum = 0.0f;
    for (int j = 0; j < NW; j++) {
        int k = k0 + j - WA;
        float e = 0.0f, dm = 0.0f;
        if (k >= 0 && k < K) {
            dm = ctr[k] - m0;
            float a = dm * inv_sigma;             // t0 - t_k
            e = expf(a * (2.0f * t0 - a));        // exp(t0^2 - t_k^2) = exp(z_k - z_k0)
        }
        s.a[j] = e; s.dm[j] = dm; sum += e;
    }
    float inv = 1.0f / sum;
    float f = 0.0f, md = 0.0f;
    for (int j = 0; j < NW; j++) { s.a[j] *= inv; f += s.a[j] * (float)(j - WA); md += s.a[j] * s.dm[j]; }
    float ckm = 0.0f;
    for (int j = 0; j < NW; j++) { ckm += s.a[j] * (float)(j - WA) * (s.dm[j] - md); }
    s.k0 = k0; s.f = f; s.md = md; s.ckm = ckm;
    return s;
}

// soft minimum and maximum of two expected indices, each given as (integer cell, fraction)
struct Range {
    int k_lo; float off_lo;   // sigmoid argument of the low edge at cell c: rho * (float(c - k_lo) + off_lo)
    int k_hi; float off_hi;
    float q;                  // d(min)/d(first) = d(max)/d(second); 1 - q the other two
    float q1;                 // 1 - q, computed without cancellation
};

K9_INLINE Range soft_range(int ka, float fa, int kb, float fb, float beta, float inv_beta) {
    Range r;
    float delta = (float)(kb - ka) + (fb - fa);            // second minus first
    float a = expf(-beta * fabsf(delta));
    float corr = (a < 1.0e-4f ? a * (1.0f - 0.5f * a) : logf(1.0f + a)) * inv_beta;
    bool first_low = delta >= 0.0f;
    r.k_lo = first_low ? ka : kb; r.k_hi = first_low ? kb : ka;
    r.off_lo = (0.5f - (first_low ? fa : fb)) + corr;
    r.off_hi = (0.5f - (first_low ? fb : fa)) - corr;
    float big = 1.0f / (1.0f + a), small = a * big;
    r.q = first_low ? big : small; r.q1 = first_low ? small : big;
    return r;
}

// sigmoid(x1) - sigmoid(x2) for x1 > x2, without the cancellation of 1 - 1
K9_INLINE float box(float x1, float x2) {
    float a1 = expf(-fabsf(x1)), a2 = expf(-fabsf(x2));
    float num = x2 >= 0.0f ? a2 - a1 : (x1 <= 0.0f ? a1 - a2 : 1.0f - a1 * a2);
    return num / ((1.0f + a1) * (1.0f + a2));
}

// the box and the derivatives of its two edges, sigmoid'(x) = a / (1 + a)^2 with a = exp(-|x|), from two exponentials
struct Edge { float bx, dlo, dhi; };

K9_INLINE Edge edges(float x1, float x2) {
    float a1 = expf(-fabsf(x1)), a2 = expf(-fabsf(x2));
    float i1 = 1.0f / (1.0f + a1), i2 = 1.0f / (1.0f + a2);
    float num = x2 >= 0.0f ? a2 - a1 : (x1 <= 0.0f ? a1 - a2 : 1.0f - a1 * a2);
    Edge e;
    e.bx = num * i1 * i2; e.dlo = a1 * i1 * i1; e.dhi = a2 * i2 * i2;
    return e;
}

struct Pins { float sx, sxe, sy, sye, tx, txe, ty, tye; };

// one coordinate (axis 0: x, 1: y) of pin u in a lane: the float32 sum position + offset and its rounding error.
// A pin of a port (macro index N) sits at its offset.
K9_INLINE k9_f2 pin_coord(const float* pos, const int* pin_macro, const float* pin_off, int lane, int u, int N, int axis) {
    int m = pin_macro[u];
    float base = 0.0f;
    if (m < N) { base = pos[2u * ((unsigned)lane * (unsigned)N + (unsigned)m) + (unsigned)axis]; }
    return k9_two_sum(base, pin_off[2 * u + axis]);
}

K9_INLINE Pins pins(const float* pos, const int* src_pin, const int* snk_pin, const int* pin_macro, const float* pin_off,
                    int lane, int p, int N) {
    int us = src_pin[p], ut = snk_pin[p];
    k9_f2 a = pin_coord(pos, pin_macro, pin_off, lane, us, N, 0), b = pin_coord(pos, pin_macro, pin_off, lane, us, N, 1);
    k9_f2 c = pin_coord(pos, pin_macro, pin_off, lane, ut, N, 0), d = pin_coord(pos, pin_macro, pin_off, lane, ut, N, 1);
    Pins r; r.sx = a.x; r.sxe = a.y; r.sy = b.x; r.sye = b.y; r.tx = c.x; r.txe = c.y; r.ty = d.x; r.tye = d.y;
    return r;
}

// ip: M (merged pairs), N, R, C, G (rows of the grouped factors), B (lanes of this call)
// fp: 1/sigma_x, 1/sigma_y, beta, 1/beta, rho, 1/cell_w, 1/cell_h, 2/sigma_x^2, 2/sigma_y^2 (a device buffer, as in v2)

// forward 1, a thread per (lane, pair): the two runs of the pair as (integer end cells, sigmoid offsets)
K9_KERNEL ranges(int* RI, float* RF, const float* pos, const int* src_pin, const int* snk_pin, const int* pin_macro,
                 const float* pin_off, const float* xc, const float* yc, const int* ip, const float* fp, int k9_n) {
    K9_THREAD(i);
    int M = ip[0], N = ip[1], R = ip[2], C = ip[3];
    int lane = i / M, p = i - lane * M;
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
K9_KERNEL assign_rows(float* FR, float* FC, const float* pos, const int* gs_pin, const int* gt_pin, const int* pin_macro,
                      const float* pin_off, const float* xc, const float* yc, const int* ip, const float* fp, int k9_n) {
    K9_THREAD(i);
    int N = ip[1], R = ip[2], C = ip[3], G = ip[4], B = ip[5];
    int lane = i / G, g = i - lane * G;
    int us = gs_pin[g], ut = gt_pin[g];                    // -1: a padding row
    unsigned r0 = ((unsigned)lane * (unsigned)G + (unsigned)g) * (unsigned)R, c1 = ((unsigned)(B + lane) * (unsigned)G + (unsigned)g) * (unsigned)C;
    if (us >= 0) {
        k9_f2 y = pin_coord(pos, pin_macro, pin_off, lane, us, N, 1);
        Soft ys = soft_assign(y.x, y.y, yc, R, fp[6], fp[1]);
        for (int r = 0; r < R; r++) { int j = r - ys.k0 + WA; FR[r0 + (unsigned)r] = (j >= 0 && j < NW) ? ys.a[j] : 0.0f; }
    } else {
        for (int r = 0; r < R; r++) { FR[r0 + (unsigned)r] = 0.0f; }
    }
    if (ut >= 0) {
        k9_f2 x = pin_coord(pos, pin_macro, pin_off, lane, ut, N, 0);
        Soft xt = soft_assign(x.x, x.y, xc, C, fp[5], fp[0]);
        for (int c = 0; c < C; c++) { int j = c - xt.k0 + WA; FC[c1 + (unsigned)c] = (j >= 0 && j < NW) ? xt.a[j] : 0.0f; }
    } else {
        for (int c = 0; c < C; c++) { FC[c1 + (unsigned)c] = 0.0f; }
    }
}

// forward 3 in two launches (v4, BOXSPLIT). A pair is skipped where its box is below the truncation bound (outside
// [k_lo - DL, k_hi + DH]); the pairs of a group are added in list order.
// box_c, a thread per (lane, group row g, column e): FC[lane, g, e] = sum over the pairs of source group g of w box_columns(e) (for H)
K9_KERNEL box_c(float* FC, const int* RI, const float* RF, const float* pw, const int* gs_start, const int* ip,
                const float* fp, int k9_n) {
    K9_THREAD(i);
    int M = ip[0], C = ip[3], G = ip[4];
    int lane = i / (G * C), rem = i - lane * (G * C);
    int g = rem / C, e = rem - g * C;
    float rho = fp[4], s = 0.0f;
    unsigned pbase = (unsigned)lane * (unsigned)M;
    for (int k = gs_start[g]; k < gs_start[g + 1]; k++) {          // the merged pairs are stored in source order
        unsigned q = 4u * (pbase + (unsigned)k);
        int k_lo = RI[q], k_hi = RI[q + 1];
        if (e >= k_lo - DL && e <= k_hi + DH) {
            s += pw[k] * box(rho * ((float)(e - k_lo) + RF[q]), rho * ((float)(e - k_hi) + RF[q + 1]));
        }
    }
    FC[i] = s;
}

// box_r, a thread per (lane, group row g, row r): FR[B + lane, g, r] = sum over the pairs of sink group g of w box_rows(r) (for V)
K9_KERNEL box_r(float* FR, const int* RI, const float* RF, const float* pw, const int* gt_start, const int* gt_order,
                const int* ip, const float* fp, int k9_n) {
    K9_THREAD(i);
    int M = ip[0], R = ip[2], G = ip[4], B = ip[5];
    int lane = i / (G * R), rem = i - lane * (G * R);
    int g = rem / R, r = rem - g * R;
    float rho = fp[4], s = 0.0f;
    unsigned pbase = (unsigned)lane * (unsigned)M;
    for (int k = gt_start[g]; k < gt_start[g + 1]; k++) {
        int p = gt_order[k];
        unsigned q = 4u * (pbase + (unsigned)p);
        int k_lo = RI[q + 2], k_hi = RI[q + 3];
        if (r >= k_lo - DL && r <= k_hi + DH) {
            s += pw[p] * box(rho * ((float)(r - k_lo) + RF[q + 2]), rho * ((float)(r - k_hi) + RF[q + 3]));
        }
    }
    FR[(unsigned)B * (unsigned)G * (unsigned)R + (unsigned)i] = s;
}

// backward 1, a thread per (lane, pair): steps a to e inside the pair's windows. Writes (g_sx, g_sy, g_tx, g_ty).
K9_KERNEL pair_grad(float* PG, const float* GV, const float* GH, const float* pos, const int* src_pin, const int* snk_pin,
                    const int* pin_macro, const float* pin_off, const float* pw, const float* xc, const float* yc,
                    const int* ip, const float* fp, int k9_n) {
    K9_THREAD(i);
    int M = ip[0], N = ip[1], R = ip[2], C = ip[3];
    int lane = i / M, p = i - lane * M;
    Pins z = pins(pos, src_pin, snk_pin, pin_macro, pin_off, lane, p, N);
    float w = pw[p], rho = fp[4];
    Soft xs = soft_assign(z.sx, z.sxe, xc, C, fp[5], fp[0]);
    Soft xt = soft_assign(z.tx, z.txe, xc, C, fp[5], fp[0]);
    Soft ys = soft_assign(z.sy, z.sye, yc, R, fp[6], fp[1]);
    Soft yt = soft_assign(z.ty, z.tye, yc, R, fp[6], fp[1]);
    Range rc = soft_range(xs.k0, xs.f, xt.k0, xt.f, fp[2], fp[3]);
    Range rr = soft_range(ys.k0, ys.f, yt.k0, yt.f, fp[2], fp[3]);
    unsigned g0 = (unsigned)lane * (unsigned)(R * C);
    // H: g_hr[c] = sum_j A_ys[j] G_H[r_j, c] feeds the two box edges (step b); dA[j] = sum_c (G_H[r_j, c] - G_H[r_k0, c]) box[c]
    // is g_Ays[j] - g_Ays[k0] (steps a and e), the difference taken per cell so that the constant part of G_H cancels there
    float cmin = 0.0f, cmax = 0.0f, dA[NW];
    for (int j = 0; j < NW; j++) { dA[j] = 0.0f; }
    int c_end = k9_mini(C - 1, rc.k_hi + DH);
    for (int c = k9_maxi(0, rc.k_lo - DL); c <= c_end; c++) {
        Edge e = edges(rho * ((float)(c - rc.k_lo) + rc.off_lo), rho * ((float)(c - rc.k_hi) + rc.off_hi));
        float g_k0 = GH[g0 + (unsigned)(ys.k0 * C + c)], gcol = 0.0f;
        for (int j = 0; j < NW; j++) {
            int r = ys.k0 + j - WA;
            if (r >= 0 && r < R) {
                float g = GH[g0 + (unsigned)(r * C + c)];
                gcol += ys.a[j] * g;
                dA[j] += (g - g_k0) * e.bx;
            }
        }
        cmin += gcol * e.dlo; cmax += gcol * e.dhi;
    }
    // V: the same with rows and columns exchanged, on the sink column
    float rmin = 0.0f, rmax = 0.0f, dB[NW];
    for (int j = 0; j < NW; j++) { dB[j] = 0.0f; }
    int r_end = k9_mini(R - 1, rr.k_hi + DH);
    for (int r = k9_maxi(0, rr.k_lo - DL); r <= r_end; r++) {
        Edge e = edges(rho * ((float)(r - rr.k_lo) + rr.off_lo), rho * ((float)(r - rr.k_hi) + rr.off_hi));
        unsigned row = g0 + (unsigned)(r * C);
        float g_k0 = GV[row + (unsigned)xt.k0], grow = 0.0f;
        for (int j = 0; j < NW; j++) {
            int c = xt.k0 + j - WA;
            if (c >= 0 && c < C) {
                float g = GV[row + (unsigned)c];
                grow += xt.a[j] * g;
                dB[j] += (g - g_k0) * e.bx;
            }
        }
        rmin += grow * e.dlo; rmax += grow * e.dhi;
    }
    float g_cmin = -rho * w * cmin, g_cmax = rho * w * cmax, g_rmin = -rho * w * rmin, g_rmax = rho * w * rmax;
    // step c: soft min/max (first = source, second = sink)
    float g_cs = rc.q * g_cmin + rc.q1 * g_cmax, g_ct = rc.q1 * g_cmin + rc.q * g_cmax;
    float g_rs = rr.q * g_rmin + rr.q1 * g_rmax, g_rt = rr.q1 * g_rmin + rr.q * g_rmax;
    // steps d and e: covariance form centred on the nearest cell
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

// backward 2 (step f): a thread per (lane, macro) sums the pin gradients of its pairs in list order (compensated)
K9_KERNEL macro_sum(float* out, const float* PG, const int* start, const int* entry, const int* ip, int k9_n) {
    K9_THREAD(i);
    int M = ip[0], N = ip[1];
    int lane = i / N, m = i - lane * N;
    unsigned base = (unsigned)lane * 4u * (unsigned)M;
    k9_f2 s = {0.0f, 0.0f}, comp = {0.0f, 0.0f};
    for (int k = start[m]; k < start[m + 1]; k++) {
        unsigned e = base + (unsigned)entry[k];
        k9_f2 y, t;
        y.x = PG[e] - comp.x; y.y = PG[e + 1] - comp.y;
        t.x = s.x + y.x; t.y = s.y + y.y;
        comp.x = (t.x - s.x) - y.x; comp.y = (t.y - s.y) - y.y;
        s = t;
    }
    out[2 * i] = s.x; out[2 * i + 1] = s.y;
}

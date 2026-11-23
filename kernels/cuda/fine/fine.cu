// fine.cu: the fine step of the batched placement descent outside the pair accumulation (float32).
// A translation of the Metal kernels of ../../metal/fine.py (prep, wl_net, bfac, cot, combine, dfac3, adam, tail5,
// setk5, dbits2, dgradw2, ovl_part, bgrad_part, hsum): the same formulas in the same order. Where the Metal kernel took
// `constant float* fp`, the values come by value (fp0, fp1, ...). k9.h is prepended by the launcher.
// Compile with -DFARW=<float> (the half width of the density gradient's window in widths: FAR sqrt 2) and
// -DOCH=<n> (hard macros per thread of ovl_part).
#if !defined(FARW) || !defined(OCH)
#error "fine.cu: compile with -DFARW=<float literal> -DOCH=<n>"
#endif

// ip: 0 N, 1 NH, 2 R, 3 C, 4 B, 5 NN (merged nets), 6 KD, 7 KC, 8 SR, 9 VW (width of VAL), 10 V_WL, 11 V_OV, 12 V_BD,
//     13 V_DN, 14 V_CG, 15 P (pin slots)
// cf: 0 v_alloc / v_capacity, 1 h_alloc / h_capacity, 2 cell_w, 3 cell_h, 4 x of the first cell edge, 5 y of it

// the clamped positions, a thread per (lane, macro)
K9_KERNEL prep(float* c, const float* pos, const float* lo, const float* hi, const int* ip, int k9_n) {
    K9_THREAD(i);
    unsigned m = (unsigned)i % (unsigned)ip[0];
    c[2 * i] = k9_min(k9_max(pos[2 * i], lo[2 * m]), hi[2 * m]);
    c[2 * i + 1] = k9_min(k9_max(pos[2 * i + 1], lo[2 * m + 1]), hi[2 * m + 1]);
}

// one coordinate of pin slot k: float32 sum position + offset and its rounding error; a port (macro N) sits at its offset
K9_INLINE k9_f2 pin(const float* c, const int* pm, const float* po, int lane, int k, int N, int ax) {
    int m = pm[k];
    float base = 0.0f;
    if (m < N) { base = c[2 * (lane * N + m) + ax]; }
    return k9_two_sum(base, po[2 * k + ax]);
}

// wirelength, a thread per (lane, net). fp0: gamma. Value w (max - min + gamma (log S_hi + log S_lo)) per axis, with
// S_hi = sum exp((x - max) / gamma), S_lo = sum exp((min - x) / gamma); pin gradient w (e_hi / S_hi - e_lo / S_lo).
K9_KERNEL wl_net(float* PGW, float* VAL, const float* c, const int* net_start, const int* pm, const float* po,
                 const float* net_w, const int* ip, float fp0, int k9_n) {
    K9_THREAD(i);
    int N = ip[0], NN = ip[5], P = ip[15];
    int lane = i / NN, n = i - lane * NN;
    int k0 = net_start[n], k1 = net_start[n + 1];
    float g = fp0, w = net_w[n], val = 0.0f;
    for (int ax = 0; ax < 2; ax++) {
        float smax = -3.0e38f, emax = 0.0f, smin = 3.0e38f, emin = 0.0f;
        for (int k = k0; k < k1; k++) {
            k9_f2 t = pin(c, pm, po, lane, k, N, ax);
            if (t.x > smax || (t.x == smax && t.y > emax)) { smax = t.x; emax = t.y; }
            if (t.x < smin || (t.x == smin && t.y < emin)) { smin = t.x; emin = t.y; }
        }
        float shi = 0.0f, slo = 0.0f;
        for (int k = k0; k < k1; k++) {
            k9_f2 t = pin(c, pm, po, lane, k, N, ax);
            shi += expf(((t.x - smax) + (t.y - emax)) / g);
            slo += expf(((smin - t.x) + (emin - t.y)) / g);
        }
        val += ((smax - smin) + (emax - emin)) + g * (logf(shi) + logf(slo));
        float ih = w / shi, il = w / slo;
        for (int k = k0; k < k1; k++) {
            k9_f2 t = pin(c, pm, po, lane, k, N, ax);
            PGW[2 * (lane * P + k) + ax] = expf(((t.x - smax) + (t.y - emax)) / g) * ih - expf(((smin - t.x) + (emin - t.y)) / g) * il;
        }
    }
    VAL[lane * ip[9] + ip[10] + n] = w * val;
}

// density factors, a thread per (lane, macro, e in C + R). fp0: sigma_scale, fp1: floor_x, fp2: floor_y.
// FX [B, N, C] mass per column, DX its derivative by x; FYT [B, R, N] = q x mass per row (q = charge / cell_area),
// FY [B, N, R] the same in macro order, DY [B, N, R] = q x its derivative by y.
K9_KERNEL dfac3(float* FX, float* DX, float* FYT, float* DY, float* FY, const float* c, const float* half_,
                const float* chq, const float* xlo, const float* xhi, const float* ylo, const float* yhi, const int* ip,
                float fp0, float fp1, float fp2, int k9_n) {
    K9_THREAD(i);
    int N = ip[0], R = ip[2], C = ip[3];
    int E = R + C;
    int lane = i / (N * E), rem = i - lane * (N * E);
    int n = rem / E, e = rem - n * E;
    int ln = lane * N + n;
    if (e < C) {
        float x = c[2 * ln];
        float w = k9_max(fp0 * half_[2 * n], fp1) * 1.41421356237309504880f;
        float tl = (xlo[e] - x) / w, th = (xhi[e] - x) / w;
        FX[ln * C + e] = 0.5f * (erff(th) - erff(tl));
        DX[ln * C + e] = (expf(-tl * tl) - expf(-th * th)) * (0.56418958354775628695f / w);
    } else {
        int r = e - C;
        float y = c[2 * ln + 1], q = chq[n];
        float w = k9_max(fp0 * half_[2 * n + 1], fp2) * 1.41421356237309504880f;
        float tl = (ylo[r] - y) / w, th = (yhi[r] - y) / w;
        float mass = q * (0.5f * (erff(th) - erff(tl)));
        FYT[(lane * R + r) * N + n] = mass;
        FY[ln * R + r] = mass;
        DY[ln * R + r] = q * ((expf(-tl * tl) - expf(-th * th)) * (0.56418958354775628695f / w));
    }
}

// macro blockage factors, a thread per (lane, hard macro, e in C + R), stock order of operations and subgradients.
// BR [2 B, NH, C]: lanes 0..B-1 x_ol (for V), B..2B-1 x_present (for H). BL [2 B, R, NH]: y_present (V), y_ol (H).
// DBX [B, NH, C, 2] = d(x_ol, x_present)/dx. DBY [B, NH, R, 2] = d(y_present, y_ol)/dy.
K9_KERNEL bfac(float* BL, float* BR, float* DBX, float* DBY, const float* c, const float* half_, const float* xlo,
               const float* xhi, const float* ylo, const float* yhi, const int* ip, const float* cf, int k9_n) {
    K9_THREAD(i);
    int N = ip[0], NH = ip[1], R = ip[2], C = ip[3], B = ip[4];
    int E = R + C;
    int lane = i / (NH * E), rem = i - lane * (NH * E);
    int n = rem / E, e = rem - n * E;
    bool isx = e < C;
    int k = isx ? e : e - C;
    float u = c[2 * (lane * N + n) + (isx ? 0 : 1)], h = half_[2 * n + (isx ? 0 : 1)];
    float cell = isx ? cf[2] : cf[3];
    float emax = isx ? xhi[k] : yhi[k], emin = isx ? xlo[k] : ylo[k];
    float a = u + h, b = u - h;
    float top = k9_min(a, emax), bot = k9_max(b, emin);
    float dtop = a < emax ? 1.0f : (a == emax ? 0.5f : 0.0f);
    float dbot = b > emin ? 1.0f : (b == emin ? 0.5f : 0.0f);
    float d = top - bot;
    float ol = k9_max(d, 0.0f), dol = d >= 0.0f ? dtop - dbot : 0.0f;
    float t = ol / cell;
    float p = k9_min(t, 1.0f), dp = t <= 1.0f ? dol / cell : 0.0f;
    if (isx) {
        BR[(lane * NH + n) * C + k] = ol; BR[((B + lane) * NH + n) * C + k] = p;
        DBX[2 * ((lane * NH + n) * C + k)] = dol; DBX[2 * ((lane * NH + n) * C + k) + 1] = dp;
    } else {
        BL[(lane * R + k) * NH + n] = p; BL[((B + lane) * R + k) * NH + n] = ol;
        DBY[2 * ((lane * NH + n) * R + k)] = dp; DBY[2 * ((lane * NH + n) * R + k) + 1] = dol;
    }
}

// overlap in short threads: a thread per (lane, hard macro a, chunk q of OCH others). jq: 0 Q (chunks), 1 RQ.
// OP [B, NH, Q, 3]: partial gradient (x, y) of macro a and partial value (pairs a < j).
K9_KERNEL ovl_part(float* OP, const float* c, const float* half_, const int* ip, const int* jq, int k9_n) {
    K9_THREAD(i);
    int N = ip[0], NH = ip[1], Q = jq[0];
    int t = i / Q, q = i - t * Q;
    int lane = t / NH, a = t - lane * NH;
    int row = 2 * lane * N;
    float xi = c[row + 2 * a], yi = c[row + 2 * a + 1], hxi = half_[2 * a], hyi = half_[2 * a + 1];
    float gx = 0.0f, gy = 0.0f, val = 0.0f;
    int j1 = k9_mini(NH, (q + 1) * OCH);
    for (int j = q * OCH; j < j1; j++) {
        if (j == a) { continue; }
        float dx = c[row + 2 * j] - xi, dy = c[row + 2 * j + 1] - yi;
        float px = (hxi + half_[2 * j]) - fabsf(dx), py = (hyi + half_[2 * j + 1]) - fabsf(dy);
        float ox = k9_max(px, 0.0f), oy = k9_max(py, 0.0f);
        if (j > a) { val += ox * oy; }
        if (px >= 0.0f) { gx += oy * k9_sign(dx); }
        if (py >= 0.0f) { gy += ox * k9_sign(dy); }
    }
    OP[3 * i] = gx; OP[3 * i + 1] = gy; OP[3 * i + 2] = val;
}

// the congestion load and the density grid stacked for ONE top-k, a thread per (lane, cell).
// TK [2 B, 2 R C]: rows 0..B-1 the load (vertical then horizontal: 5-cell box of demand over capacity and count, plus
// blockage over capacity), rows B..2B-1 the density grid and then -3e38. The two mask grids are zeroed:
// MASK [B, 2 R C] (selected load cells) and DM [B, R C] (selected density cells).
K9_KERNEL tail5(float* TK, float* MASK, float* DM, const float* V, const float* H, const float* MB, const float* D,
                const float* icv, const float* ich, const int* ip, const float* cf, int k9_n) {
    K9_THREAD(i);
    int R = ip[2], C = ip[3], B = ip[4], SR = ip[8];
    int RC = R * C;
    int lane = i / RC, cell = i - lane * RC;
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

// after the stacked top-k, a thread per (row of TK, j): a load row sets the mask of its j-th cell; a density row
// writes its j-th value into VAL, sets the density mask grid DM and writes the (row, column) of the cell.
// jp: 0 B, 1 width of a row of idx.  fp0: density weight / KD
K9_KERNEL setk5(float* MASK, float* VAL, float* DM, int* RCI, const float* vals, const long long* idx, const int* ip,
                const int* jp, float fp0, int k9_n) {
    K9_THREAD(i);
    int R = ip[2], C = ip[3], B = ip[4], KD = ip[6], KC = ip[7], KM = jp[1];
    int row = i / KM, j = i - row * KM;
    int cell = (int)idx[i];
    if (row < B) {
        if (j < KC) { MASK[(unsigned)row * (unsigned)(2 * R * C) + (unsigned)cell] = 1.0f; }
    } else if (j < KD) {
        int lane = row - B;
        VAL[lane * ip[9] + ip[13] + j] = fp0 * vals[i];
        DM[lane * R * C + cell] = 1.0f;
        int r = cell / C;
        RCI[2 * (lane * KD + j)] = r; RCI[2 * (lane * KD + j) + 1] = cell - r * C;
    }
}

// the selected density cells of a (lane, row) as bits, a thread per (lane, row, 32-column word)
K9_KERNEL dbits2(unsigned* BITS, const float* DM, const int* ip, int k9_n) {
    K9_THREAD(i);
    int C = ip[3];
    int w = (int)((unsigned)i & 1u), t = (int)((unsigned)i >> 1u);
    int c0 = 32 * w, c1 = k9_mini(C, c0 + 32);
    unsigned bits = 0u;
    for (int cc = c0; cc < c1; cc++) {
        if (DM[t * C + cc] != 0.0f) { bits |= 1u << (unsigned)(cc - c0); }
    }
    BITS[i] = bits;
}

K9_INLINE unsigned range32(int a, int b) {        // bits a..b of a 32-bit word, empty when the range misses the word
    a = k9_maxi(a, 0); b = k9_mini(b, 31);
    if (b < a) { return 0u; }
    unsigned upto = b == 31 ? 0xFFFFFFFFu : ((1u << (unsigned)(b + 1)) - 1u);
    return upto & ~((1u << (unsigned)a) - 1u);
}

// the density gradient of one macro, a thread per (lane, macro): the rows of its window (FARW widths and one cell each
// side), and in each row the set bits inside the window's columns; compensated sums.
// fp0: weight / KD, fp1: sigma_scale, fp2: floor_x, fp3: floor_y.  gf: x and y of the first cell edge, cell width, cell height
K9_KERNEL dgradw2(float* GD, const float* FX, const float* DX, const float* FY, const float* DY, const unsigned* BITS,
                  const float* c, const float* half_, const int* ip, const float* gf, float fp0, float fp1, float fp2,
                  float fp3, int k9_n) {
    K9_THREAD(i);
    int N = ip[0], R = ip[2], C = ip[3];
    int lane = i / N, n = i - lane * N;
    float x = c[2 * i], y = c[2 * i + 1];
    float wx = k9_max(fp1 * half_[2 * n], fp2) * FARW, wy = k9_max(fp1 * half_[2 * n + 1], fp3) * FARW;
    int c_lo = k9_clampi((int)floorf((x - wx - gf[0]) / gf[2]) - 1, 0, C - 1), c_hi = k9_clampi((int)floorf((x + wx - gf[0]) / gf[2]) + 1, 0, C - 1);
    int r_lo = k9_clampi((int)floorf((y - wy - gf[1]) / gf[3]) - 1, 0, R - 1), r_hi = k9_clampi((int)floorf((y + wy - gf[1]) / gf[3]) + 1, 0, R - 1);
    unsigned mlo = range32(c_lo, c_hi), mhi = range32(c_lo - 32, c_hi - 32);
    int fb = i * C, rb = i * R;
    k9_f2 s = {0.0f, 0.0f}, comp = {0.0f, 0.0f};
    for (int r = r_lo; r <= r_hi; r++) {
        unsigned lo = BITS[2 * (lane * R + r)] & mlo, hi = BITS[2 * (lane * R + r) + 1] & mhi;
        if ((lo | hi) == 0u) { continue; }
        float fy = FY[rb + r], dy = DY[rb + r];
        while (lo != 0u) {
            int cc = k9_ctz(lo);
            lo &= lo - 1u;
            k9_f2 v, t;
            v.x = fy * DX[fb + cc] - comp.x; v.y = dy * FX[fb + cc] - comp.y;
            t.x = s.x + v.x; t.y = s.y + v.y;
            comp.x = (t.x - s.x) - v.x; comp.y = (t.y - s.y) - v.y;
            s = t;
        }
        while (hi != 0u) {
            int cc = 32 + k9_ctz(hi);
            hi &= hi - 1u;
            k9_f2 v, t;
            v.x = fy * DX[fb + cc] - comp.x; v.y = dy * FX[fb + cc] - comp.y;
            t.x = s.x + v.x; t.y = s.y + v.y;
            comp.x = (t.x - s.x) - v.x; comp.y = (t.y - s.y) - v.y;
            s = t;
        }
    }
    GD[2 * i] = fp0 * s.x; GD[2 * i + 1] = fp0 * s.y;
}

// congestion cotangents, a thread per (lane, cell). fp0: ct = weight / KC, fp1: cm = weight x mean weight / (2 R C).
// d(contribution)/d(load) = ct mask + cm; GV, GH are its transposed box sums over the capacity (for pair_grad).
// LOAD: the first B rows of TK.
K9_KERNEL cot(float* GV, float* GH, float* VAL, const float* LOAD, const float* MASK, const float* icv, const float* ich,
              const int* ip, float fp0, float fp1, int k9_n) {
    K9_THREAD(i);
    int R = ip[2], C = ip[3], SR = ip[8];
    int RC = R * C;
    int lane = i / RC, cell = i - lane * RC;
    int r = cell / C, cc = cell - r * C;
    int base = lane * 2 * RC;
    float ct = fp0, cm = fp1, gv = 0.0f, gh = 0.0f;
    for (int s = -SR; s <= SR; s++) {
        int c2 = cc + s, r2 = r + s;
        if (c2 >= 0 && c2 < C) { gv += ct * MASK[base + r * C + c2] + cm; }
        if (r2 >= 0 && r2 < R) { gh += ct * MASK[base + RC + r2 * C + cc] + cm; }
    }
    GV[i] = gv * icv[cell]; GH[i] = gh * ich[cell];
    float lv = LOAD[base + cell], lh = LOAD[base + RC + cell];
    VAL[lane * ip[9] + ip[14] + cell] = ct * (MASK[base + cell] * lv + MASK[base + RC + cell] * lh) + cm * (lv + lh);
}

// one row of the blockage gradient's window: a thread per (lane, hard macro, q), row r_lo + q. fp0: ct, fp1: cm (as cot).
// BP [B, NH, RQ, 2]. The window is the macro's cells and one cell of margin.
K9_KERNEL bgrad_part(float* BP, const float* MASK, const float* BL, const float* BR, const float* DBX, const float* DBY,
                     const float* c, const float* half_, const int* ip, const float* cf, const int* jq, float fp0,
                     float fp1, int k9_n) {
    K9_THREAD(i);
    int N = ip[0], NH = ip[1], R = ip[2], C = ip[3], B = ip[4], RQ = jq[1];
    int RC = R * C;
    int t = i / RQ, q = i - t * RQ;
    int lane = t / NH, n = t - lane * NH;
    float x = c[2 * (lane * N + n)], y = c[2 * (lane * N + n) + 1], hx = half_[2 * n], hy = half_[2 * n + 1];
    int c_lo = k9_clampi((int)floorf((x - hx - cf[4]) / cf[2]) - 1, 0, C - 1), c_hi = k9_clampi((int)floorf((x + hx - cf[4]) / cf[2]) + 1, 0, C - 1);
    int r_lo = k9_clampi((int)floorf((y - hy - cf[5]) / cf[3]) - 1, 0, R - 1), r_hi = k9_clampi((int)floorf((y + hy - cf[5]) / cf[3]) + 1, 0, R - 1);
    float ct = fp0, cm = fp1, kv = cf[0], kh = cf[1];
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

// the partial sums of a hard macro, a thread per (lane, hard macro): overlap (value and gradient) and blockage
// gradient. fp0: overlap_lambda / area
K9_KERNEL hsum(float* GO, float* GB, float* VAL, const float* OP, const float* BP, const int* ip, const int* jq,
               float fp0, int k9_n) {
    K9_THREAD(i);
    int NH = ip[1], Q = jq[0], RQ = jq[1];
    int lane = i / NH, a = i - lane * NH;
    float gx = 0.0f, gy = 0.0f, val = 0.0f;
    for (int q = 0; q < Q; q++) { gx += OP[3 * (i * Q + q)]; gy += OP[3 * (i * Q + q) + 1]; val += OP[3 * (i * Q + q) + 2]; }
    GO[2 * i] = fp0 * gx; GO[2 * i + 1] = fp0 * gy;
    VAL[lane * ip[9] + ip[11] + a] = fp0 * val;
    float bx = 0.0f, by = 0.0f;
    for (int q = 0; q < RQ; q++) { bx += BP[2 * (i * RQ + q)]; by += BP[2 * (i * RQ + q) + 1]; }
    GB[2 * i] = bx; GB[2 * i + 1] = by;
}

// the gradient, a thread per (lane, macro): CSR sum of the wirelength pin gradients (compensated), plus the other
// gradients, times the clamp's gradient (1 inside and on a bound), plus the boundary value and gradient, times the
// movable mask. fp0: the terms to include (bits: 1 wirelength, 2 density, 4 congestion, 8 overlap, 16 boundary),
// fp1: weight_boundary / area.
K9_KERNEL combine(float* G, float* VAL, const float* pos, const float* lo, const float* hi, const float* mov,
                  const float* PGW, const int* mstart, const int* mentry, const float* GD, const float* GC,
                  const float* GB, const float* GO, const int* ip, float fp0, float fp1, int k9_n) {
    K9_THREAD(i);
    int N = ip[0], NH = ip[1], P = ip[15];
    int lane = i / N, m = i - lane * N;
    int sel = (int)fp0;
    k9_f2 g = {0.0f, 0.0f};
    if (sel & 1) {
        k9_f2 s = {0.0f, 0.0f}, comp = {0.0f, 0.0f};
        unsigned base = 2u * (unsigned)lane * (unsigned)P;
        for (int k = mstart[m]; k < mstart[m + 1]; k++) {
            unsigned e = base + 2u * (unsigned)mentry[k];
            k9_f2 y, t;
            y.x = PGW[e] - comp.x; y.y = PGW[e + 1] - comp.y;
            t.x = s.x + y.x; t.y = s.y + y.y;
            comp.x = (t.x - s.x) - y.x; comp.y = (t.y - s.y) - y.y;
            s = t;
        }
        g = s;
    }
    if (sel & 2) { g.x += GD[2 * i]; g.y += GD[2 * i + 1]; }
    if (sel & 4) {
        g.x += GC[2 * i]; g.y += GC[2 * i + 1];
        if (m < NH) { g.x += GB[2 * (lane * NH + m)]; g.y += GB[2 * (lane * NH + m) + 1]; }
    }
    if ((sel & 8) && m < NH) { g.x += GO[2 * (lane * NH + m)]; g.y += GO[2 * (lane * NH + m) + 1]; }
    float px = pos[2 * i], py = pos[2 * i + 1];
    float lx = lo[2 * m], ly = lo[2 * m + 1], hx = hi[2 * m], hy = hi[2 * m + 1];
    g.x = (px >= lx && px <= hx) ? g.x : 0.0f;
    g.y = (py >= ly && py <= hy) ? g.y : 0.0f;
    if (sel & 16) {
        float ux = k9_max(lx - px, 0.0f), uy = k9_max(ly - py, 0.0f), ox = k9_max(px - hx, 0.0f), oy = k9_max(py - hy, 0.0f);
        g.x += (2.0f * fp1) * (ox - ux); g.y += (2.0f * fp1) * (oy - uy);
        VAL[lane * ip[9] + ip[12] + m] = fp1 * (ux * ux + ox * ox + uy * uy + oy * oy);
    }
    G[2 * i] = g.x * mov[m]; G[2 * i + 1] = g.y * mov[m];
}

// the Adam update, a thread per coordinate, in place.
// fp0: 1 - beta1, fp1: beta2, fp2: 1 - beta2, fp3: lr / (1 - beta1^t), fp4: sqrt(1 - beta2^t), fp5: eps
K9_KERNEL adam(float* pos, float* m, float* v, const float* g, float fp0, float fp1, float fp2, float fp3, float fp4,
               float fp5, int k9_n) {
    K9_THREAD(i);
    float gi = g[i];
    float mi = m[i] + fp0 * (gi - m[i]);
    float vi = fp1 * v[i] + fp2 * (gi * gi);
    m[i] = mi; v[i] = vi;
    pos[i] = pos[i] - fp3 * (mi / (sqrtf(vi) / fp4 + fp5));
}

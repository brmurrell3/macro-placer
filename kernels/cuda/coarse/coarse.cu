// coarse.cu: the coarse step of the lane-batched placement descent as CUDA kernels (float32).
//
// A translation of the Metal kernels of ../../metal/coarse.py, kernel by kernel and statement by statement:
// the same arithmetic in the same order, the two small approximations included (softmax weights below WCUT skipped in
// the backward kernel of the routing term; the sigmoid's argument clamped at 40). One thread per output, no colliding
// writes, no atomics. Compiled with k9.h in front (k9cu.py); plain CUDA C that is also valid HIP: no NVIDIA-only
// intrinsic here.
//
// What differs from the Metal text:
//   - `constant float* fp` is passed by value (fp0, fp1, ...); every kernel ends with `int k9_n` and starts with K9_THREAD;
//   - float2 is written out per component (the compensated sums keep their order: y = v - comp, t = s + y,
//     comp = (t - s) - y, s = t, for x and for y);
//   - min, max, clamp, sign are the helpers of k9.h (Metal semantics); c10::metal::erf is erff; exp, log, sqrt, fabs
//     are expf, logf, sqrtf, fabsf;
//   - the indices of torch.topk are `long long` (Metal: long);
//   - kernel names carry the prefix coarse_, so that a profile of a whole descent tells them from the fine step's;
//   - ONE subgradient choice: where the Metal kernels pass the gradient of a clamp at exactly its bound (d >= 0, the
//     choice of torch 2.8.0), these pass it only above the bound when COARSE_CLAMP_AT_BOUND is 0, which is what the
//     stock torch of this port does (see below). Five comparisons: the mass and its normalisation in coarse_dgrad, the
//     cover and the two overlap widths in coarse_hard.
//
// ip: 0 N, 1 NH, 2 R, 3 C, 4 B, 5 NN (merged nets), 6 KD, 7 KR, 8 KA (horizontal keys), 9 KT (all keys), 10 K2 (width
//     of the product), 11 VW, 12 V_WL, 13 V_OV, 14 V_DN, 15 V_RT, 16 P (wirelength pin slots), 17 E (pair entries),
//     18 KOFF (first blockage column)
// cf: 0 wl_scale x, 1 wl_scale y, 2 R / 0.6, 3 C / 0.6, 4 3 R, 5 3 C, 6 h_blockage R C, 7 v_blockage R C, 8 R C
#define GMAX 32
#define WCUT 1.0e-10f

// The stock subgradient of clamp_min(x, bound) where x is exactly on the bound. torch 2.8.0, the stock code of the
// Metal parts, passes the gradient there (the Metal kernels: d >= 0). torch 2.15.0, the stock code of this port, does
// not (measured: 0 on the CPU and on CUDA). 1 = passes, 0 = does not. step.py
// sets it from the installed torch at construction, as it takes the grid numbers from the live stage.
#ifndef COARSE_CLAMP_AT_BOUND
#define COARSE_CLAMP_AT_BOUND 0
#endif
K9_INLINE bool coarse_passes(float x, float bound) { return COARSE_CLAMP_AT_BOUND ? x >= bound : x > bound; }

K9_INLINE float coarse_sigm(float t) { return 1.0f / (1.0f + expf(-k9_clamp(t, -40.0f, 40.0f))); }

K9_INLINE float coarse_dsigm(float t) {
    float e = expf(-k9_min(fabsf(t), 40.0f));
    float q = 1.0f + e;
    return e / (q * q);
}

K9_INLINE k9_f2 coarse_pin(const float* pos, const int* pm, const float* po, int lane, int k, int N, int ax) {
    int m = pm[k];
    float base = 0.0f;
    if (m < N) { base = pos[2 * (lane * N + m) + ax]; }
    return k9_two_sum(base, po[2 * k + ax]);
}

// wirelength, a thread per (lane, net). fp0: gamma.
K9_KERNEL coarse_wl(float* PGW, float* VAL, const float* pos, const int* net_start, const int* pm, const float* po,
                    const float* net_w, const int* ip, const float* cf, float fp0, int k9_n) {
    K9_THREAD(i);
    int N = ip[0], NN = ip[5], P = ip[16];
    int lane = i / NN, n = i - lane * NN;
    int k0 = net_start[n], k1 = net_start[n + 1];
    float g = fp0, w = net_w[n], val = 0.0f;
    for (int ax = 0; ax < 2; ax++) {
        float smax = -3.0e38f, emax = 0.0f, smin = 3.0e38f, emin = 0.0f;
        for (int k = k0; k < k1; k++) {
            k9_f2 t = coarse_pin(pos, pm, po, lane, k, N, ax);
            if (t.x > smax || (t.x == smax && t.y > emax)) { smax = t.x; emax = t.y; }
            if (t.x < smin || (t.x == smin && t.y < emin)) { smin = t.x; emin = t.y; }
        }
        float shi = 0.0f, slo = 0.0f;
        for (int k = k0; k < k1; k++) {
            k9_f2 t = coarse_pin(pos, pm, po, lane, k, N, ax);
            shi += expf(((t.x - smax) + (t.y - emax)) / g);
            slo += expf(((smin - t.x) + (emin - t.y)) / g);
        }
        float sc = cf[ax];
        val += sc * (((smax - smin) + (emax - emin)) + g * (logf(shi) + logf(slo)));
        float ih = w * sc / shi, il = w * sc / slo;
        for (int k = k0; k < k1; k++) {
            k9_f2 t = coarse_pin(pos, pm, po, lane, k, N, ax);
            PGW[2 * (lane * P + k) + ax] = expf(((t.x - smax) + (t.y - emax)) / g) * ih - expf(((smin - t.x) + (emin - t.y)) / g) * il;
        }
    }
    VAL[lane * ip[11] + ip[12] + n] = w * val;
}

// density factors, a thread per (lane, macro). fp0, fp1: floor_x, floor_y.
// FX [B, N, C] = normalised mass per column; FYT [B, R, N] = area R C x normalised mass per row.
K9_KERNEL coarse_dfac(float* FYT, float* FX, const float* pos, const float* sig, const float* area, const float* xe,
                      const float* ye, const int* ip, const float* cf, float fp0, float fp1, int k9_n) {
    K9_THREAD(i);
    int N = ip[0], R = ip[2], C = ip[3];
    int lane = i / N, n = i - lane * N;
    float m[GMAX];
    for (int ax = 0; ax < 2; ax++) {
        int cnt = ax == 0 ? C : R;
        float u = pos[2 * i + ax];
        float w = k9_max(sig[2 * n + ax], ax == 0 ? fp0 : fp1) * 1.41421356237309504880f;
        float S = 0.0f;
        float prev = erff(((ax == 0 ? xe[0] : ye[0]) - u) / w);
        for (int j = 0; j < cnt; j++) {
            float cur = erff(((ax == 0 ? xe[j + 1] : ye[j + 1]) - u) / w);
            m[j] = k9_max(cur - prev, 0.0f);
            S += m[j];
            prev = cur;
        }
        float inv = 1.0f / k9_max(S, 1.0e-8f);
        if (ax == 0) {
            for (int j = 0; j < cnt; j++) { FX[i * C + j] = m[j] * inv; }
        } else {
            float q = area[n] * cf[8] * inv;
            for (int j = 0; j < cnt; j++) { FYT[(lane * R + j) * N + n] = q * m[j]; }
        }
    }
}

// density gradient, a thread per (lane, macro); the first KD threads of a lane also write the selected values.
// fp0, fp1: floor_x, floor_y; fp2: weight x 0.5 / KD.
K9_KERNEL coarse_dgrad(float* GD, float* VAL, const float* pos, const float* sig, const float* area, const float* xe,
                       const float* ye, const long long* idx, const float* vals, const int* ip, const float* cf,
                       float fp0, float fp1, float fp2, int k9_n) {
    K9_THREAD(i);
    int N = ip[0], R = ip[2], C = ip[3], KD = ip[6];
    int lane = i / N, n = i - lane * N;
    float p[2][GMAX], dp[2][GMAX];
    for (int ax = 0; ax < 2; ax++) {
        int cnt = ax == 0 ? C : R;
        float u = pos[2 * i + ax];
        float w = k9_max(sig[2 * n + ax], ax == 0 ? fp0 : fp1) * 1.41421356237309504880f;
        float S = 0.0f, dS = 0.0f;
        float tp = ((ax == 0 ? xe[0] : ye[0]) - u) / w;
        float prev = erff(tp), gp = expf(-tp * tp);
        for (int j = 0; j < cnt; j++) {
            float tc = ((ax == 0 ? xe[j + 1] : ye[j + 1]) - u) / w;
            float cur = erff(tc), gc = expf(-tc * tc);
            float d = cur - prev;
            p[ax][j] = k9_max(d, 0.0f);
            dp[ax][j] = coarse_passes(d, 0.0f) ? (gp - gc) * (1.12837916709551257390f / w) : 0.0f;
            S += p[ax][j]; dS += dp[ax][j];
            prev = cur; gp = gc;
        }
        float Sc = k9_max(S, 1.0e-8f);
        if (!coarse_passes(S, 1.0e-8f)) { dS = 0.0f; }
        float inv = 1.0f / Sc;
        for (int j = 0; j < cnt; j++) {
            float q = p[ax][j] * inv;
            dp[ax][j] = (dp[ax][j] - q * dS) * inv;
            p[ax][j] = q;
        }
    }
    float sx = 0.0f, sy = 0.0f, cx = 0.0f, cy = 0.0f;
    for (int j = 0; j < KD; j++) {
        int cell = (int)idx[lane * KD + j];
        int r = cell / C, cc = cell - r * C;
        float yx = p[1][r] * dp[0][cc] - cx, yy = dp[1][r] * p[0][cc] - cy;
        float tx = sx + yx, ty = sy + yy;
        cx = (tx - sx) - yx; cy = (ty - sy) - yy;
        sx = tx; sy = ty;
    }
    float q = fp2 * area[n] * cf[8];
    GD[2 * i] = q * sx; GD[2 * i + 1] = q * sy;
    if (n < KD) { VAL[lane * ip[11] + ip[14] + n] = fp2 * vals[lane * KD + n]; }
}

// routing forward, a thread per (lane, key). Horizontal key k < KA: weight axis y (R cells), span axis x (C cells).
K9_KERNEL coarse_rfwd(float* LEFT, float* RIGHT, const float* pos, const int* kmac, const float* kwoff, const int* kstart,
                      const int* eoth, const float* eown, const float* eooff, const float* ew, const float* xg,
                      const float* yg, const int* ip, const float* cf, int k9_n) {
    K9_THREAD(i);
    int N = ip[0], R = ip[2], C = ip[3], KA = ip[8], KT = ip[9], K2 = ip[10];
    int lane = i / KT, k = i - lane * KT;
    bool H = k < KA;
    int nW = H ? R : C, nS = H ? C : R, wax = H ? 1 : 0, sax = H ? 0 : 1;
    float kap = H ? cf[2] : cf[3], ks = H ? cf[5] : cf[4];
    int m = kmac[k];
    float bw = 0.0f, bs = 0.0f;
    if (m < N) { bw = pos[2 * (lane * N + m) + wax]; bs = pos[2 * (lane * N + m) + sax]; }
    k9_f2 u = k9_two_sum(bw, kwoff[k]);
    float wv[GMAX], acc[GMAX], gs[GMAX];
    float lmax = -3.0e38f;
    for (int j = 0; j < nW; j++) {
        float z = ((u.x - (H ? yg[j] : xg[j])) + u.y) * kap;
        wv[j] = -0.5f * z * z;
        lmax = k9_max(lmax, wv[j]);
    }
    float Z = 0.0f;
    for (int j = 0; j < nW; j++) { wv[j] = expf(wv[j] - lmax); Z += wv[j]; }
    float iz = 1.0f / Z;
    for (int j = 0; j < nS; j++) { acc[j] = 0.0f; gs[j] = H ? xg[j] : yg[j]; }
    int e1 = kstart[k + 1];
    for (int e = kstart[k]; e < e1; e++) {
        k9_f2 a = k9_two_sum(bs, eown[e]);
        int mo = eoth[e];
        float bo = 0.0f;
        if (mo < N) { bo = pos[2 * (lane * N + mo) + sax]; }
        k9_f2 b = k9_two_sum(bo, eooff[e]);
        float w = ew[e];
        for (int j = 0; j < nS; j++) {
            float sa = coarse_sigm(ks * ((gs[j] - a.x) - a.y));
            float sb = coarse_sigm(ks * ((gs[j] - b.x) - b.y));
            acc[j] += w * fabsf(sa - sb);
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
K9_KERNEL coarse_bfac(float* LEFT, float* RIGHT, const float* pos, const float* half_, const float* xe, const float* ye,
                      const int* ip, const float* cf, int k9_n) {
    K9_THREAD(i);
    int N = ip[0], NH = ip[1], R = ip[2], C = ip[3], K2 = ip[10], KOFF = ip[18];
    int lane = i / NH, n = i - lane * NH;
    int col = KOFF + n;
    float hb = cf[6], vb = cf[7];
    for (int ax = 0; ax < 2; ax++) {
        int cnt = ax == 0 ? C : R;
        float u = pos[2 * (lane * N + n) + ax], h = half_[2 * n + ax];
        float a = u + h, b = u - h;
        for (int j = 0; j < cnt; j++) {
            float emin = ax == 0 ? xe[j] : ye[j], emax = ax == 0 ? xe[j + 1] : ye[j + 1];
            float cov = k9_max(k9_min(a, emax) - k9_max(b, emin), 0.0f);
            if (ax == 0) {
                RIGHT[((2 * lane) * K2 + col) * C + j] = cov;
                RIGHT[((2 * lane + 1) * K2 + col) * C + j] = cov;
            } else {
                LEFT[((2 * lane) * R + j) * K2 + col] = cov * hb;
                LEFT[((2 * lane + 1) * R + j) * K2 + col] = cov * vb;
            }
        }
    }
}

// cotangent of the loads and the routing value, a thread per (lane, load cell). fp0: ct = weight / KR; fp1: cm = weight x
// mean weight / (2 R C).
K9_KERNEL coarse_cot(float* COT, float* VAL, const float* LOAD, const long long* idx, const int* ip, float fp0, float fp1,
                     int k9_n) {
    K9_THREAD(i);
    int W = 2 * ip[2] * ip[3], KR = ip[7];
    int lane = i / W, cell = i - lane * W;
    float sel = 0.0f;
    for (int j = 0; j < KR; j++) { if ((int)idx[lane * KR + j] == cell) { sel = 1.0f; } }
    float co = fp0 * sel + fp1;
    COT[i] = co;
    VAL[lane * ip[11] + ip[15] + cell] = co * LOAD[i];
}

// routing backward, a thread per (lane, key). KG [B, KT, 2] = gradient of the key's macro (x, y) through the key's
// weight and through its own span pins; PG [B, E] = gradient of each entry's other pin on the span axis.
K9_KERNEL coarse_rbwd(float* KG, float* PG, const float* COT, const float* LEFT, const float* RIGHT, const float* pos,
                      const int* kmac, const float* kwoff, const int* kstart, const int* eoth, const float* eown,
                      const float* eooff, const float* ew, const float* xg, const float* yg, const int* ip,
                      const float* cf, int k9_n) {
    K9_THREAD(i);
    int N = ip[0], R = ip[2], C = ip[3], KA = ip[8], KT = ip[9], K2 = ip[10], E = ip[17];
    int lane = i / KT, k = i - lane * KT;
    bool H = k < KA;
    int nW = H ? R : C, nS = H ? C : R, wax = H ? 1 : 0, sax = H ? 0 : 1;
    float kap = H ? cf[2] : cf[3], ks = H ? cf[5] : cf[4];
    int m = kmac[k];
    float bw = 0.0f, bs = 0.0f;
    if (m < N) { bw = pos[2 * (lane * N + m) + wax]; bs = pos[2 * (lane * N + m) + sax]; }
    k9_f2 u = k9_two_sum(bw, kwoff[k]);
    float wv[GMAX], zc[GMAX], acc[GMAX], gs[GMAX], U[GMAX];
    float lmax = -3.0e38f;
    for (int j = 0; j < nW; j++) {
        float z = ((u.x - (H ? yg[j] : xg[j])) + u.y) * kap;
        zc[j] = z;
        wv[j] = -0.5f * z * z;
        lmax = k9_max(lmax, wv[j]);
    }
    float Z = 0.0f;
    for (int j = 0; j < nW; j++) { wv[j] = expf(wv[j] - lmax); Z += wv[j]; }
    float iz = 1.0f / Z, zbar = 0.0f;
    for (int j = 0; j < nW; j++) { wv[j] *= iz; zbar += wv[j] * zc[j]; }
    int g0 = lane * 2 * R * C + (H ? 0 : R * C);
    int kb = k - KA;
    for (int s = 0; s < nS; s++) {
        U[s] = 0.0f;
        gs[s] = H ? xg[s] : yg[s];
        acc[s] = H ? RIGHT[((2 * lane) * K2 + k) * C + s] : LEFT[((2 * lane + 1) * R + s) * K2 + kb];
    }
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
    float gown = 0.0f;
    int e1 = kstart[k + 1];
    for (int e = kstart[k]; e < e1; e++) {
        k9_f2 a = k9_two_sum(bs, eown[e]);
        int mo = eoth[e];
        float bo = 0.0f;
        if (mo < N) { bo = pos[2 * (lane * N + mo) + sax]; }
        k9_f2 b = k9_two_sum(bo, eooff[e]);
        float sg = a.x < b.x ? -1.0f : (a.x > b.x ? 1.0f : 0.0f);
        float da = 0.0f, db = 0.0f;
        if (sg != 0.0f) {
            for (int s = 0; s < nS; s++) {
                da += U[s] * coarse_dsigm(ks * ((gs[s] - a.x) - a.y));
                db += U[s] * coarse_dsigm(ks * ((gs[s] - b.x) - b.y));
            }
        }
        float w = ew[e] * ks * sg;
        gown += w * da;
        PG[lane * E + e] = -w * db;
    }
    KG[2 * i + sax] = gown;
    KG[2 * i + wax] = gw;
}

// a thread per (lane, hard macro): the blockage gradient (stock subgradients) and the overlap. fp0: penalty.
// The decisions at a kink are float32 comparisons of a = u + h and b = u - h with the float32 cell edges: sums and
// differences only, so the contraction of multiply-add cannot touch them.
K9_KERNEL coarse_hard(float* GB, float* GO, float* VAL, const float* COT, const float* pos, const float* half_,
                      const float* xe, const float* ye, const int* ip, const float* cf, float fp0, int k9_n) {
    K9_THREAD(i);
    int N = ip[0], NH = ip[1], R = ip[2], C = ip[3];
    int RC = R * C;
    int lane = i / NH, n = i - lane * NH;
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
            float d = k9_min(a, emax) - k9_max(b, emin);
            cv[ax][j] = k9_max(d, 0.0f);
            dv[ax][j] = coarse_passes(d, 0.0f) ? dtop - dbot : 0.0f;
        }
    }
    float gx = 0.0f, gy = 0.0f;
    int base = lane * 2 * RC;
    float hb = cf[6], vb = cf[7];
    for (int r = 0; r < R; r++) {
        float yc = cv[1][r], yd = dv[1][r];
        if (yc == 0.0f && yd == 0.0f) { continue; }
        for (int c = 0; c < C; c++) {
            float xc = cv[0][c], xd = dv[0][c];
            if (xc == 0.0f && xd == 0.0f) { continue; }
            float g = hb * COT[base + r * C + c] + vb * COT[base + RC + r * C + c];
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
        float px = (hxi + half_[2 * j]) - fabsf(dx), py = (hyi + half_[2 * j + 1]) - fabsf(dy);
        float ox = k9_max(px, 0.0f), oy = k9_max(py, 0.0f);
        if (j > n) { val += ox * oy; }
        if (coarse_passes(px, 0.0f)) { ox_ += oy * k9_sign(dx); }
        if (coarse_passes(py, 0.0f)) { oy_ += ox * k9_sign(dy); }
    }
    GO[2 * i] = fp0 * ox_; GO[2 * i + 1] = fp0 * oy_;
    VAL[lane * ip[11] + ip[13] + n] = fp0 * val;
}

// the gradient, a thread per (lane, macro), and with fp1 > 0.5 the Adam update and the projection in place.
// fp0 terms to include (bits 1 wirelength, 2 density, 4 routing, 8 overlap), fp1 update flag, fp2 1 - beta1, fp3 beta2,
// fp4 1 - beta2, fp5 lr / (1 - beta1^t), fp6 sqrt(1 - beta2^t), fp7 eps
K9_KERNEL coarse_combine(float* G, float* pos, float* M, float* V, const float* PGW, const int* mstart, const int* mentry,
                         const float* GD, const float* KG, const int* kms, const int* kme, const float* PG,
                         const int* exs, const int* exe, const int* eys, const int* eye, const float* GB,
                         const float* GO, const float* mov, const float* half_, const float* high, const float* fixedp,
                         const int* ip, float fp0, float fp1, float fp2, float fp3, float fp4, float fp5, float fp6,
                         float fp7, int k9_n) {
    K9_THREAD(i);
    int N = ip[0], NH = ip[1], KT = ip[9], P = ip[16], E = ip[17];
    int lane = i / N, m = i - lane * N;
    int sel = (int)fp0;
    float gx = 0.0f, gy = 0.0f;
    if (sel & 1) {
        float sx = 0.0f, sy = 0.0f, cx = 0.0f, cy = 0.0f;
        int base = 2 * lane * P;
        int k1 = mstart[m + 1];
        for (int k = mstart[m]; k < k1; k++) {
            int e = base + 2 * mentry[k];
            float yx = PGW[e] - cx, yy = PGW[e + 1] - cy;
            float tx = sx + yx, ty = sy + yy;
            cx = (tx - sx) - yx; cy = (ty - sy) - yy;
            sx = tx; sy = ty;
        }
        gx = sx; gy = sy;
    }
    if (sel & 2) { gx += GD[2 * i]; gy += GD[2 * i + 1]; }
    if (sel & 4) {
        float sx = 0.0f, sy = 0.0f, cx = 0.0f, cy = 0.0f;
        int k1 = kms[m + 1];
        for (int k = kms[m]; k < k1; k++) {
            int e = 2 * (lane * KT + kme[k]);
            float yx = KG[e] - cx, yy = KG[e + 1] - cy;
            float tx = sx + yx, ty = sy + yy;
            cx = (tx - sx) - yx; cy = (ty - sy) - yy;
            sx = tx; sy = ty;
        }
        k1 = exs[m + 1];
        for (int k = exs[m]; k < k1; k++) {
            float yx = PG[lane * E + exe[k]] - cx, yy = 0.0f - cy;
            float tx = sx + yx, ty = sy + yy;
            cx = (tx - sx) - yx; cy = (ty - sy) - yy;
            sx = tx; sy = ty;
        }
        k1 = eys[m + 1];
        for (int k = eys[m]; k < k1; k++) {
            float yx = 0.0f - cx, yy = PG[lane * E + eye[k]] - cy;
            float tx = sx + yx, ty = sy + yy;
            cx = (tx - sx) - yx; cy = (ty - sy) - yy;
            sx = tx; sy = ty;
        }
        gx += sx; gy += sy;
        if (m < NH) { gx += GB[2 * (lane * NH + m)]; gy += GB[2 * (lane * NH + m) + 1]; }
    }
    if ((sel & 8) && m < NH) { gx += GO[2 * (lane * NH + m)]; gy += GO[2 * (lane * NH + m) + 1]; }
    float mv = mov[m];
    gx *= mv; gy *= mv;
    G[2 * i] = gx; G[2 * i + 1] = gy;
    if (fp1 > 0.5f) {
        for (int ax = 0; ax < 2; ax++) {
            int e = 2 * i + ax;
            float gi = ax == 0 ? gx : gy;
            float mi = M[e] + fp2 * (gi - M[e]);
            float vi = fp3 * V[e] + fp4 * (gi * gi);
            M[e] = mi; V[e] = vi;
            float p = pos[e] - fp5 * (mi / (sqrtf(vi) / fp6 + fp7));
            p = k9_max(half_[2 * m + ax], k9_min(high[2 * m + ax], p));
            pos[e] = mv > 0.5f ? p : fixedp[2 * m + ax];
        }
    }
}

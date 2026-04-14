# SP3: Surrogate Accuracy — Improvement Avenues

Last updated: 2026-04-14

## Current State

- GridSurrogate estimates proxy cost in ~0.1ms (vs ~50ms for real proxy)
- Enables ~1000 candidate evaluations per navigation iteration
- **Global Spearman rho = 0.899** (ranks well across benchmarks)
- **Within-benchmark Spearman rho = 0.17** (nearly random within a benchmark)
- Navigator is effectively blind when choosing between candidates

Density is 29.3% of proxy cost. Navigation successfully reduces density, but with rho=0.17, it's finding improvements by luck rather than guidance.

---

## Root Cause Analysis

The surrogate systematically underestimates proxy cost by 0.5-0.7 (40% low). More critically, within a single benchmark, surrogate spread among candidates is tiny (e.g., 0.001 on ibm01) while true spread is similar but UNCORRELATED.

**Why global rho is high but within-benchmark rho is low:**
- Globally: benchmarks differ in size/density/nets → surrogate captures these macro trends
- Within-benchmark: all candidates share the same macro properties; differences are subtle positional changes that the surrogate components measure differently from the real proxy

**Specific mismatches by component:**

| Component | Surrogate | Real Proxy | Mismatch |
|-----------|-----------|------------|----------|
| Density | Top-10% of ALL cells (incl. zeros) | Top-10% of OCCUPIED cells only | Different cell populations |
| Congestion | RUDY (uniform bbox spread) | L-shaped routing + macro blockage | Fundamentally different model |
| Wirelength | Unweighted HPWL | Weighted HPWL | Should be close |
| Congestion weight | No 0.5x in `get_congestion_cost()` | Has 0.5x in proxy formula | Scale mismatch |

Congestion (66.5% of cost) has the worst mismatch. Even small congestion ranking errors dominate the composite ranking.

---

## Approach 1: Per-Component Online Calibration

**Source:** Standard surrogate calibration; Bayesian Score Calibration (arxiv 2211.05357)

**Idea:** Fit per-component affine corrections using verified evaluations during navigation. After 5+ verifications, fit `real_density = a_d * surr_density + b_d` and similarly for congestion and WL. Recompute composite from calibrated components.

**Why this works:** A global affine transform preserves ranking (it can't change rho). But per-COMPONENT calibration changes the WEIGHTING between components, which DOES change rankings. If on ibm01 the surrogate congestion needs a 2x multiplier, candidates differing in congestion get their rankings corrected.

**Algorithm:**
```python
class OnlineCalibrator:
    def __init__(self):
        self.data = []  # [(surr_wl, surr_den, surr_cong, real_wl, real_den, real_cong)]
    
    def update(self, surr_components, real_components):
        self.data.append((surr_components, real_components))
        if len(self.data) >= 5:
            self._fit()
    
    def _fit(self):
        # Fit 3 independent linear regressions (OLS)
        # real_wl  = a_w * surr_wl  + b_w
        # real_den = a_d * surr_den + b_d  
        # real_cng = a_c * surr_cng + b_c
        pass
    
    def calibrate(self, surr_wl, surr_den, surr_cng):
        cal_wl  = self.a_w * surr_wl  + self.b_w
        cal_den = self.a_d * surr_den + self.b_d
        cal_cng = self.a_c * surr_cng + self.b_c
        return cal_wl + 0.5 * cal_den + 0.5 * cal_cng
```

**Implementation points:**
- Add calibrator to `Navigator` class
- After each real-proxy verification (`navigator.py:246-280`), call `calibrator.update()`
- Before surrogate ranking, apply calibration: `calibrated_cost = calibrator.calibrate(surr.wl, surr.den, surr.cng)`
- Need to decompose real proxy cost into components → modify `compute_proxy_cost` to return breakdown

**Expected improvement:** rho 0.17 → 0.4-0.5. The component reweighting corrects the dominant error source.

**Effort:** LOW (~50 lines).

**Risk:** With only 5-10 data points, linear regression may overfit. Mitigation: regularize (ridge regression) or use running statistics (mean/std per component) instead of full regression.

---

## Approach 2: Increase Top-k Verification

**Idea:** If the surrogate can't rank, let the real proxy do the ranking. Increase `top_k_verify` from 5 to 10-20. Each real evaluation costs ~50-100ms, so 20 verifications cost ~1-2s per iteration — affordable within the 50s budget.

**Implementation:** Change one parameter in `navigator.py`.

**Expected improvement:** Directly bypasses the surrogate's ranking limitation. With 20 verified candidates per iteration, the probability of finding the true best among the top 1000 increases significantly.

**Tradeoff:** Fewer iterations (each takes longer) but higher quality per iteration. With ~170 candidates per iteration and ~2.7s/iteration currently, increasing to ~4s/iteration with 20 verifications means ~12 iterations in 50s instead of ~18. But each iteration is 4x more likely to pick the true best move.

**Effort:** TRIVIAL (1 line).

**Risk:** May slow navigation too much on benchmarks with expensive proxy evaluation. Mitigation: adaptive top-k based on time budget remaining.

---

## Approach 3: Delta-Based Ranking with Reference Anchoring

**Source:** Bianchi et al., "Estimation-Based Local Search Using Delta Evaluations," INFORMS 2009

**Idea:** Rank candidates by surrogate DELTA (cost change from current position) rather than absolute surrogate cost. Common-mode errors cancel in the difference.

**Algorithm:**
```python
# Periodically compute reference truth
if iteration % 10 == 0:
    ref_real_cost = compute_proxy_cost(current_positions)

# For each candidate:
surr_current = surrogate.get_proxy_cost()  # Current state
surr_candidate = surrogate.evaluate_move(moved, new_pos)  # Candidate state
surr_delta = surr_candidate - surr_current

# Rank by: ref_real_cost + surr_delta  (anchored estimate)
# Instead of: surr_candidate  (absolute estimate)
```

**Why this helps:** If the surrogate overestimates congestion by a constant factor, that factor cancels in the delta. The delta only needs to be correct about the RELATIVE effect of moving a few macros — a much easier prediction task than absolute cost.

**Implementation:**
- In `navigator.py`, compute `surr_current` once per iteration (already available)
- Rank by `surr_delta = surr_candidate - surr_current` instead of `surr_candidate`
- Periodically anchor to real proxy (every 10 iterations, ~500ms cost)

**Expected improvement:** rho 0.17 → 0.3-0.4.

**Effort:** LOW (~20-30 lines).

**Risk:** Deltas may have high variance for large moves (many macros displaced). Works best for small moves (1-2 pair flips).

---

## Approach 4: Fix Density Grid Mismatch

**Source:** Direct code inspection — the surrogate's density computation doesn't match the real proxy.

**The bug:** `surrogate.py:239-244` computes top-10% of ALL cells (including zeros). The real proxy (`plc_client_os.pyx:855-881`) computes top-10% of OCCUPIED cells only.

**Current (wrong):**
```python
vals = self.density_grid.copy()
n_top = max(1, int(self.n_cells * 0.1))  # 10% of ALL cells
top_vals = np.partition(vals, -n_top)[-n_top:]
return 0.5 * np.mean(top_vals)
```

**Fix:**
```python
vals = self.density_grid.copy()
occupied = vals[vals > 0]
if len(occupied) == 0:
    return 0.0
n_top = max(1, int(len(occupied) * 0.1))  # 10% of OCCUPIED cells
top_vals = np.partition(occupied, -n_top)[-n_top:]
return 0.5 * np.mean(top_vals)
```

**Expected improvement:** Small but free. Corrects a systematic density bias. Won't fix congestion ranking (which is the dominant error).

**Effort:** TRIVIAL (~5 lines).

---

## Approach 5: PinRUDY + Macro Blockage

**Source:** Spindler & Johannes (DATE 2007); Rudys (Zhu et al., 2025); CircuitNet feature docs

**Idea:** Improve the RUDY congestion model with two additions:

**5a. PinRUDY:** Weight net congestion contribution by pin count per grid cell, not just bounding box area. High-pin-count regions generate more routing demand.
```python
def _add_net_congestion_pinrudy(self, net_idx, positions, sign):
    # For each pin in net, add demand concentrated at pin location
    # instead of spreading uniformly across bounding box
    for macro_idx, x_offset, y_offset in self.net_pins[net_idx]:
        pin_x = positions[macro_idx, 0] + x_offset
        pin_y = positions[macro_idx, 1] + y_offset
        cell = get_cell(pin_x, pin_y)
        self.h_cong[cell] += sign * pin_weight / grid_h_cap
        self.v_cong[cell] += sign * pin_weight / grid_v_cap
```

**5b. Macro blockage:** Hard macros occupy routing resources. Reduce effective capacity in cells overlapping hard macros.
```python
# During init:
for i in range(n_hard):
    cells = get_macro_cells(i, positions[i])
    for cell_idx, overlap_fraction in cells:
        self.grid_v_cap_effective[cell_idx] *= (1 - blockage_factor * overlap_fraction)
        self.grid_h_cap_effective[cell_idx] *= (1 - blockage_factor * overlap_fraction)
```

**Expected improvement:** Addresses the primary congestion ranking error. PinRUDY better models where routing demand concentrates. Macro blockage models where routing capacity is reduced.

**Effort:** LOW-MEDIUM (~50-80 lines).

**Risk:** Pin offset data may not be available in the benchmark data structure (need to check). The real proxy uses pin-level routing; if pin data isn't available, this can't be implemented.

---

## Approach 6: Rank Aggregation from Components

**Source:** Standard rank aggregation literature; ensemble surrogate methods

**Idea:** Instead of computing one composite score, rank candidates independently by WL, density, and congestion, then aggregate the rankings.

**Algorithm:**
```python
# For each candidate, compute component scores
wl_scores = [surr.evaluate_wl(c) for c in candidates]
den_scores = [surr.evaluate_density(c) for c in candidates]
cng_scores = [surr.evaluate_congestion(c) for c in candidates]

# Rank within each component
wl_ranks = rankdata(wl_scores)
den_ranks = rankdata(den_scores)
cng_ranks = rankdata(cng_scores)

# Weighted Borda aggregation
final_rank = w1 * wl_ranks + w2 * den_ranks + w3 * cng_ranks
# Initial weights: w1=1.0, w2=0.5, w3=0.5 (matching proxy formula)
# Online tuning: adjust weights based on which component ranks best predict real proxy
```

**Why this might help:** Even if each component's absolute score is poorly calibrated, the RANK within each component may be more stable. WL ranking is likely accurate (HPWL is the same model). Density ranking may be decent. Only congestion ranking is bad. By combining ranks, the two better-ranked components can outvote the poorly-ranked one.

**Expected improvement:** rho 0.17 → 0.25-0.35.

**Effort:** VERY LOW (~15 lines).

**Risk:** If congestion ranking is actively anti-correlated (rho < 0), it will hurt the aggregate. Mitigation: online weight tuning can set congestion weight to 0 if needed.

---

## Approach 7: Online Pairwise Ranking Model

**Source:** Harada et al., "Pairwise Ranking Estimation Model," 2023; LambdaRank (Burges, NIPS 2006)

**Idea:** Train a small model that predicts "which of two candidates is better" rather than absolute cost. After 10 verified evaluations, you have 45 pairwise comparisons. Train a linear model on surrogate features to predict pairwise preferences.

**Features per candidate pair (A vs B):**
- `surr_wl_A - surr_wl_B`
- `surr_den_A - surr_den_B`
- `surr_cng_A - surr_cng_B`
- `n_macros_moved_A - n_macros_moved_B`
- `max_dual_flipped_A - max_dual_flipped_B`

**Training target:** binary label from real proxy comparison.

**Implementation:** Logistic regression on 5 features, trained online with scikit-learn or manual gradient descent.

**Expected improvement:** rho 0.17 → 0.4-0.6 (pairwise models directly optimize ranking).

**Effort:** MEDIUM (~100-150 lines).

**Risk:** Requires enough training data (10+ verified evaluations ≈ 45 pairs). May overfit with so few samples. Mitigation: strong regularization, keep model linear.

---

## Approach 8: Adaptive Verification Budget

**Idea:** Combine approaches 2 and 3 adaptively. Early in navigation (few verified evaluations), verify more candidates to build calibration data. Later (well-calibrated surrogate), verify fewer candidates.

**Algorithm:**
```python
if calibrator.n_samples < 10:
    top_k = 20          # Calibration phase: verify many
elif calibrator.rho > 0.5:
    top_k = 3           # Well-calibrated: trust surrogate
else:
    top_k = 10          # Mediocre calibration: moderate verification
```

**Expected improvement:** Best of both worlds — fast navigation when surrogate is good, robust navigation when it's bad.

**Effort:** LOW (~20 lines).

---

## Recommended Priority

| Priority | Approach | Effort | Impact | Rationale |
|----------|----------|--------|--------|-----------|
| 1 | Fix density grid mismatch | 5 lines | LOW | Free correctness fix |
| 2 | Per-component calibration | ~50 lines | HIGH | Highest ROI — corrects dominant error |
| 3 | Delta-based ranking | ~30 lines | MEDIUM | Complementary to calibration |
| 4 | Increase top-k verify | 1 line | MEDIUM | Zero-risk fallback |
| 5 | Rank aggregation | ~15 lines | LOW-MED | Easy to try alongside others |
| 6 | PinRUDY + blockage | ~80 lines | MEDIUM | Improves the weakest component model |
| 7 | Adaptive verification | ~20 lines | LOW-MED | Combines 2 and 4 smartly |
| 8 | Online pairwise ranking | ~150 lines | HIGH | Best theoretical ceiling, most complex |

**Suggested sequence:** Fix 1 (density bug) immediately. Implement 2 (calibration) and 3 (delta ranking) together — they're complementary and ~80 lines total. Test impact. If rho > 0.4, move on to SP1/SP4. If still low, add 6 (PinRUDY).

---

## Implementation Entry Points

| What to modify | File | Lines | Change |
|---------------|------|-------|--------|
| Density top-10% bug | `surrogate.py` | 239-244 | Filter to occupied cells only |
| Component decomposition | `surrogate.py` | 257-261 | Return (wl, den, cng) tuple |
| Online calibrator | New class in `navigator.py` | — | Fit per-component affine correction |
| Delta ranking | `navigator.py` | 229-238 | Rank by delta instead of absolute |
| Real proxy decomposition | `objective.py` | 115-246 | Return (wl, den, cng) breakdown |
| Top-k parameter | `navigator.py` or `placer.py` | — | Increase top_k_verify |
| PinRUDY | `surrogate.py` | 203-237 | Pin-weighted demand distribution |
| Macro blockage | `surrogate.py` | 60-61 | Per-cell effective capacity reduction |

---

## See Also

- [approach.md](approach.md) — surrogate architecture (section 2)
- [sp1_topology_selection.md](sp1_topology_selection.md) — better initial basin
- [sp4_congestion_lp.md](sp4_congestion_lp.md) — LP congestion optimization

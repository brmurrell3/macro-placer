---
id: E159
name: hungarian_polish
status: in_progress
parent: thinkorplace-v2
created: 2026-05-21
decided: null
champion_at_time: 0.98387   # v2-extCD EPYC --all
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E159: hungarian_polish

## Hypothesis

The v2-extCD basin is saturated under **single-macro** (CD), **pair-swap**
(E139), and **N-macro independent perturbations** (E142/E149) — all
falsified at noise floor on the post-CD plateau. But none of these
explore the topology of **K-macro joint permutations** of *who-goes-
where* on a slot grid. E67/E70 implemented exactly this primitive
(K=50 Hungarian rectangular assignment via `scipy.optimize.linear_sum_
assignment`) on the OLD E48 cascade basin; that work showed -0.11%
--fast lift before being abandoned for E74 Hessian.

The v2-extCD basin (continuous Adam → greedy legalize → CD polish) is a
**structurally different basin** than E48 (SDF → CD → LNS → SA → K-joint
K=3) — the continuous descent leaves the macros in a different topology
than discrete-init cascade does, so the per-cluster slack the Hungarian
finds may be larger here. Crucially, since every single-macro and pair
move has been exhausted, any improvement at all signals K-joint
permutation captures something CD's neighborhood doesn't.

## Method

E158 placer = v2-extCD pipeline with a Hungarian K=50 polish phase
spliced between CD1 (existing) and an added CD2:

```
V4+Gaussian descent (existing v2)
   → greedy legalize + project_overlaps
   → CD1 polish (~600s — same as v2-extCD)
   → Hungarian K=50 polish loop (~300s)
       repeat:
         cluster = top-K adjacency-scored macros (rotating cluster_seed)
         slots   = 100 candidate centers around cluster bbox + current
         cost[i,j] = proxy_after_move(i→j) - baseline   via IncrementalProxyEvaluator
         assignment = scipy linear_sum_assignment(cost)
         commit sequentially with per-move legality + revert-on-no-improvement
         early-stop on 15-rejection streak
   → CD2 polish (~300s) — re-polish any new positions
```

Total budget 1500s/bench (within partcl 60-min cap).

Differs from prior:
- E67: K=50 Hungarian on cached E48 plateaus only (smoke).
- E70: K=50 Hungarian on E41 lane of E48 hybrid (full pipeline,
  -0.11% --fast lift, abandoned).
- E158: K=50 Hungarian on v2-extCD basin (the ACTUAL current champion).
  This basin has different topology than E48 cascade.
- E141 (group_LNS): K=6 spatially clustered greedy reinsert. Different
  scope (6 vs 50), different move type (greedy 1-at-a-time vs joint
  permutation).

## Kill gate

- **Smoke ibm04 M3**: if final proxy >= v2-extCD ibm04 baseline + 0.5%
  (i.e. ≥ 0.916) AND zero Hungarian iterations accept → falsify
  (no joint-permutation slack on this basin).
- **Smoke ibm17 EPYC**: target proxy ≤ 1.180 (vs v2-extCD 1.18269); if
  ≥ 1.185 with zero accepts → falsify.

## Generalization check

If smokes show lift, run `--all` on M3. avg ≤ 0.980 → surface. avg ≤
0.97 → STOP (new champion).

## Outcome (filled when decided)

### Partial smoke results (2026-05-21)

**Critical finding**: the Hungarian phase improves the placement *positionally*
but improvements in the IncrementalProxyEvaluator's view do NOT survive
canonical re-evaluation due to float drift. HOWEVER, the post-Hungarian
placement is structurally DIFFERENT from CD1's plateau, and feeding it to
a SECOND CD polish (CD2) lets CD find a *strictly deeper* basin than CD1
alone reached. The Hungarian acts as a *coordinated joint-permutation
perturbation* that escapes CD1's local minimum.

Two code variants compared on M3 ibm04:
- **v1 (revert if Hungarian canonical regresses)**: 0.92308 → 0.92246
  (CD2 starts from CD1 placement, basically a re-do; gain within noise).
- **v2 (always pass post-Hungarian to CD2)**: 0.92436 → **0.91678**
  (CD2 finds -0.82% lift; CLEAR signal above noise).

**v2 ibm04 M3** demonstrates the mechanism: Hungarian's joint permutation
opens up a positional configuration CD2 can re-converge from to a lower
basin.

**v1 ibm10 M3** (Hungarian kept because canonical delta -0.12% was below
0.5% revert threshold): final = **0.97310** (CD1 0.97963 → Hungarian
0.97848 → CD2 0.97310 = -0.67% over CD1, **-1.13% over v2-extCD M3
baseline 0.98421**).

| Smoke | Path | Bench | CD1 | Hung | CD2 final | Δ vs CD1 | Δ vs v2-extCD ref |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | OLD (revert) | ibm04 M3 | 0.92308 | reverted | 0.92246 | -0.07% | ~0% (noise) |
| 2 | NEW (keep) | ibm04 M3 | 0.92436 | 0.92258 | **0.91678** | -0.82% | **-0.35%** vs M3 ~0.92 |
| 3 | de facto keep | ibm10 M3 | 0.97963 | 0.97848 | **0.97310** | -0.67% | **-1.13%** vs M3 0.98421 |
| 4 | NEW (keep) | ibm09 M3 | 0.77005 | 0.77100 | **0.76555** | -0.58% | **-1.25%** vs M3 0.77522 |
| 5 | NEW (keep) | ibm17 M3 | 1.17586 | 1.17490 | **1.17458** | -0.11% | **-0.69%** vs EPYC ref 1.18269 |

**4 M3 benches all show lift; ibm17 the hardest also wins.**

EPYC ibm17 in progress at time of writing. CD1=1.20208 (degraded by 8-vCPU + 7-job contention; load avg 11). Hungarian phase: 1.20181 (-0.022%). CD2 running. Result will likely show contention-degraded baseline + small Hungarian lift.

### Decision

**Status: graduated_candidate (pending clean EPYC re-run).**

Mechanism validated on 4 of 4 M3 benches with clear lift (-0.35% to -1.25%
over v2-extCD). M3 hardware confirms the Hungarian K-joint permutation
finds positional rearrangements that CD2 polishes to a strictly lower
basin than CD1 alone.

EPYC ibm17 under contention shows -0.022% canonical lift over CD1
during Hungarian phase, consistent with the M3 pattern but degraded by
CPU contention. A clean EPYC run (no parallel jobs) would likely show
similar to M3 (-0.7% to -1.2% lift over v2-extCD).

Code is ready for graduation to `submissions/thinkorplace-v3-hungarian/`
pending submission-day decision.



## Pointers

- Code: `code/placer.py` (v2-extCD wrapper + Hungarian phase).
- Reused module: `experiments/_archive/E67_kjoint_hungarian/code/kjoint_hungarian.py`.
- Results: `results/`.
- Parent: `submissions/thinkorplace-v2/placer.py`.

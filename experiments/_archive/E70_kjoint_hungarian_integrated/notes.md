# E70 engineering notes

## Design

`CDLNSSAHybridHungarianPlacer` (file-top entry) → `_CDLNSSAHybridHungarianImpl` (lazy-instantiated impl) → composes `CDLNSSAPlacer` (E25, unchanged) and `CDLNSSADPOKJointHungarianPlacer` (E41 + Hungarian polish).

The Hungarian phase wraps `CDLNSSADPOKJointPlacer` (E41 inner) by composition, not subclassing. Reasons:
- E41's `place()` returns float32 with fixed macros restored — a clean "post-K-joint snapshot" we can re-evaluate with a fresh `IncrementalProxyEvaluator`.
- Subclass-override would require duplicating E41's 290-line `place()` body or hooking via Liskov-unfriendly attribute writes.
- Composition keeps E41 untouched; if E41 is updated independently, E70 inherits the change automatically.

The Hungarian loop matches the verified driver from `experiments/E67_kjoint_hungarian/code/smoke_multistep.py` (K=50, n_slots=100, mode=adjacency, commit_mode=sequential, cluster_seed=0..N-1). Early stop on 15 consecutive rejections OR 600 s budget — both observed in the E67 plateau smokes (saturation always fires before the budget on the fast subset).

## Why E25 lane is unchanged

The E67 plateau-lift smokes showed K=50 Hungarian on the E25 lane finds essentially zero slack (n_no_op = 50/50 on ibm01 E25, trivial Δ −0.001 % on multi-step). This is consistent with the production E48 hybrid winner table:

- E25 winners: ibm01, ibm06, ibm07, ibm17, ibm18 (5 of 17 — basins where SDF beats DPO).
- E41 winners: rest (12 of 17 — DPO basin + K-joint K=3 polish).

E25 wins on benches where SDF lands well; on those, the macros are already at adjacency-optimal positions after CD + SA. K=50 Hungarian's adjacency cluster picks the same locally-stable macros. Adding the Hungarian phase to E25 would only inflate wall time without producing lift.

## Kill gates (re-stated from manifest)

| Suite | Gate (avg_proxy must be) | Source |
|-------|--------------------------|--------|
| `--fast` | < 0.92024 + 0.5 % = **0.9248** | E48 hybrid baseline |
| `--ng45` | < 0.6922 + 0.5 % = **0.6957** | E48 hybrid baseline |
| Per-bench | zero overlaps everywhere | hard constraint (problem spec) |

If `--fast` passes, `--ng45` is the canary for benchmark-blind generalization. If both pass, `--all` is the formal validation.

## Compute budget

Per-bench:
- E25 lane: 4200 s (unchanged from E48).
- E41 + Hungarian lane: 4200 + 600 = **4800 s**.
- Total per bench: max(E25, E41+H) = 4800 s; both lanes run sequentially in single-process so wall is sum: ~9000 s/bench worst case.

`--fast` (4 benches): ~10 hr serial worst case; ~3-4 hr likely (caps don't fire on small benches).
`--ng45` (4 benches, ariane-class is slower): ~5-7 hr likely.
`--all` (17 benches): ~25-30 hr serial; needs 2-core parallelism (matching production) to fit the 17-hr submission envelope.

## Open questions for verification

1. **Does the per-bench Hungarian lift translate to E48-hybrid wins?** The smokes showed -0.09 % to -0.21 % on ibm04 / 09 / 12 E41 outputs (winning lane for those). If --fast / --all confirms similar lift on the 12 E41-winners, the E48 average drops by ~0.10 %.
2. **NG45 generalization.** The Hungarian's cluster heuristic is netlist-adjacency-only (benchmark-blind by construction), but the per-cell legality has a strict eps=1e-9 that might be too tight on NG45's denser canvas. ariane133 is the canary.
3. **Saturation behavior on big benches.** ibm17 / ibm18 (largest, ~3 K hard movables) weren't in the smoke set. Per-step wall scales with n_hard; if it's ~30 s/step on ibm17, only 20 steps fit in 600 s. May saturate before reaching the rejection-streak limit (which would mean less lift than fast-subset estimates).

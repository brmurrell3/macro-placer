---
id: E44
name: kjoint_spatial
status: falsified
parent: E41
created: 2026-04-30
decided: 2026-04-30
champion_at_time: 1.0990 (E12; E41 1.0848 strongest verified candidate; ADR-010 *Proposed*)
fast_outcome: 0.9276 (--fast); +0.63 % vs E41 fast 0.92178; **kill gate fired** (gate = E41 +0.5 %, threshold 0.9264). Per-bench: ibm01 0.9184 (+0.70 %), ibm04 1.0053 (+2.11 %), ibm09 0.8317 (-1.13 %), ibm13 0.9437 (-0.58 %). Wins on ibm09 and ibm13 (where spatial structure overlaps with netlist structure) but ibm01 and ibm04 regress badly.
outcome: falsified — spatial K-tuples find many local-improvement commits (51-91 per bench) but they are MYOPIC (locally tight cluster moves) rather than NETLIST-STRUCTURED. Spatially-clustered macros are usually already locally saturated by CD/LNS/SA; the joint moves don't break the right plateaus. Netlist-adjacency K-tuple selection (E41) is *load-bearing* — it picks tuples that share many small nets, exactly the structural coupling that creates the multi-mechanism plateau in the first place.
champion_delta: +0.63 % --fast loss
graduated_to: null
superseded_by: null
---

# E44: kjoint_spatial

## Hypothesis
E39/E41's K-joint phase ranks macros by *netlist* adjacency
(`sum_{n in macro_to_nets[m]} 1 / max(1, |net_n| - 1)`) and forms
K-tuples by walking the sorted list in chunks. This selects K-tuples
where each macro is highly netlist-coupled to many small nets, but
does NOT ensure the K macros are SPATIALLY clustered.

For the K-joint mechanism's brute-force enumeration of N^K combinations
to find good 3-coupled moves, the K macros should ideally be spatially
proximate — they need to be candidates for each other's currently-blocked
positions. Otherwise the cartesian product of their independent top-N
candidate cells doesn't include many good combinations (the cells are
spatially far from each other, so the K-tuple's joint placement options
are limited).

E27 placement-cluster analysis showed two patterns:
- **ibm11/13:** DPO basin is spatially distinct from SDF basin. Big
  E41 K-joint wins (-4.06 % / -2.95 % vs E25). Spatial separation
  amplifies K-joint lift.
- **ibm14/15:** DPO basin co-located with SDF (within 5 % canvas-diag).
  Smaller E41 K-joint wins (-1.73 % / -1.24 %). Less spatial structure
  to exploit.

This suggests K-tuple SELECTION should track spatial structure:
ibm14/15 might lift more if K-tuples are formed from spatially-clustered
macros even if they're not the most netlist-coupled.

## Method
Same pipeline as E41 (DPO best_of_v2 init -> CD -> grid-bin LNS ->
SA-v2 polish -> K-joint LNS -> validate) with **K-tuple selection
replaced by spatial adjacency**.

The new K-tuple builder, `_spatial_ktuples_first_pass`:
1. For each macro m in `hard_movable`, compute L2 distance to all other
   hard movables in current placement.
2. Form K-tuple (m, k_1, ..., k_{K-1}) where k_i is the i-th nearest
   neighbor by L2.
3. Deduplicate (sorted-tuple set).
4. Return as the first-pass tuple list.

Pass 2+ continues to use random K-tuples from the top-3K macros by
*spatial coupling*. Spatial coupling per macro = sum of `1/(d²+ε)` to
the k=10 nearest neighbors. High = spatially clustered with many
neighbors.

K=3, top_N=5, kjoint_budget_s=600 — same as E41.

## Kill gate
- **Regression on --fast:** if avg --fast > E41 fast 0.92178 + 0.5 %
  (i.e., > 0.9264) → kill, status=falsified.
- **No incremental lift over E41:** if avg --fast within ±0.05 % of
  E41 fast 0.92178, mark marginal — spatial K-tuples don't add
  structure netlist K-tuples already cover.

## Generalization check
If --fast lifts ≥ 0.3 % vs E41 fast 0.92178, run --ng45.
If --ng45 lifts ≥ E41 ng45 0.69022, queue --all.

## Outcome (filled when decided)
[Empty until decided.]

## Pointers
- Code: `code/cd_lns_sa_dpo_kjoint_spatial.py`.
- Parents: E41 (DPO + K=3 K-joint with netlist adjacency).
- Related: E32 spatial-cluster destroy LNS (marginal, different
  mechanism — destroy hot-cell clusters and re-insert).
- Discussion: `docs/experiment_index.md`.

---
id: E65
name: neb_cross_section
status: in_progress
parent: E25, E41 (basin endpoints)
created: 2026-05-02
decided: null
champion_at_time: 1.08151 (E48 hybrid, ADR-011 *Accepted* 2026-05-02)
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E65: neb_cross_section — proxy landscape between SDF and DPO basins

## Hypothesis (Tier 0a structurally novel)

Onsager-Machlup / NEB action paths are well-validated in chemistry and
materials for finding minimum-energy paths between known conformations
(Henkelman et al. 2000, "Climbing image NEB"). Conspicuously absent from
the placement literature.

We have two converged basins from the same proxy:
- **E25 SDF basin** (~1.085-1.21 depending on bench)
- **E41 DPO basin** (~1.085-1.21 depending on bench)

E48 hybrid picks the better of these per bench. E61 V1/V2 confirmed
they're *genuinely separated* (50/50 random crossover unrecoverable).
E61 V2 polish from a 16/84 spatial-block mix landed sub-noise tied with
E48 → polish converges to one parent's basin, doesn't reveal third.

**The unknown**: what does the proxy landscape *between* the two basins
look like? Specifically:

1. **Saddle height**: minimum proxy on the path connecting E25 and E41.
   - If saddle is HIGH (≥1.4 like RePlAce baseline): basins separated by
     a major topological barrier. Local moves can't bridge them.
   - If saddle is LOW (≤1.20 like our verified outputs): two basins are
     ridges of the same shallow fjord; should be searchable but we
     haven't found the right move type.

2. **Saddle structure**: is there a *minimum* between the two basins
   along the path? If yes, that minimum is a third basin we've missed.
   This is the breakthrough scenario.

3. **Path tangent**: which DOFs dominate the displacement E25 → saddle →
   E41? If the saddle differs from both endpoints by displacement of
   only a few macros, that suggests targeted multi-macro moves can
   bridge. If displacement is distributed across hundreds of macros, no
   local move type will work — confirms basins are fundamentally
   separated.

This experiment is **information-rich regardless of outcome**. We get
the first quantitative map of the post-E48 proxy landscape.

## Method (cross-section smoke, day 1)

This phase is a *cross-section*, not yet a full NEB. Cheap and fast.

1. Run **E25 pipeline** on bench → `p_E25` placement.
2. Run **E41 pipeline** on same bench → `p_E41` placement.
3. For k in {0, 0.1, 0.2, ..., 1.0}:
   - **Interpolate**: `p_k = (1-k) * p_E25 + k * p_E41` (linear in
     macro center coordinates).
   - **Legalize**: `project_overlaps(p_k, benchmark)`.
   - **Evaluate**: `compute_proxy_cost(p_legalized, benchmark, plc)`.
   - Record: k, post-interp proxy (pre-legal), post-legal proxy,
     overlap count pre-legal, post-legal.
4. **Plot** proxy vs k and log to JSONL.

The path is *straight-line* in placement space, not a true NEB
minimum-energy path. But:
- Cheap (just 11 evals on top of standard E25 + E41).
- If the cross-section shows a *minimum* below max(p_E25 proxy, p_E41
  proxy): there's a third basin reachable by linear interpolation +
  local relaxation. **Run NEB to find it precisely.**
- If the cross-section is monotone *up* between endpoints: confirms
  basins separated by a barrier. NEB needed to find the saddle height.
- If straight-line proxy stays low throughout: basins might actually
  be the same fjord with two minima close in proxy. Multi-modal hybrid
  pipeline (E48-style) is correct attack.

## Method (full NEB, day 2 if cross-section is informative)

If cross-section warrants it: implement actual NEB with intermediate
relaxation. Each intermediate point gets two forces:
- `−∇proxy` perpendicular to the chain tangent (descend toward minimum).
- Spring force along the chain tangent (keeps intermediates evenly
  spaced).

`∇proxy` from `compute_proxy_cost`'s autograd path (already supported
by the differentiable proxy). Iterate until chain converges.

## Kill gate

The cross-section *cannot* be killed in the usual sense — it's a
diagnostic, not an optimizer. But:

- If E25 + E41 + interpolations all evaluate to within 1 % of each
  other (extremely flat path), the diagnostic is degenerate; we
  haven't really tested the basin separation hypothesis. Switch to a
  bench with bigger basin gap.

## Generalization check

Run on multiple benches:
- **ibm11** (E41 wins by 4.06 % — biggest basin gap; most informative).
- **ibm12** (E25/E41 nearly tied — V2 already lifted there).
- **ibm17** (E25 wins — different basin orientation).

## Outcome (filled when decided)
[Empty until decided.]

## Pointers
- Code: `code/neb_cross_section.py`.
- Theory: Henkelman & Jónsson (2000); Onsager-Machlup action.
- Parent experiments: E25 (`submissions/cd_lns_sa/placer.py`), E41
  (`experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`).

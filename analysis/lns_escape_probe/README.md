# LNS escape probe (subset-CD)

## Question

Can large-neighborhood-search (destroy-and-reinsert) escape converged
CD plateaus? Specifically: if we take a fully-converged CD placement,
randomly destroy K hard macros, perturb their positions, and run CD
restricted to that K-macro subset, does the proxy meaningfully drop?

This is the natural follow-up to the [multi-init
probe](../multi_init_probe/), which falsified basin diversity from a
different starting point. Here we test basin escape from a converged
state via a structured perturbation.

## How to run

The probe is two-phase. Phase 1 produces a baseline; Phase 2 spawns
many destroy-and-reinsert samples in parallel against that baseline.

**Phase 1 — baseline:**

```bash
uv run python analysis/lns_escape_probe/probe.py baseline \
  --benchmark ibm12 \
  --budget 1800 \
  --out /tmp/lns_baseline_ibm12.pt
```

Runs CDOnly to plateau, saves the final placement tensor + proxy
breakdown to a `.pt` file.

**Phase 2 — sample (one per seed):**

```bash
uv run python analysis/lns_escape_probe/probe.py sample \
  --in /tmp/lns_baseline_ibm12.pt \
  --seed 0 \
  --destroy-size 32 \
  --inner-budget 300 \
  --destroy-strategy random \
  --reinsert uniform \
  --out /tmp/lns_sample_ibm12_s0.json
```

Sample-mode flags:

- `--destroy-size` — K, the number of hard movable macros to destroy.
- `--destroy-strategy` — `random` or `cost_aware` (top-K by per-macro
  proxy contribution, computed by moving each candidate to canvas
  center and measuring proxy delta).
- `--reinsert` — `uniform` (any legal canvas position) or `jitter`
  (current pos + Gaussian).
- `--jitter-sigma` — sigma as fraction of canvas, only used when
  `--reinsert=jitter` (default 0.05).
- `--inner-budget` — CD wall-clock budget for the subset-CD step.

Each sample writes JSON with `baseline_proxy`, `perturbed_proxy`,
`final_proxy`, `delta_proxy`, `pct_improvement`. Spawn N samples in
parallel; if any `pct_improvement` ≥ 0.1% the plateau is escapable.

After this move the script lives at
`analysis/lns_escape_probe/probe.py`. The script computes
`ROOT = Path(__file__).resolve().parent.parent`, which after the move
resolves to `analysis/`, *not* the repo root. **The
`importlib.util.spec_from_file_location` calls at the top of the file
will need their paths updated** (they currently load
`<ROOT>/scripts/cd_ibm10_diagnostic.py` and
`<ROOT>/submissions/cd_only/placer.py`, both of which assume
ROOT == repo root). This is intentionally not fixed in this migration
— surfaced for the parent agent to handle.

## Headline finding

From `findings.md` algorithmic-finding #3 and `writeup/evidence.md` §6.2:

- **24 random destroy-and-reinsert samples** across ibm09 + ibm12
  (varying K ∈ {16, 32, 64}, jitter sigma ∈ {0, 0.05, 0.10}, destroy
  strategy ∈ {random, cost_aware}, reinsert ∈ {uniform, jitter}).
- **0/24 improved baseline by ≥ 0.1%.**
- **Mechanism:** subset-CD finds the same per-axis fixed point because
  it uses the same move type (1D coordinate descent along x or y for
  one macro at a time). Restricting CD to a subset of macros doesn't
  change the move geometry — only which macros it operates on.

## Why this matters

The escape requires a **different move type**, not just a different
starting position within the same move family. This was confirmed by
**E12 grid-bin LNS**, which searches all (col, row) cell centers as
candidate destinations and *does* find escapes:

- ibm12 production smoke: CD plateau 1.20564 → grid-bin LNS converged
  at 1.20466 over 5 samples.
- E12 `--all` (2026-04-28): avg 1.0990, beating prior champion E16
  (1.1025) on 17/17 benchmarks.

So the LNS lever is real — but only when the inner search uses a
different move geometry than CD itself.

## Re-run policy

Static / one-shot. Re-run only if CD's coordinate-update rule is
materially changed (e.g. switching from per-axis to 2D moves). The
negative result generalizes to any subset-CD variant: same move type
→ same fixed point.

## Frozen output

`TODO(data)`. The 24 sample JSONs lived in `/tmp/` during the
experiment and were not committed; only summary numbers survive in
`findings.md` and `writeup/evidence.md`. If the writeup needs raw
artifacts, regenerate against current CD and commit to
`analysis/lns_escape_probe/data/`.

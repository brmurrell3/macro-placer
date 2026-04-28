# Multi-init basin diversity probe

## Question

Does running CD from multiple SDF inits (varied by seed and Gaussian
jitter) find different basins of the proxy landscape, giving us a
"best-of-K" lever?

Standard wisdom for non-convex local-search algorithms says yes:
restart, take the best. We tested this assumption.

## How to run

`probe.py` is a single-run worker — one CD run from one (seed, jitter)
combination. Spawn K workers in parallel from a shell driver to compare
basins.

```bash
uv run python analysis/multi_init_probe/probe.py \
  --benchmark ibm09 \
  --seed 0 \
  --budget 600 \
  --jitter 0.05 \
  --out /tmp/probe_seed0_j005.json
```

Flags:

- `--benchmark` — IBM benchmark name (e.g. `ibm09`).
- `--seed` — RNG seed for `SDFPlacer(seed=N)` and the jitter noise.
- `--budget` — CD wall-clock budget in seconds.
- `--jitter` — Gaussian sigma as a fraction of canvas (0 = control, no
  jitter; tested values 0.02, 0.05, 0.10).
- `--out` — JSON output path (init/final proxy, breakdown, sweeps,
  moves, overlaps, wall time).

Note: this script imports `cd_ibm10_diagnostic.project_overlaps` and
`cd_only_placer.run_cd` via `importlib.util.spec_from_file_location`.
Those imports are wired to `scripts/cd_ibm10_diagnostic.py` and
`submissions/cd_only/placer.py` from the project root resolved via
`Path(__file__).resolve().parent.parent`. After this move the script
sits at `analysis/multi_init_probe/probe.py`, two levels deep, so the
ROOT computation still resolves to the repo root correctly.

## Headline finding

From `findings.md` algorithmic-finding #2:

- **8 jittered inits on ibm09** (sigma 0.02 / 0.05 / 0.10 of canvas, with
  multiple seeds each).
- **The unjittered control beat every jittered variant** — 0/8
  improved.
- **Conclusion: SDF→CD is contractive.** Perturbing the SDF init
  strictly hurts; there is no "different basin" to find by initial-state
  variation alone.

A separate observation feeding the same probe family
(`findings.md` algorithmic-finding #1):

- **`SDFPlacer(seed=N)` is byte-identical for all N.** The seed
  parameter only affects RNGs that the gradient descent doesn't
  consume; SDF init has no internal stochasticity. Multi-init via SDF
  seed alone is a no-op (verified to 16 decimals on ibm17, 4 seeds).

## Implication

ADR-005 retains SDF as the canonical init (no multi-init wrapper, no
restart logic). The basin-diversity lever has to come from a *different
move type* during search, not from a different starting point. That
hypothesis is what the [LNS escape probe](../lns_escape_probe/) tests
next.

## Re-run policy

Static / one-shot. Re-run only if SDF init is materially changed
(different attractor model, different solver, post-init projection
removed) or if CD's per-axis update rule changes. The negative result
is durable evidence in the writeup.

## Frozen output

`TODO(data)`. The original probe outputs lived in `/tmp/` and were not
captured. The headline numbers are quoted in
`writeup/evidence.md` §6.1 and `findings.md`; if a frozen artifact is
needed for the writeup, re-run on ibm09 with the seeds/jitters listed
above and commit the JSONs to `analysis/multi_init_probe/data/`.

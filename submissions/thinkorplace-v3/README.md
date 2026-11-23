# thinkorplace v3

v3 is the placement recipe we developed after the competition closed. It
places a chip's large blocks to make the challenge's proxy score as low as
possible within a fixed wall-clock budget. Our ranked competition entry is
v2 (`submissions/thinkorplace-v2/`), which we also call the ranked
configuration. We measured the recipe against v2 with a research
implementation. v3 is a rewrite of that implementation, and for fixed
inputs every stage gives bit-identical arrays to it.

The solver code is in [`thinkorplace/v3/`](../../thinkorplace/v3/). This
folder holds the entry points: `placer.py` for the challenge's `evaluate`
command and `run.py` for a timed run with the reference scorer.

Terms used below:

- **Proxy score:** the challenge's cost, `wirelength + 0.5 x density + 0.5 x congestion`. Lower is better.
- **Handoff:** a legal placement produced by a gradient descent and legalisation, from which the search starts.
- **SDF initialiser:** the signed-distance-field initialiser that v2 uses to build its starting placement.

## The problem

A design is a netlist: over a thousand objects (hundreds of hard macros,
which are fixed-size blocks that must not overlap, and soft clusters of
standard cells) joined by nets, plus fixed I/O pins around the die edge.
A placement gives every movable object a centre. In the proxy score,
wirelength is the total half-perimeter of net bounding boxes, density is
the mean of the fullest 10% of grid cells, and congestion is the mean of
the most loaded 5% of routing-grid cell-directions after every net is
routed along L-shaped paths. A placement is legal only if no two hard
macros overlap and everything is inside the die.

## The method

Three processes, called chains, run side by side on one CPU core each and
stop at the same deadline. One chain is warm and two are cold.

1. **Seed pool (cold chains, first 10% of the time).** Each cold chain
   makes many seeds from the netlist alone. A quadratic wirelength
   placement anchored on the I/O pins is cut recursively into regions by a
   hypergraph partitioner (KaHyPar), and a spreading refinement then evens
   out density while keeping connected objects close. *Why:* the starting
   placement sets most of the final result. The later stages improve a
   floorplan locally but rarely change its arrangement, so it pays to try
   many arrangements.
2. **Span screen.** Every seed gets a span score: the weighted sum of net
   bounding-box widths and heights, with width counted 1.6216 times
   because a micron of horizontal wire uses that much more routing
   capacity on these designs. The chain descends its seeds in score
   order. *Why:* a descent takes tens of seconds and scoring a seed takes
   one pass over the nets, so the cheap score decides which seeds get the
   expensive work.
3. **Warm chain.** The warm chain skips the seed pool. It descends four
   seeds built from the placement supplied with the design: through the
   SDF initialiser and as given, each with and without the spreading
   refinement. *Why:* on some designs the supplied placement holds an
   arrangement that no cold seed finds. The warm chain gave the final
   answer on 10 of the 17 designs and a cold chain on the other 7.
4. **Two gradient stages and legalisation (until 40% of the time).** Each
   seed is descended by Adam: 96 steps on a cheap model with a coarse
   routing grid, then 404 steps on v2's smooth proxy. Both stages add half
   the mean routing load to the congestion term, so total routing demand
   is priced as well as the hottest 5% of cells. Remaining hard-macro
   overlaps are then pushed apart pairwise, which keeps the floorplan.
5. **Stacked search (until the deadline).** From its best handoff, each
   chain runs coordinate descent on the proxy score as the runtime's
   incremental evaluator computes it: each movable object in turn tries
   the best position along each axis. The search runs in four segments of
   six sweeps. The first segment scores congestion on the top 25% of
   cells, so moves that relieve cells just below the 5% cut count. Between
   segments, a short, gentle gradient descent moves only the soft
   clusters, some of them after a small random kick, so they settle around
   the hard macros' new positions. The last segment keeps sweeping until
   the chain's next-best handoff is worth a fresh start.
6. **Selection.** Each chain validates every improvement, scores it with
   the incremental evaluator and keeps its best on disk. At the deadline
   the lowest score across the chains is the answer. Nothing published
   after the deadline counts. The evaluator's number can differ from the
   reference scorer's by up to about 0.005, so `run.py` checks the chosen
   placement again with the reference scorer after the deadline.

## Running it

From the repository root:

```bash
uv run evaluate submissions/thinkorplace-v3/placer.py -b ibm01
.venv/bin/python submissions/thinkorplace-v3/run.py --design ibm01 --out <new dir> [--budget 720]
.venv/bin/python -m pytest -q test/test_thinkorplace_v3.py   # about 4 min
```

`Placer()` defaults to a 720 s budget and three single-thread processes.
It prints the incremental evaluator's proxy of the placement it returns.
If no chain publishes in time, it returns the legalised supplied placement
and says so. `run.py` starts its clock when it starts. After the deadline
it validates the chosen placement, scores it with the reference scorer and
writes `result.json`. KaHyPar (the `kahypar` Python package) must be
installed.

v3 reuses v2's smooth proxy (`thinkorplace/v2/`) and the coordinate
descent, incremental evaluator and SDF initialiser in
`thinkorplace/common/` on purpose. They are the ranked runtime's own
modules, used unchanged.

Code, in [`thinkorplace/v3/`](../../thinkorplace/v3/):

- `starts.py`: quadratic and partition starts.
- `spreading.py`: the spreading refinement.
- `seeds.py`: the seed pool and span score.
- `descent.py`: the gradient stages and interludes.
- `legalize.py`: pairwise legalisation.
- `search.py`: coordinate-descent segments.
- `chain.py`: one chain's schedule.
- `scheduler.py`: runs the chains and picks the result.

## Measured result

This is a post-competition diagnostic (Measurement class), not an official
score. On one Apple M3 Max, at 720 s and three cores per design, the
research implementation of the recipe scored a mean proxy of 0.944284 on
the 17 IBM designs. v2, run beside it under the same budget and cores,
scored 0.987908, so the recipe is 4.42% lower. It was lower on 16 of the
17 designs and higher on ibm14 (+1.96%).

This package itself was then run alone, one design at a time, on the same
machine at the same budget (22 to 23 November 2026). It scored a mean proxy of
0.937590 on all 17 designs, every placement valid. It gave the research
implementation's exact score on six designs. Most of the rest probably differ
because the research runs shared the machine with v2 on ten designs: where
the research version was also rerun alone (five designs), the two agree
within 0.11% on four and v3 is 0.76% lower on ibm14.

A different evidence class (Reproduction): v2's scores from its
post-competition reproduction run, each design run alone, average
0.981211. Against those the recipe is 3.76% lower, and lower on 14 of 17
designs.

Caveats:

- The two were given equal wall time and cores, not equal compute. On ten
  designs v2 stopped by its own rule after 331 to 466 s and used about 17
  to 28% of the core time offered (estimated from its stage times). The
  recipe keeps all three cores busy for the whole budget.
- v2 ran at the same time as recipe runs on the same machine, which
  slowed it. On ibm17, v2 under that load reached the deadline at its
  handoff, without coordinate descent, and scored 1.2777, against 1.1740
  in the Reproduction record.
- There was one run per design on one machine. Single runs vary by
  several percent on some designs, so single-design differences of a
  percent or two mean little; the 17-design mean averages over that.
- The recipe's settings were chosen on ibm01 and ibm07, which are among
  the 17 designs scored.
- The score is the challenge's proxy. It says nothing about routed
  timing, power or area.

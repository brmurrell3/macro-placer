# Thinkorplace: Chip Macro Placement with Gradient Descent and Exact Local Search

Brendan Murrell, Xiaofei Zhang, Andrew Murrell · 15 November 2026 · Technical report 1.2

Macro placement decides where to place large components on a chip before its
wires are routed. Each of these blocks, called a macro, is a pre-designed
circuit. Macros that have wires connecting them should sit nearby so the wires
stay short. However, macros that are too close leave no room to route those
wires. Additionally, macros may not overlap.

Thinkorplace is the placer I designed and implemented for the Partcl/Hudson
River Trading Macro Placement Challenge 2026. It finished ninth on the final
leaderboard (of 132 verified eligible submissions) with a verified mean proxy of
0.9974. My score is a 31.6% improvement over the published RePlAce baseline.
Runtime was 12 minutes per test case (max 60 minutes per test case).

[Technical report (PDF)](thinkorplace-technical-report.pdf) ·
[Source code](https://github.com/brmurrell3/macro-placer) ·
[Placement explorer](placement-comparison.html) ·
[Cite](https://github.com/brmurrell3/macro-placer/blob/main/CITATION.cff)

![Mean proxy cost of all 129 ranked entries on the final leaderboard, plotted by final rank, where lower is better. Thinkorplace is highlighted at rank 9 with 0.9974. Dashed lines mark the RePlAce baseline at 1.4578 and the simulated-annealing baseline at 2.1251. Filled dots are entries verified by the organizers and hollow dots are not verified. Ranks 128 and 129 are off scale at 3.60183 and 7.215.](interactive/leaderboard-all.svg)

## How entries were scored

Entries were ranked on 17 IBM benchmark designs using the proxy cost defined by
Google’s Circuit Training combining wirelength, density, and congestion.
Wirelength estimates how much wire is needed to connect each net. Density
measures how crowded the layout is, averaging the most crowded 10% of grid
cells. Congestion compares routing demand with the available routing tracks and
averages the worst 5% of cells. Density and congestion each carry half the
weight of wirelength. Lower scores are better.

The reference evaluator takes between 1.4 and 27 seconds to score a placement.
That allows roughly 130 to 2,600 full evaluations per design in an hour, which
makes it impractical to test individual moves this way.

## How Thinkorplace works

Thinkorplace is a two stage process. First, it finds a quick approximation of a
good overall layout. Then it refines that layout through small moves evaluated
with the exact score.

The first stage uses gradient descent on a smooth approximation of the score,
allowing all the blocks to move together. A legalizer then resolves macro
overlaps by moving overlapping macros into nearby free space, starting with the
largest.

Because the approximation differs from the actual score, the second stage
optimizes the actual score directly. It uses coordinate descent, moving one
block along one axis at a time and accepting a move only if the exact score
improves. Each block stays within the free gap around it, so accepted moves
preserve a legal placement.

An incremental scorer makes this practical. Moving one block affects only a few
nets and grid cells, so the scorer updates those instead of recalculating the
entire placement. On sampled moves, this took 1.3 to 3.7 milliseconds, compared
with 1.4 to 27 seconds for the reference evaluator. The scores agreed within
floating-point rounding. That makes it possible to try about 100,000 moves in
ten minutes instead of a few hundred.

## What I learned

The simpler pipeline performed better in my May development runs. The earlier
version combined two starting methods, three types of local search, and a
curvature stage, taking 3,192 seconds per design. The two stage version took 524
seconds and scored 7.1% lower on average, improving 16 of the 17 designs. Lower
congestion accounted for 84.5% of the gain. My best explanation is that exact
local search tends to stay close to its starting layout, so a better starting
point matters more than adding more search. Though, I haven’t tested that
explanation by comparing the pipelines with the same compute budget.

Curvature kicks also helped, though I’m less sure what to make of that result.
When local search stalled, I nudged the placement along a direction where the
smooth model curved downward, then ran local search again. In research runs,
this lowered the score in 190 of 204 attempts, but the directions and step sizes
I tried worked about equally well. A random nudge followed by the same local
search might therefore work just as well, though I haven’t run that control yet.

I also found that an optimal reassignment can make the placement worse. The
Hungarian method finds the optimal assignment of blocks to slots for the costs
it is given. Here, those costs treat each move as if every other block stays
put. That assumption breaks when blocks sharing a net move together. In a
constructed example with two blocks, the costs predict a wirelength decrease of
18, but making both moves increases wirelength by 14.

After the competition, I reran the configuration I believe was used for the
ranked entry on my laptop’s CPU. It reached a mean proxy score of 0.98121, with
valid placements for all 17 designs. The official run used different hardware,
so this wasn’t a controlled comparison with the competition result.

All of these results are proxy scores, not measurements from a full
place-and-route flow. I also developed the method on the same 17 public designs
used for the ranking.

Thinkorplace is written in Python using PyTorch, NumPy, and SciPy, with support
for CUDA and Apple GPUs.

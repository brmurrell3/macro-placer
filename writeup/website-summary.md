# Thinkorplace: smooth exploration, exact scoring

Brendan Murrell · 8 September 2026 · Technical report 0.4

Macro placement chooses where a chip's largest blocks sit. Connections should
stay short without concentrating too much area or routing demand, and hard macros
must remain inside the canvas without overlap. Thinkorplace was built for the
2026 Macro Placement Challenge to search these continuous coordinates and
discrete geometric arrangements.

The difficulty is that each choice affects the others. Bringing two connected
blocks closer can shorten their wires while narrowing a channel needed by
other connections. Exchanging crowded blocks may require several to move
together. The report introduces these physical tradeoffs with a diagram before
formalizing the geometry and objective.

The system uses a differentiable model to move many macros together, restores
geometric feasibility, then improves that basin with an incremental implementation
of the competition proxy. Its final competition-era program runs three parallel
pipelines: canonical coordinate descent alone;
coordinate descent with a bounded curvature-guided perturbation; and coordinate
descent with a Hungarian macro-to-slot reassignment. Each pipeline legalizes and polishes its smooth starting basin. Coordinate moves and assignment batches use incremental component scores.
Curvature candidates and phase endpoints receive full evaluator checks, and the
supervisor returns the lowest-scoring completed lane with no hard overlaps.

Surviving historical records cover all 17 IBM designs. A v1 ancestor recorded a
1.05750 mean proxy, an earlier v2 lineage 1.00835, and a Gaussian-descent ancestor
0.98199, each with zero recorded hard overlaps. The leaderboard separately
verified a May 20 submission at 0.9974. Different code, hardware, budgets, and
concurrency prevent attributing these differences to one mechanism.

The post-competition package adds an enforced deadline, atomic phase checkpoints,
explicit resource settings, full placement validation, and process cleanup. A
hierarchy-aware topology repair restored all 5,498 hard-pin occurrences used by
the ariane133 net traversal; IBM01 was
unchanged. A second fix restores 859 hard-macro edges in the SDF initializer.
These checks verify connectivity, not an NG45 quality gain.

In small CPU samples on IBM01 and IBM10, incremental component updates took
1.3–3.7 milliseconds per move, while full recomputation took 1.4–26.9 seconds.
The component values agreed to floating-point tolerance. These measurements
exclude setup and overlap validation and do not measure whole-solver speedup. A frozen IBM01 endpoint
diagnostic found an automatic-differentiation Rayleigh quotient of -0.442458,
alongside a nonzero raw gradient and poor finite-difference agreement. It supplies
a candidate direction, not evidence of a stationary saddle.

The report derives the feasible coordinate slices, a score-error bound for
reliable move acceptance, and the conditions for curvature descent. A concrete
HPWL counterexample shows why an optimal assignment proposal can still worsen
the joint placement.

A separate 12-run engineering batch returned nine valid results; all three
IBM17 runs hit the external time limit. After both connectivity repairs, all
three entry points returned valid ariane133 placements in a short integration
check. A controlled full-suite comparison of
corrected v1, Lane A, and A+B+C remains open. Read the [technical report](../output/pdf/thinkorplace-technical-report.pdf)
or explore the [interactive placement comparison](interactive/placement-comparison.html).

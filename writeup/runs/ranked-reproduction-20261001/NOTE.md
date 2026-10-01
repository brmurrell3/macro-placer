# Ranked-configuration reproduction, attempt 2 (stopped)

Config: `writeup/study/ranked_reproduction.json`; source commit 64b50a7, runtime
archive `corrected-runtime-cbfbff0c81c3`, which includes the checkpoint bounds
fix that attempt 1 exposed. Run 2026-10-01 on the author's Apple M3 Max (macOS
27.0.1, CPU, three threads, one design at a time). Every completed record is
kept.

| Design | Proxy | Solver outcome | Wall (s) |
|---|---:|---|---:|
| IBM01 | 0.82996 | pipeline completed | 329 |
| IBM02 | 1.00227 | pipeline completed | 409 |
| IBM03 | 0.91446 | pipeline completed | 378 |
| IBM04 | 0.91910 | pipeline completed | 347 |
| IBM06 | 1.03906 | pipeline completed | 350 |
| IBM07 | 1.00667 | pipeline completed | 374 |
| IBM08 | 1.01842 | pipeline completed | 414 |
| IBM09 | 0.77703 | pipeline completed | 357 |
| IBM10 | 1.14944 | checkpoint recovery (pre-descent state) | 749 |
| IBM11 | 0.78998 | pipeline completed | 390 |
| IBM12 | 1.08597 | pipeline completed | 712 |
| IBM13 | 0.84098 | pipeline completed | 399 |
| IBM14 | 1.06636 | pipeline completed | 712 |
| IBM15 | 1.05107 | pipeline completed | 560 |
| IBM16 | 1.14491 | checkpoint recovery (pre-descent state) | 925 |
| IBM17 | — | stopped by the operator | — |
| IBM18 | — | stopped by the operator | — |

The checkpoint fix worked: IBM03, which attempt 1 lost, completed. IBM10 and
IBM16 exposed a second defect. On this CPU, smooth descent took about 118 s on
IBM10 and 154 s on IBM16, and coordinate descent was then budgeted all
remaining time to the 720 s deadline (576 s and 526 s), leaving none for its final checkpoint and score (two canonical
evaluations, about 27 s each on IBM10). The supervisor stopped the lane at the
deadline, so the run returned the last valid checkpoint, from before coordinate
descent. The operator stopped the run during IBM17 to fix the budget reserve.
These records are not a suite result.

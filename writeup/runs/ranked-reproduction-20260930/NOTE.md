# Ranked-configuration reproduction, attempt 1 (stopped)

Config: `writeup/study/ranked_reproduction.json` (commit cb554f1). Started
2026-09-30 on the author's Apple M3 Max (macOS 27.0.1, CPU, three threads, one
design at a time). Stopped by the operator during IBM06, after IBM03 exposed a
runtime defect; IBM06 has no result. Every completed record is kept.

| Design | Proxy | Solver outcome |
|---|---:|---|
| IBM01 | 0.82996 | pipeline completed |
| IBM02 | 1.00200 | pipeline completed |
| IBM03 | 1.10002 | checkpoint recovery (pre-descent state) |
| IBM04 | 0.91909 | pipeline completed |

On IBM03 coordinate descent completed, but the final checkpoint failed strict
validation with "Macros outside horizontal canvas bounds", so the lane returned
its earlier `descent_legalized` checkpoint. The corrected runtime's
`checkpoint()` validates strictly without the one-step inward projection that
`finalize_placement()` applies, so any bounds excess at that point discards the
coordinate-descent result. The rejected IBM03 candidate was not saved, so the
size of its excess is unknown; the likely cause is a float32 rounding overshoot
at the upper canvas edge, as found in 12 of 34 historical cascade endpoints.
These four designs do not form a suite result.

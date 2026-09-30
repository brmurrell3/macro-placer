# Thinkorplace v2

`placer.py` is the compatibility entry point for `thinkorplace.v2.placer.Placer`.
The runtime is under `thinkorplace/v2/` with shared modules in
`thinkorplace/common/`.

The three lanes start with Gaussian-density analytical descent and canonical
coordinate descent. Lane B adds curvature-guided perturbation and polish. Lane C
adds Hungarian slot proposals with sequential legality-aware commits and polish.
The supervisor selects the lowest canonical cost among valid checkpoints from
that execution. Resource contention and deadlines can change which stages finish.

Use the [publication runner](../../writeup/study/README.md) to select Lane A or
A+B+C, set an explicit device/thread allocation, enforce an external wall cap,
and preserve manifests, logs, valid checkpoints, and full legality checks.

The historical 0.98387 figure belongs to a standalone extended-CD run; 0.97745
is an offline per-design oracle. Neither establishes the final operating ensemble
mean. See the [source/results audit](../../docs/reviews/2026-09-06-thinkorplace.md)
and [report](../../writeup/latex/main.tex) for the supported evidence.

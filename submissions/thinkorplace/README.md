# Thinkorplace v1

`placer.py` is the compatibility entry point for
`thinkorplace.v1.placer.CDLNSSACascadeStackedPeripheryPlacer`.

The runtime combines an SDF/CD/LNS/SA lane, a DPO/CD/LNS/SA/joint-search lane,
canonical selection, a curvature cascade, a weighted proposal portfolio, and a
periphery refinement. Budget-dependent stages can be skipped. Historical code is
preserved in Git and in the publication source snapshot.

Use the [publication runner](../../writeup/study/README.md) for an external wall
cap and full result validation. Several older inner loops have advisory phase
budgets; the external supervisor remains the execution boundary. Runtime imports
come from the `thinkorplace` package and canonical `macro_place` infrastructure.

The [technical report](../../writeup/latex/main.tex) distinguishes historical
records from corrected runs. The archived 1.05749545 mean covers 17 IBM designs
and is not a fresh benchmark of the repaired package.

# Submission dependency bundles

This directory is for binary/install dirs that get mounted into the partcl
`eval_docker` container. Bundle them via `run_eval.sh` extras.

## dreamplace_install/

A DREAMPlace install compiled against `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`
(matching the eval_docker base image). Provides `dreamplace/Placer.py` and the
compiled `_*.so` extensions.

Built via `eval_docker_build/dreamplace.dockerfile` (see that dir).

Updated: when DREAMPlace build completes.

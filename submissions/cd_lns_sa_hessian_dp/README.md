# cd_lns_sa_hessian_dp — champion + optional DREAMPlace lane

Drop-in extension of the E74 champion. Adds DREAMPlace as a 3rd init
lane between E41 and the Hessian saddle escape, when DREAMPlace is
installed on the host. Falls back gracefully to the E74 pipeline if
DREAMPlace is missing.

This is the submission to use when running on the partcl evaluation
hardware (AMD EPYC 9655P + RTX 6000 Ada + Linux), which can host
DREAMPlace cleanly. On macOS M3 Max it behaves identically to the
E74 champion at `submissions/cd_lns_sa_hessian/placer.py`.

## Pipeline

```
benchmark
  → Phase 1: E25 (SDF init + CD + LNS + SA-v2)              ← CPU
  → Phase 2: E41 (DPO init + CD + LNS + SA-v2 + K-joint)    ← CPU
  → Phase 3: DREAMPlace (Nesterov on smooth proxy, 1024×1024 bins)  ← GPU [optional]
  → Phase 4: plateau pick = min(E25, E41, DREAMPlace)
  → Phase 5: Hessian saddle escape on plateau (Lanczos + ε perturbation + CD polish)
  → final: best-of {E25, E41, DREAMPlace, saddle}
```

Phase 3 is skipped silently if `DREAMPLACE_ROOT` env var doesn't point
to a working install. No code paths break.

## Cloud setup (one-time)

```bash
# 1. Provision: AMD EPYC or similar Linux box + CUDA-capable GPU
#    (Lambda A10/A100, RunPod RTX 4090, Modal H100 — all fine; ~$0.30–2/hr)

# 2. Clone repo + submodules
git clone https://github.com/<user>/macro-place-challenge-2026.git
cd macro-place-challenge-2026
git submodule update --init external/MacroPlacement

# 3. uv setup
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 4. Install DREAMPlace (Docker is simplest)
docker pull limbo018/dreamplace-cuda
# Or build natively:
# git clone --recursive https://github.com/limbo018/DREAMPlace.git /opt/DREAMPlace
# cd /opt/DREAMPlace && mkdir build && cd build && cmake .. && make -j && make install

# 5. Point env at DREAMPlace
export DREAMPLACE_ROOT=/opt/DREAMPlace

# 6. Verify import + run a smoke
uv run evaluate submissions/cd_lns_sa_hessian_dp/placer.py -b ibm01 --json
```

## Smoke checks before going wide

```bash
# Confirm DREAMPlace lane fires (look for "[DP] DREAMPlace done in ..." in log)
uv run evaluate submissions/cd_lns_sa_hessian_dp/placer.py -b ibm01 --json --hypothesis dp_smoke

# Compare proxy to E74 cached ibm01 = 0.85527 / cascading ibm01 = 0.84528
# Expectation: if DREAMPlace lands in a structurally different basin and
# its saddle escape goes deeper, we beat both. If not, hybrid pick keeps
# the best of E25/E41/saddle, so we never regress from E74's number.
```

## Going wide

```bash
# All 17 IBM, --jobs 4 in parallel (fits within 17-hr aggregate envelope)
uv run evaluate submissions/cd_lns_sa_hessian_dp/placer.py --all --jobs 4 --json

# NG45 verification (required for Tier 2 Grand Prize)
uv run evaluate submissions/cd_lns_sa_hessian_dp/placer.py --ng45 --json
```

## Wall budget

E74 champion takes ~50 min/bench on M3 Max worst case (ibm01: 96 min).
On AMD EPYC the per-core clock is slower but CD polish is the bottleneck;
expect 1.2-1.5× wall vs M3 Max → ~70 min ibm01.

Phase 3 (DREAMPlace) adds ~30s-5min depending on bench size — fast.

**This is over the 60 min/bench hard cap.** Mitigation: see
`TODO.md` §"Wall-safe E74 with mid-loop time enforcement" — that work
needs to land too. The DP lane doesn't change the wall problem; it only
adds a (small) phase. Run this in tandem with the time-enforcement
work.

## What we expect

If DREAMPlace's analytical placement lands in a structurally different
basin than SDF (E25) or DPO (E41), the Hessian saddle escape from that
basin should find a different (potentially deeper) minimum. Even if not,
the hybrid pick can't regress from E74. So this is a strictly added
lever, low downside.

Expected lift: 0.3-1.5 % aggregate if DREAMPlace's basin is useful;
0 if not.

# Morning handoff — 2026-05-16

## TL;DR

- **AWS CPU box up**: `aws-cpu` (13.220.11.97, c6a.4xlarge spot, $0.29/hr). Running 17-IBM `placer_adaptive` chain since 05:31 UTC, 4-bench parallel. Expected done ~10:00 UTC = 06:00 EDT.
- **AWS GPU box**: blocked on quota approval (filed 01:07 EDT, status PENDING). Polling daemon (`/tmp/aws_quota_poll.log`) will auto-launch g5.xlarge + bootstrap when approved.
- **Cached aggregate = 1.05156**: per-bench best across all 121 cached .pt files. Theoretical ceiling — needs rule-compliant ensemble to be usable.
- **Xplace integration ready**: bookshelf converters done for all 21 benches; runner + setup scripts in `experiments/Xplace_integration/code/`. Fires when GPU box bootstraps.

## What's running

| What | Where | Process | Expected end |
|---|---|---|---|
| `placer_adaptive --all` 4-parallel | aws-cpu | `bash ~/run_parallel.sh ...` (PID ~4154) | ~06:00 EDT |
| AWS GPU quota poll | local | PID 51349 | when quota approves OR manual kill |
| GPU bootstrap (chained) | local→aws-gpu | auto-fires from poll daemon | ~30 min after quota approval |

## Key files

| Path | What it does |
|---|---|
| `experiments/Xplace_integration/code/setup_gpu_box.sh` | Install CUDA, Xplace, deps on fresh GPU box |
| `experiments/Xplace_integration/code/launch_gpu_box.sh` | Provision g5.xlarge spot instance |
| `experiments/Xplace_integration/code/poll_quota.sh` | Background daemon: poll quota → auto-bootstrap |
| `experiments/Xplace_integration/code/xplace_runner.py` | Python placer wrapper (subprocess Xplace, parse .pl) |
| `experiments/E76_dreamplace_integration/bookshelf_out/*` | 21 benches in bookshelf format |

## Check status when you wake

```bash
# 1. AWS CPU overnight chain
ssh aws-cpu "cat ~/overnight_run/OvernightAWS_adaptive/master.log; ls ~/overnight_run/OvernightAWS_adaptive/*.log | wc -l"

# 2. Per-bench results
ssh aws-cpu "grep -E 'avg|aggregate|proxy=.*VALID' ~/overnight_run/OvernightAWS_adaptive/*.log"

# 3. GPU quota
aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --query 'Quota.Value' --output text
cat /tmp/aws_quota_poll.log  # poll daemon log

# 4. If GPU box launched
ssh aws-gpu nvidia-smi  # confirm GPU
ssh aws-gpu "tail -50 ~/setup.log"  # bootstrap progress
```

## If GPU box came up overnight

```bash
ssh aws-gpu
# wait for setup_gpu_box.sh to finish (~10-20 min for Xplace build)
cd ~/macro-place-challenge-2026
export XPLACE_ROOT=$HOME/Xplace
uv run python experiments/Xplace_integration/code/xplace_runner.py ibm01 3300
# expect: proxy=<X.XXX> ovl=<Y>; Xplace output at ~/Xplace/result/.../<bench>.pl
```

## Strategic context

- Submission floor: 1.07820 (`submissions/cd_lns_sa_cascade/placer_adaptive.py`, PATH A post-A1).
- Cached best per-bench = 1.05156 (theoretical ceiling for ensemble).
- vmallela #1 at 1.011 (same Hessian saddle algorithm class).
- Carrotato 0.967 via Xplace+Triton, 3.8 min/bench — this is what Xplace integration targets.
- Gap to vmallela = 1.078 − 1.011 = 0.067. To beat #1 we need ~6.6% lift.

## What's NOT done

- Innovation writeup (1-2 days of writing, no compute)
- ORFS Tier 2 feasibility check on judges' hardware
- Macro orientation sidecar (Klein-4 N/FN/FS/S)

## Costs incurred overnight

| Item | Rate | Hours | Total |
|---|---:|---:|---:|
| aws-cpu c6a.4xlarge spot | $0.29/hr | ~8 | $2.32 |
| aws-gpu g5.xlarge spot (if launched) | ~$0.50/hr | ~4 | $2.00 |
| **Worst case** | | | **~$4.50** |

## Caveat

The AWS CPU chain is **cross-validation** of our existing 1.078 floor on EPYC hardware — it doesn't unlock new score. The breakthrough is gated on GPU access for Xplace; if quota doesn't approve by morning, the real Xplace work is blocked until either quota approves or lambda/OCI comes back online.

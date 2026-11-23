# GPU kernels for the placement descent

v3's first stage is gradient descent on a smooth approximation of the proxy score:
96 coarse steps, then 404 fine steps of Adam, over a batch of independent
placements (lanes). In stock PyTorch each step is about 1,000 small operations
plus an autograd graph. This folder holds that step as custom GPU kernels with
every gradient derived by hand: in Metal for Apple GPUs, ported to CUDA for
NVIDIA GPUs, and with two kernels also written in Triton.

AI coding agents wrote and verified the kernels under the author's direction,
against correctness and speed rules fixed before each measurement. The kernels
are not part of the ranked competition entry, and the v3 package in
[`../thinkorplace/v3/`](../thinkorplace/v3/) still runs its descent on the CPU.

## Layout

| Path | Contents |
|---|---|
| [`metal/fine.py`](metal/fine.py) | The fine step: Metal shader source and its driver, one class (`MetalFineStep`). |
| [`metal/coarse.py`](metal/coarse.py) | The coarse step, the same way (`MetalCoarseStep`). |
| [`cuda/fine/`](cuda/fine/) | The fine step in CUDA: `fine.cu`, `pairs.cu` (the congestion term's pair sums) and the driver `step.py` (`CudaFine`). |
| [`cuda/coarse/`](cuda/coarse/) | The coarse step in CUDA: `coarse.cu`, the static lists in `lists.py`, the driver `step.py` (`CudaCoarse`). |
| [`cuda/k9.h`](cuda/k9.h), [`cuda/k9cu.py`](cuda/k9cu.py) | The launch path: a header that gives CUDA the Metal semantics of `min`, `max`, `clamp` and friends, and a CuPy launcher that compiles the `.cu` files at construction and checks every launch's arguments against the kernel's parameter list. |
| [`cuda/triton/`](cuda/triton/) | Two fine-step kernels in Triton (`kernels.py`) and `fine_step.py`, which swaps one CUDA launch for its Triton twin. |
| [`figures/`](figures/) | The figures below. |

## What a step computes

For each lane, with `c` the positions clamped to the canvas,

    loss = wirelength(c) + 0.5 density(c) + 0.5 congestion(c) + λ overlap(c) / area + 50 boundary(pos) / area

Wirelength is a log-sum-exp smoothing of each net's half-perimeter. Density spreads each
block over the grid with Gaussian factors and averages the fullest cells.
Congestion routes every source-sink pair along an L-shape with sigmoid-softened
ends, adds the routing
capacity that hard blocks take, and averages the most loaded 5% of cells plus
half the mean load. Overlap penalises pairs of hard blocks.

The kernels compute the loss and its gradient in one forward and one backward
pass written by hand, then apply Adam in place. The fine step is 20 kernel
launches and 5 PyTorch operations (three batched matrix products, one top-k,
one sum). The top-k selections that choose the hottest cells stay in PyTorch;
everything around them is fused.

## Correctness

Each step is checked against the same formulas in float64 on the CPU, on three
designs at 1 to 16 lanes (nine cases). The limit is 1e-5 of the largest gradient
entry.

- **Metal, M3 Max.** The fine step's gradient is within 3.4e-7 to 8.1e-7 of the
  reference, closer than the stock float32 PyTorch step. The coarse step is
  within 9.1e-7 to 2.8e-6. Results repeat bit for bit between processes at the
  same batch size; the stock code's do not.
- **CUDA, NVIDIA L4.** Fine gate 9 of 9 and coarse gate 9 of 9. One rule was not
  met as written: compared with the live stock PyTorch coarse step, one entry on
  ibm01 at 16 lanes was out of tolerance, so that comparison reads 8 of 9.
- **Triton.** Each Triton kernel passes 9 of 9 against the same reference.
- **Placement quality.** From v3's own seeds, every lane legalises after a
  kernel descent. Against v3's CPU descent, the Metal kernels' legalised proxy
  differs by +0.09%, +0.11% and -0.39% on the three designs, none above two
  standard errors; the CUDA descent shows no difference on any of the three.
  Against the Metal descent, the CUDA descent scored 0.42% higher (worse) on
  ibm17, at three standard errors over six seeds, and showed no difference on
  ibm01 and ibm07. The cause is not isolated.

## Speed

Baseline: the same step in stock PyTorch on the same GPU. Designs ibm01 (1,140
blocks) and ibm07 (1,331) at 16 lanes, ibm17 (2,604) at 6 lanes. float32.

**Metal on an Apple M3 Max** (torch 2.8.0, MPS):

| | ibm01 | ibm07 | ibm17 |
|---|---:|---:|---:|
| Fine step | 29 times | 43 times | 61 times |
| Coarse step | 13 times | 19 times | 24 times |
| Whole 500-step descent, stock | 21.3 s | 36.0 s | 55.7 s |
| Whole 500-step descent, kernels | 1.17 s | 1.33 s | 1.37 s |

![Seconds for one whole descent on an Apple M3 Max for each stage of the work, on a logarithmic scale, for ibm01 and ibm07 at 16 lanes and ibm17 at 6 lanes. Stock PyTorch takes 21.3, 36.0 and 55.7 s. Rewriting the fine step in plain PyTorch operations gives 2.6 to 4.2 times, a custom kernel for the congestion pair sums alone 2.5 to 4.3 times, fused fine-step kernels 5.1 to 7.9 times, and adding the coarse-step kernels 18, 27 and 41 times, at 1.17, 1.33 and 1.37 s.](figures/descent-ladder.png)

The second row is a control. Rewriting the fine step with PyTorch operations
alone, no custom kernel, reaches 2.6 to 4.2 times, so most of the gain comes
from the fused kernels, not from a better arrangement of stock operations.

**CUDA on one NVIDIA L4** (AWS g6.xlarge, torch 2.15.0, CuPy 14.2.0, Triton 3.9.0):

| | ibm01 | ibm07 | ibm17 |
|---|---:|---:|---:|
| Fine step | 21 times | 42 times | 51 times |
| Coarse step | 17 times | 31 times | 45 times |
| Whole 500-step descent, stock | 14.5 s | 32.1 s | 58.8 s |
| Whole 500-step descent, kernels | 0.77 s | 0.91 s | 1.26 s |
| Speed-up of the whole descent | 19 times | 35 times | 47 times |

A second agent reran the CUDA timing chain on the same card; none of 139
compared quantities moved by more than 5% (largest 2.7%). Kernel compilation
is outside the clock; with a cold compile cache the whole-descent speed-up is
6 to 17 times. The Triton twin of the density-factor kernel runs the whole step
at 0.99 to 1.00 of the CUDA step's speed.

## Where the time goes

Nsight Systems traces on the L4: the kernel fine step makes 26 launches and the
coarse step 16, against 532 to 754 for the stock steps, and keeps the GPU busy
for 90 to 96% of the step. Launch overhead is 19 to 34% of the synchronised
fine step by a model of launches times the measured 22.4 µs per launch.

![Two panels for the NVIDIA L4 at its 72 W power cap. Left, a roofline with a measured 237 GB/s bandwidth roof and 19.1 TFLOP/s compute roof, and each kernel launch of the port plotted by arithmetic intensity against achieved throughput, with the stock fine and coarse steps far below the roof at the left. Right, each launch's share of the attainable roof at its intensity, mostly 10 to 45%, with the density-factor kernel near 66 to 69%.](figures/roofline.png)

The roofs were measured on the card. Byte and flop counts are modelled from the
source, since hardware counters need administrator rights the instance did not
give. Private GPU-event timings on a busy GPU put the density-factor kernel
above 100% of its modelled roof, which shows the model's limits.

## Limits

- **Not runnable on its own.** The gates, benches, frozen inputs (about
  130 MB) and the stock PyTorch stages they compare against stay in the
  research tree. This folder is the kernel code and the drivers that launch it.
- **One GPU each.** CUDA and Triton ran on one NVIDIA L4. The `.cu` files are
  written to compile as HIP, but were never compiled as HIP or run on an AMD GPU.
- **Input checks.** A second agent found failures on input the descent does
  not produce: on a non-default CUDA stream the fine step crashes and the coarse
  step returns wrong gradients, and the coarse step does not check the macro
  count it is given. They are not fixed.
- **Edits after the runs.** The CUDA and Triton files differ from the ones
  that ran only in comments, labels and the two lines of
  `cuda/triton/fine_step.py` that load the CUDA fine step; they were not rerun.
  The Metal files were restructured from a chain of research versions into one
  class each, with unused kernels removed. On the Mac, the restructured classes
  give bit-identical loss, gradient and Adam state to the originals on all nine
  cases, pass both float64 gates 9 of 9, and reproduce a whole recorded ibm01
  descent bit for bit. One unused output differs: the CUDA `setk5` still writes
  a list of selected cells that nothing reads, which the Metal version no
  longer writes.

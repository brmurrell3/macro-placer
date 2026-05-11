# E76 — DREAMPlace integration via cloud GPU

End-to-end workflow to run DREAMPlace on all our benchmarks via a
rented Linux+CUDA box and integrate its placements into our champion
pipeline as an additional init lane.

## Local prep (already done)

1. **`code/tilos_to_bookshelf.py`** — converts a TILOS-protobuf
   benchmark to the 6-file Bookshelf format DREAMPlace consumes
   (`.aux`, `.nodes`, `.pl`, `.nets`, `.scl`, `.wts`). Validated on
   ibm01.

2. **`code/bookshelf_to_pt.py`** — converts DREAMPlace's `.gp.pl`
   output back into our `torch.Tensor[num_macros, 2]` placement
   format, with optional `project_overlaps` legalization. Output is
   a `.pt` file with the same shape as our cached E25/E41
   placements, ready to drop into E85.

3. **`cloud/run_dreamplace.sh`** — cloud-side runner; assumes
   DREAMPlace is installed and uses a small JSON config that points
   at the Bookshelf `.aux` file. Default settings: target_density
   0.85, density_weight 8e-5, Nesterov optimizer, 1000 iterations,
   1024×1024 bins, legalize on, GPU on.

## Cloud workflow

### 1. Generate Bookshelf inputs locally for all benches

```bash
for B in ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 \
         ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18; do
  uv run python experiments/E76_dreamplace_integration/code/tilos_to_bookshelf.py \
      $B experiments/E76_dreamplace_integration/bookshelf_out/$B
done
```

Same for NG45:
```bash
for B in ariane133 ariane136 mempool_tile nvdla; do
  uv run python experiments/E76_dreamplace_integration/code/tilos_to_bookshelf.py \
      $B experiments/E76_dreamplace_integration/bookshelf_out/$B
done
```

### 2. Provision cloud box

Lambda Labs / Modal / RunPod / Vast.ai with CUDA-capable GPU. For
DREAMPlace, an A10 / A100 / RTX 3090 / RTX 4090 are all fine. We
need 20-30 GB RAM and CUDA 11+ runtime. Estimated cost: ~$1-10
total (the actual DREAMPlace runs are seconds-to-minutes per bench;
most time is install).

### 3. Cloud install

The fastest path: pull the official Docker image.

```bash
docker pull limbo018/dreamplace-cuda
docker run --gpus all -it -v $(pwd):/workspace limbo018/dreamplace-cuda bash
# inside the container: /opt/DREAMPlace is the source tree
```

Alternative (native install on Ubuntu CUDA box):
```bash
git clone --recursive https://github.com/limbo018/DREAMPlace.git
cd DREAMPlace
# follow their INSTALL.md
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/DREAMPlace
make -j && make install
```

### 4. Upload Bookshelf inputs + run script to cloud

```bash
# from local
rsync -av experiments/E76_dreamplace_integration/bookshelf_out/ \
          experiments/E76_dreamplace_integration/cloud/run_dreamplace.sh \
          cloud-box:dreamplace_runs/
```

### 5. Run DREAMPlace per bench on cloud

```bash
# on cloud
cd dreamplace_runs
chmod +x run_dreamplace.sh
for B in */; do
  ./run_dreamplace.sh "$B" /opt/DREAMPlace
done
```

Wall: ~30 s – 5 min per bench depending on size and GPU. 17 + 4 = 21
benches total.

### 6. rsync outputs back to local

```bash
# from local
rsync -av --include='*.gp.pl' --include='*/' --exclude='*' \
      cloud-box:dreamplace_runs/ \
      experiments/E76_dreamplace_integration/bookshelf_out/
```

### 7. Convert outputs to our .pt format

```bash
for B in ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 \
         ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18; do
  uv run python experiments/E76_dreamplace_integration/code/bookshelf_to_pt.py \
      $B \
      experiments/E76_dreamplace_integration/bookshelf_out/$B/$B.gp.pl \
      experiments/E76_dreamplace_integration/results/dreamplace_$B.pt
done
```

### 8. Plug into E85 multi-init or E74

Once `experiments/E76_dreamplace_integration/results/dreamplace_*.pt`
exists for each bench, E85 will pick it up automatically (we'll add
a `DREAMPlace_cached` init label). Smoke test:

```bash
uv run python experiments/E85_multi_init_ensemble/code/multi_init_ensemble.py \
    ibm01 --inits E25_cached,E41_cached,DREAMPlace_cached
```

If lift, run on all 17.

## Risk

- **Install fragility.** DREAMPlace's CUDA build can fail on
  CUDA-version mismatches. Docker image mitigates.
- **Bookshelf-format incompatibility.** Our converter writes
  `terminal` flag for fixed macros; some DREAMPlace branches expect
  `MOVETYPE` field instead. If we see "no movable nodes" errors,
  patch the converter.
- **DREAMPlace optimizes for HPWL only by default.** Our proxy is
  WL + 0.5·density + 0.5·congestion. The DREAMPlace placement may
  be WL-optimal but density-poor — that's fine because we polish
  via our pipeline downstream (CD + LNS + SA + Hessian saddle
  escape).

## Expected outcome

If DREAMPlace lands in a structurally different basin than
SDF (E25) or DPO (E41), the Hessian saddle escape from its plateau
should find a different (potentially deeper) minimum. Even if not,
the per-bench best-of {E25, E41, DREAMPlace} can't regress from
current E74/E84 — it's a strictly added lane.

Lift estimate: 0.3 – 1.5 % aggregate if DREAMPlace lands a useful
basin; 0 if not.

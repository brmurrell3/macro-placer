# Build helpers for our submission deps

## dreamplace.dockerfile

Builds DREAMPlace against `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`
with `-D_GLIBCXX_USE_CXX11_ABI=0` (matches PyTorch 2.5.1 official
binary ABI). Outputs install to `/opt/dreamplace_install/` in the image.

To rebuild and refresh `submit_deps/dreamplace_install/`:

```bash
# Build on a Linux+Docker host (this Mac uses colima):
sudo docker build -t dreamplace:2.5.1-cuda12.4 -f eval_docker_build/dreamplace.dockerfile .

# Extract install:
sudo docker create --name dp_extract dreamplace:2.5.1-cuda12.4
sudo docker cp dp_extract:/opt/dreamplace_install /tmp/dreamplace_install
sudo docker rm dp_extract

# Slim (remove non-runtime dirs):
rm -rf /tmp/dreamplace_install/{unittest,test,benchmarks,include,bin}

# Rsync to repo:
rsync -avz --delete /tmp/dreamplace_install/ submit_deps/dreamplace_install/
```

## Why we rebuilt

The first build had inconsistent C++ ABI between sub-projects (Limbo's
libverilogparser.a was NEW ABI, PyTorch + place_io_cpp.so were OLD ABI).
Loading `dreamplace.ops.place_io.place_io_cpp` failed with
`undefined symbol: _ZN13VerilogParser15VerilogDataBase22verilog_assignment_cbkERKSsRKNS_5RangeES2_S5_`.

Fix: pass `-D_GLIBCXX_USE_CXX11_ABI=0` via `CMAKE_CXX_FLAGS` AND
`CMAKE_C_FLAGS` so the flag propagates to every subproject's
compile commands.

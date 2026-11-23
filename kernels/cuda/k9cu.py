"""CUDA launch path of the kernels (float32).

Kernels live in single-source files (k9.h + *.cu, written so that they also compile as HIP). Here they are compiled at
construction by CuPy through NVIDIA's run-time compiler and called with the call shape of torch.mps.compile_shader, so
that the Metal drivers port line by line:

    lib = k9cu.Library([HERE / "fine.cu"], block=256, blocks={"wl_net": 64})
    lib.wl_net(PGW, VAL, c, net_start, pm, po, net_w, ip, [gamma], threads=b * NN)
    lib.wl_net(..., threads=b * NN, group_size=128)      # block size for this call only

Argument rules. CuPy checks nothing ("No validation is performed by CuPy for arguments passed to the kernel"), so the
count and the kinds are checked here against the kernel's own parameter list, read from the source:
  torch.Tensor (CUDA, contiguous)        -> its device pointer; dtype float32 for float*, int32 for int*, etc.
  list or tuple of Python numbers        -> that many `float` parameters by value (Metal: `constant float* fp`)
  Python float -> `float`; Python int or bool -> `int`; a NumPy scalar is passed as it is
The trailing `int k9_n` of every kernel (k9.h) is appended by the call: it is `threads`.

Launches go to CuPy's current stream. torch's default stream and CuPy's default stream are both the device's default
stream (checked by `stream_report()`), so launches and torch operations keep their order without any binding.
`torch.cuda.synchronize()` waits for all streams of the device and is the synchronisation used for timing.
"""
from __future__ import annotations

import re
from pathlib import Path

import cupy
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
HEADER = HERE / "k9.h"
_SIG = re.compile(r"K9_KERNEL\s+(\w+)\s*\(([^)]*)\)", re.S)
_DTYPES = {"float": torch.float32, "int": torch.int32, "unsigned": torch.int32, "uint": torch.int32,
           "double": torch.float64, "long long": torch.int64, "char": torch.int8, "unsigned char": torch.uint8}


def _parse(source):
    """{kernel name: [("ptr", torch dtype) | ("float",) | ("int",), ...]} without the trailing k9_n."""
    out = {}
    source = re.sub(r"/\*.*?\*/", " ", source, flags=re.S)       # a signature inside a comment is not a kernel
    source = re.sub(r"//[^\n]*", " ", source)
    for name, params in _SIG.findall(source):
        kinds = []
        for p in [q.strip() for q in params.replace("\n", " ").split(",") if q.strip()]:
            words = p.replace("*", " * ").split()
            base = " ".join(w for w in words[:-1] if w not in ("const", "*", "__restrict__", "restrict"))
            if "*" in words:
                kinds.append(("ptr", _DTYPES.get(base)))
            elif base == "float":
                kinds.append(("float",))
            elif base in ("int", "unsigned", "uint"):
                kinds.append(("int",))
            else:
                raise ValueError(f"kernel {name}: parameter '{p}' has a type the launcher does not know")
        if not kinds or kinds[-1] != ("int",) or not params.replace("\n", " ").rstrip().endswith("k9_n"):
            raise ValueError(f"kernel {name}: the last parameter must be `int k9_n`")
        out[name] = kinds[:-1]
    return out


class Library:
    def __init__(self, sources, block=256, blocks=None, options=(), check=True):
        self.paths = [Path(s) for s in sources]
        self.source = HEADER.read_text() + "\n" + "\n".join(p.read_text() for p in self.paths)
        self.options = tuple(options)
        self.signatures = _parse(self.source)
        self.module = cupy.RawModule(code=self.source, options=self.options)
        self.block, self.blocks, self.check = int(block), dict(blocks or {}), bool(check)
        self._calls = {}
        self.launches = 0                                   # counted, so that a profile can be checked against it
        for name in self.signatures:                        # compile now, not at the first step
            self.module.get_function(name)

    def __getattr__(self, name):
        calls = self.__dict__.get("_calls")
        if calls is None or name.startswith("_"):
            raise AttributeError(name)
        call = calls.get(name)
        if call is None:
            if name not in self.signatures:
                raise AttributeError(f"no kernel named {name} in {[p.name for p in self.paths]}")
            call = calls[name] = self._make(name)
        return call

    def _make(self, name):
        fn, kinds, default = self.module.get_function(name), self.signatures[name], self.blocks.get(name, self.block)
        uintp, f32, i32, check = np.uintp, np.float32, np.int32, self.check

        def call(*args, threads, group_size=None):
            packed = []
            for a in args:
                if isinstance(a, torch.Tensor):
                    packed.append(a)
                elif isinstance(a, (list, tuple)):
                    packed.extend(f32(x) for x in a)
                elif isinstance(a, float):
                    packed.append(f32(a))
                elif isinstance(a, (bool, int)):
                    packed.append(i32(a))
                else:
                    packed.append(a)
            if len(packed) != len(kinds):
                raise TypeError(f"{name}: {len(packed)} arguments after expansion, the kernel takes {len(kinds)} before k9_n")
            if check:
                for j, (a, kind) in enumerate(zip(packed, kinds)):
                    if kind[0] == "ptr":
                        if not isinstance(a, torch.Tensor) or not a.is_cuda or not a.is_contiguous():
                            raise TypeError(f"{name}: argument {j} must be a contiguous CUDA tensor")
                        if kind[1] is not None and a.dtype != kind[1]:
                            raise TypeError(f"{name}: argument {j} is {a.dtype}, the kernel reads {kind[1]}")
                    elif kind[0] == "float" and not isinstance(a, np.float32):
                        raise TypeError(f"{name}: argument {j} must be a float")
                    elif kind[0] == "int" and not isinstance(a, np.int32):
                        raise TypeError(f"{name}: argument {j} must be an int")
            n = int(threads)
            if n <= 0:
                return
            block = default if group_size is None else int(group_size)
            packed = [uintp(a.data_ptr()) if isinstance(a, torch.Tensor) else a for a in packed]
            packed.append(i32(n))
            fn(((n + block - 1) // block,), (block,), tuple(packed))
            self.launches += 1
        call.__name__ = name
        return call


def stream_report():
    """What the two libraries launch on. Both pointers 0 means both use the device's default stream."""
    t = int(torch.cuda.current_stream().cuda_stream)
    c = int(cupy.cuda.get_current_stream().ptr)
    return {"torch_current_stream": t, "cupy_current_stream": c, "same_default_stream": t == 0 and c == 0}


def versions():
    import triton
    return {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "cupy": cupy.__version__,
            "triton": triton.__version__, "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
            "cupy_adds": "-ftz=true (CuPy 14 compiler.py); NVRTC defaults otherwise (--fmad=true)"}

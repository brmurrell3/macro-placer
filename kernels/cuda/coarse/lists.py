"""The lists and constants of the coarse-step kernels as numpy arrays: no device, no torch.

`build_lists(static)` is the constructor of the Metal coarse step (../../metal/coarse.py): nets with the same pins
merged, routing pairs with the same pins merged, the horizontal and vertical key lists, the CSR lists the last kernel
sums over, and the keys of each grid in descending order of their number of pairs. It is kept apart from step.py so
that it can run where there is no CUDA; every array was compared with the Metal class on the Mac.
"""
from __future__ import annotations

import math
import types

import numpy as np


def _bits_to_float(column):
    return np.ascontiguousarray(np.asarray(column).astype(np.int32)).view(np.float32)


def _csr(owner, count):
    """Entries grouped by owner (owners >= count dropped): (start [count + 1], entry indices)."""
    keep = np.flatnonzero(owner < count)
    order = keep[np.argsort(owner[keep], kind="stable")]
    start = np.concatenate([[0], np.cumsum(np.bincount(owner[keep], minlength=count))])
    return start, order


def build_lists(st):
    """Every list and constant of the Metal coarse step as numpy arrays of the kernels' types (no device)."""
    L = types.SimpleNamespace()
    as_ = lambda array, kind: np.ascontiguousarray(array, dtype=kind)  # noqa: E731
    N, NH = int(st["n_macros"]), int(st["n_hard"])
    R, C = int(st["grid_rows"]), int(st["grid_cols"])
    L.N, L.NH, L.R, L.C, L.RC = N, NH, R, C, R * C
    L.KD, L.KR = int(st["dens_k"]), int(st["route_k"])
    if max(R, C) + 1 > 32 or N < L.KD:
        raise ValueError("grid side above 31 cells or fewer macros than selected density cells")
    L.half, L.high = as_(st["half"], np.float32), as_(st["high"], np.float32)
    L.fixed = as_(st["fixed_ref"], np.float32)
    L.mov = as_(np.asarray(st["movable"]), np.float32)
    L.xe, L.ye = as_(st["x_edges"], np.float32), as_(st["y_edges"], np.float32)
    L.xg, L.yg = as_(st["x_grid"], np.float32), as_(st["y_grid"], np.float32)

    # wirelength: nets as pin lists, nets with the same pins merged (weights added), CSR of pin slots per macro
    merged = {}
    for g in range(int(st["net_groups"])):
        owners, offsets = st[f"net{g}_owners"], np.ascontiguousarray(st[f"net{g}_offsets"], dtype=np.float32)
        mask, weights = st[f"net{g}_mask"], st[f"net{g}_weights"]
        bits = offsets.view(np.int32)
        for i in range(owners.shape[0]):
            key = tuple(sorted((int(owners[i, k]), int(bits[i, k, 0]), int(bits[i, k, 1])) for k in range(owners.shape[1]) if mask[i, k]))
            merged[key] = merged.get(key, 0.0) + float(weights[i])
    L.nets_given = int(sum(st[f"net{g}_owners"].shape[0] for g in range(int(st["net_groups"]))))
    L.NN = len(merged)
    pins = np.asarray([p for key in merged for p in key], dtype=np.int64).reshape(-1, 3)
    degree = np.asarray([len(key) for key in merged], dtype=np.int64)
    L.P = int(pins.shape[0])
    L.net_start = as_(np.concatenate([[0], np.cumsum(degree)]), np.int32)
    L.pin_macro = as_(pins[:, 0], np.int32)
    L.pin_off = as_(_bits_to_float(pins[:, 1:]).reshape(-1, 2), np.float32)
    L.net_w = as_(np.asarray(list(merged.values()), dtype=np.float64), np.float32)
    start, order = _csr(pins[:, 0], N)
    L.mstart, L.mentry = as_(start, np.int32), as_(order, np.int32)

    # density
    L.sig = as_(np.asarray(st["sizes"], dtype=np.float64) / math.sqrt(12.0), np.float32)
    L.area = as_(st["macro_area"], np.float32)

    # routing: merged pairs (same source pin and sink pin), then the two key lists
    so, to = (np.ascontiguousarray(st[k], dtype=np.float32).view(np.int32) for k in ("src_offset", "sink_offset"))
    cols = np.stack([st["src"].astype(np.int64), so[:, 0], so[:, 1], st["sink"].astype(np.int64), to[:, 0], to[:, 1]], axis=1).astype(np.int64)
    uniq, inverse = np.unique(cols, axis=0, return_inverse=True)
    weight = np.zeros(uniq.shape[0], dtype=np.float64)
    np.add.at(weight, inverse.ravel(), np.asarray(st["route_weight"], dtype=np.float64))
    L.pairs_given, L.pairs_merged = int(cols.shape[0]), int(uniq.shape[0])
    h_cap, v_cap = float(st["h_capacity"]), float(st["v_capacity"])
    parts = []
    # (key columns, own span offset, other macro, other span offset, capacity)
    for key_cols, own, other, other_off, cap in (((0, 2), 1, 3, 4, h_cap), ((3, 4), 5, 0, 2, v_cap)):
        keys, which = np.unique(uniq[:, list(key_cols)], axis=0, return_inverse=True)
        which = which.ravel()
        order = np.argsort(which, kind="stable")
        parts.append(types.SimpleNamespace(keys=keys, count=np.bincount(which, minlength=keys.shape[0]), own=_bits_to_float(uniq[order, own]),
                                           other=uniq[order, other], other_off=_bits_to_float(uniq[order, other_off]),
                                           w=weight[order] / cap))
    ph, pv = parts
    L.KA, L.KB = int(ph.keys.shape[0]), int(pv.keys.shape[0])
    L.KT = L.KA + L.KB
    L.E = 2 * L.pairs_merged
    L.KOFF = max(L.KA, L.KB)
    L.K2 = L.KOFF + NH
    kmac = as_(np.concatenate([ph.keys[:, 0], pv.keys[:, 0]]), np.int32)
    kwoff = as_(np.concatenate([_bits_to_float(ph.keys[:, 1]), _bits_to_float(pv.keys[:, 1])]), np.float32)
    kstart = as_(np.concatenate([[0], np.cumsum(np.concatenate([ph.count, pv.count]))]), np.int32).astype(np.int64)
    eoth = as_(np.concatenate([ph.other, pv.other]), np.int32)
    eown = as_(np.concatenate([ph.own, pv.own]), np.float32)
    eooff = as_(np.concatenate([ph.other_off, pv.other_off]), np.float32)
    ew = as_(np.concatenate([ph.w, pv.w]), np.float32)

    # the keys of each grid in descending order of their number of pairs (a permutation of the lists above)
    KA, M = L.KA, L.pairs_merged
    count = np.diff(kstart)
    perm = np.concatenate([np.argsort(-count[:KA], kind="stable"), KA + np.argsort(-count[KA:], kind="stable")])   # new key -> old key
    entries = np.concatenate([np.arange(kstart[k], kstart[k + 1]) for k in perm])                                    # new entry -> old entry
    assert entries.shape[0] == 2 * M and (entries[:M] < M).all() and (entries[M:] >= M).all()
    kmac, eoth = kmac[perm].astype(np.int64), eoth[entries].astype(np.int64)
    L.kmac, L.kwoff = as_(kmac, np.int32), as_(kwoff[perm], np.float32)
    L.kstart = as_(np.concatenate([[0], np.cumsum(count[perm])]), np.int32)
    L.eoth, L.eown, L.eooff, L.ew = as_(eoth, np.int32), as_(eown[entries], np.float32), as_(eooff[entries], np.float32), as_(ew[entries], np.float32)
    start, order = _csr(kmac, N)
    L.kms, L.kme = as_(start, np.int32), as_(order, np.int32)
    is_h = np.arange(2 * M) < M
    start, order = _csr(np.where(is_h, eoth, N), N)          # horizontal entries: the other pin's x
    L.exs, L.exe = as_(start, np.int32), as_(order, np.int32)
    start, order = _csr(np.where(is_h, N, eoth), N)          # vertical entries: the other pin's y
    L.eys, L.eye = as_(start, np.int32), as_(order, np.int32)

    ws = np.asarray(st["wl_scale"], dtype=np.float64)
    slope, assign = float(st["window_slope_per_cell"]), float(st["assign_sigma_cells"])
    L.cf = np.asarray([ws[0], ws[1], R / assign, C / assign, slope * R, slope * C, float(st["h_blockage"]) * R * C,
                       float(st["v_blockage"]) * R * C, float(R * C)], dtype=np.float32)

    # the value buffer: one segment per term
    L.seg, at = {}, 0
    for term, width in (("wirelength", L.NN), ("overlap", NH), ("density", L.KD), ("routing", 2 * R * C)):
        L.seg[term] = (at, at + width)
        at += width
    L.VW = at
    return L


ARRAYS = ("half", "high", "fixed", "mov", "xe", "ye", "xg", "yg", "net_start", "pin_macro", "pin_off", "net_w", "mstart", "mentry",
          "sig", "area", "kmac", "kwoff", "kstart", "eoth", "eown", "eooff", "ew", "kms", "kme", "exs", "exe", "eys", "eye", "cf")
SCALARS = ("N", "NH", "R", "C", "RC", "KD", "KR", "nets_given", "NN", "P", "pairs_given", "pairs_merged", "KA", "KB", "KT", "E",
           "KOFF", "K2", "VW")

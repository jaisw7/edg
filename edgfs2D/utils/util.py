# -*- coding: utf-8 -*-
import numpy as np
import torch as t
from typing_extensions import Union

np_map = {"float": np.float32, "double": np.float64}
np_rmap = {np.float32: "float", np.float64: "double"}

torch_map = {"float": t.float32, "double": t.float64}
torch_rmap = {t.float32: "float", t.float64: "double"}
torch_cmap = {"float": t.complex64, "double": t.complex128}


def to_torch(data: Union[t.Tensor, np.ndarray]):
    return data if isinstance(data, t.Tensor) else t.from_numpy(data)


def to_torch_device(data: np.ndarray, cfg):
    return t.from_numpy(data).to(device=cfg.device, dtype=cfg.ttype)


def filter_tol(mat, tol=1e-15, val=0.0):
    mat[abs(mat) < tol] = val
    return mat


def ndgrid(*v):
    return list(reversed(np.meshgrid(*v, indexing="ij")))


def fuzzysort(arr, idx, dim=0, tol=1e-6):
    # Extract our dimension and argsort
    arrd = arr[dim]
    srtdidx = sorted(idx, key=arrd.__getitem__)

    if len(srtdidx) > 1:
        i, ix = 0, srtdidx[0]
        for j, jx in enumerate(srtdidx[1:], start=1):
            if arrd[jx] - arrd[ix] >= tol:
                if j - i > 1:
                    srtdidx[i:j] = fuzzysort(arr, srtdidx[i:j], dim + 1, tol)
                i, ix = j, jx

        if i != j:
            srtdidx[i:] = fuzzysort(arr, srtdidx[i:], dim + 1, tol)

    return srtdidx


def split_vargs(args):
    vars = {}
    for key, value in args:
        keyscope = key.split("::")
        child = None
        parent = vars
        while keyscope:
            child = keyscope.pop(0)
            if not parent or child not in parent:
                parent[child] = dict()
            if keyscope:
                parent = parent[child]
            else:
                parent[child] = value
    return vars

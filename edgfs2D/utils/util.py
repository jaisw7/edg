# -*- coding: utf-8 -*-
# import pycuda.driver as cuda
# from pycuda import compiler
import numpy as np
import torch as t
from gimmik import generate_mm
from typing_extensions import Union

np_map = {"float": np.float32, "double": np.float64}
np_rmap = {np.float32: "float", np.float64: "double"}

torch_map = {"float": t.float32, "double": t.float64}
torch_rmap = {t.float32: "float", t.float64: "double"}


def get_kernel(module, funcname, args):
    func = module.get_function(funcname)
    func.prepare(args)
    func.set_cache_config(cuda.func_cache.PREFER_L1)
    return func


def to_torch(data: Union[t.Tensor, np.ndarray]):
    return data if isinstance(data, t.Tensor) else t.from_numpy(data)


def filter_tol(mat, tol=1e-15, val=0.0):
    mat[abs(mat) < tol] = val
    return mat


def get_mm_kernel(mat, alpha=1.0, beta=0.0, tol=1e-15):
    matSrc = generate_mm(
        filter_tol(mat, tol=tol),
        dtype=mat.dtype,
        platform="cuda",
        alpha=alpha,
        beta=beta,
    )
    matMod = compiler.SourceModule(matSrc)
    matKern = get_kernel(matMod, "gimmik_mm", "iPiPi")
    return matKern


def get_mm_proxycopy_kernel(mat, alpha=1.0, beta=0.0, tol=1e-15):
    class generate_mm_proxycopy:
        def prepared_call(*x):
            cuda.memcpy_dtod(
                x[5], x[3], x[2] * mat.shape[0] * mat.dtype.itemsize
            )

    return generate_mm_proxycopy


def cross(args):
    return it.product(*args)


def get_kernel_op(module, names, pointers):
    return map(
        lambda v: lambda *args: get_kernel(module, v[0], v[1]).prepared_call(
            grid_Nv, block, *list(map(lambda c: c.ptr, args))
        ),
        zip(names, pointers),
    )


from functools import reduce


def ndkron(*v):
    return reduce(np.kron, v)


def ndgrid(*v):
    return list(reversed(np.meshgrid(*v, indexing="ij")))


def check(truth_value, *args):
    if not truth_value:
        raise ValueError(*args)


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

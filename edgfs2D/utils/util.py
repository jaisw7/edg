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
    if dim >= len(arr):
        return idx

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


class IterableProxy(list):
    def __getattr__(self, attr):
        return IterableProxy(getattr(x, attr) for x in self)

    def __setattr__(self, attr, val):
        for x in self:
            setattr(x, attr, val)

    def __call__(self, *args, **kwargs):
        return IterableProxy(x(*args, **kwargs) for x in self)

    def __getitem__(self, index):
        return IterableProxy([x[index] for x in self])

    def apply(self, func):
        return IterableProxy(func(x) for x in self)

    # access individual entries of the list
    def get(self, index):
        return super().__getitem__(index)

    def set(self, index, value):
        super().__setitem__(index, value)

    # operations on underlying element
    def __applyattr__(self, attr, other):
        return IterableProxy(getattr(x, attr)(other) for x in self)

    # Arithmetic operations
    def __add__(self, other):
        return self.__applyattr__("__add__", other)

    def __sub__(self, other):
        return self.__applyattr__("__sub__", other)

    def __mul__(self, other):
        return self.__applyattr__("__mul__", other)

    def __truediv__(self, other):
        return self.__applyattr__("__truediv__", other)

    def __floordiv__(self, other):
        return self.__applyattr__("__floordiv__", other)

    def __mod__(self, other):
        return self.__applyattr__("__mod__", other)

    def __pow__(self, other):
        return self.__applyattr__("__pow__", other)

    # Unary operations
    def __neg__(self):
        return self.__applyattr__("__neg__", None)

    def __pos__(self):
        return self.__applyattr__("__pos__", None)

    def __abs__(self):
        return self.__applyattr__("__abs__", None)

    # Bitwise operations
    def __and__(self, other):
        return self.__applyattr__("__and__", other)

    def __or__(self, other):
        return self.__applyattr__("__or__", other)

    def __xor__(self, other):
        return self.__applyattr__("__xor__", other)

    def __lshift__(self, other):
        return self.__applyattr__("__lshift__", other)

    def __rshift__(self, other):
        return self.__applyattr__("__rshift__", other)

    # Comparisons
    def __eq__(self, other):
        return self.__applyattr__("__eq__", other)

    def __ne__(self, other):
        return self.__applyattr__("__ne__", other)

    def __lt__(self, other):
        return self.__applyattr__("__lt__", other)

    def __le__(self, other):
        return self.__applyattr__("__le__", other)

    def __gt__(self, other):
        return self.__applyattr__("__gt__", other)

    def __ge__(self, other):
        return self.__applyattr__("__ge__", other)

# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
from typing_extensions import Dict, List, Tuple, TypeAlias


class Shape(str):
    pass


class FieldData(Dict[Shape, torch.Tensor]):
    """Static functions"""

    @staticmethod
    def apply(src: FieldData, funcName: str) -> FieldData:
        function = getattr(torch, funcName)
        dest = FieldData()
        for shape in src.keys():
            dest[shape] = function(src[shape])
        return dest

    @staticmethod
    def zeros_like(src: FieldData) -> FieldData:
        dest = FieldData()
        for shape in src.keys():
            dest[shape] = torch.zeros_like(src[shape])
        return dest

    @staticmethod
    def ones_like(src: FieldData) -> FieldData:
        return FieldData.apply(src, "ones_like")

    """Public functions"""

    def copy_(self, src: FieldData):
        for shape in src.keys():
            self.get(shape).copy_(src[shape])

    def unary_return(self, funcName: str, *args, **kwargs) -> FieldData:
        dest = FieldData()
        for shape, value in self.items():
            dest[shape] = getattr(value, funcName)(*args, **kwargs)
        return dest

    def clone(self, *args, **kwargs):
        return self.unary_return("clone", *args, **kwargs)

    def sum(self, *args, **kwargs):
        return self.unary_return("sum", *args, **kwargs)

    def clone_as_zeros(self):
        return FieldData.zeros_like(self)

    """Inline operators"""

    def unary_apply_(self, funcName: str, *args, **kwargs):
        for shape, value in self.items():
            getattr(value, funcName)(*args, **kwargs)
        return self

    def binary_apply_(self, funcName: str, other: FieldData, *args, **kwargs):
        for shape, value in self.items():
            getattr(value, funcName)(other[shape], *args, **kwargs)
        return self

    def mul_(self, *args, **kwargs):
        return self.unary_apply_("mul_", *args, **kwargs)

    def sin_(self, *args, **kwargs):
        return self.unary_apply_("sin_", *args, **kwargs)

    def add_(self, other: FieldData, *args, **kwargs):
        return self.binary_apply_("add_", other, *args, **kwargs)

    def sub_(self, other: FieldData, *args, **kwargs):
        return self.binary_apply_("sub_", other, *args, **kwargs)


FieldDataList: TypeAlias = List[FieldData]
FieldDataTuple: TypeAlias = Tuple[FieldData]

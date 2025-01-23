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

    def clone(self):
        dest = FieldData()
        for shape, value in self.items():
            dest[shape] = value.clone()
        return dest

    def clone_as_zeros(self):
        return FieldData.zeros_like(self)

    """Inline operators"""

    def apply_(self, funcName: str, *args, **kwargs) -> FieldData:
        for shape, value in self.items():
            getattr(value, funcName)(*args, **kwargs)

    def mul_(self, *args, **kwargs):
        self.apply_("mul_", *args, **kwargs)

    def sin_(self, *args, **kwargs):
        self.apply_("sin_", *args, **kwargs)

    def add_(self, other: FieldData, *args, **kwargs):
        for shape, value in self.items():
            getattr(value, "add_")(other[shape], *args, **kwargs)


FieldDataList: TypeAlias = List[FieldData]
FieldDataTuple: TypeAlias = Tuple[FieldData]

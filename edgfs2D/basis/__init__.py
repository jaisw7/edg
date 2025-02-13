# -*- coding: utf-8 -*-
from edgfs2D.basis.base import BaseBasis
from edgfs2D.basis.fhz import FernandezHickenZingg  # noqa
from edgfs2D.basis.hw import HesthavenWarburton  # noqa
from edgfs2D.utils.nputil import subclass_where

basis_sect = "basis"


def get_basis_by_shape(cfg, shape, *args, **kwargs):
    sect = "{}-{}".format(basis_sect, shape)
    basisKind = cfg.lookup(sect, "kind")
    return subclass_where(BaseBasis, kind=basisKind, shape=shape)(
        cfg, sect, *args, **kwargs
    )

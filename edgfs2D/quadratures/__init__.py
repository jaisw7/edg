# -*- coding: utf-8 -*-
from edgfs2D.quadratures.base import BaseQuadrature
from edgfs2D.quadratures.Jacobi import GaussLegendre, GaussLegendreLobatto
from edgfs2D.utils.nputil import subclass_where

quadrature_sect = "quadrature"


def get_quadrature_by_name(cfg, name, *args, **kwargs):
    quadratureKind = cfg.lookup(name, "kind")
    return subclass_where(BaseQuadrature, kind=quadratureKind)(cfg, *args, **kwargs)


def get_quadrature_by_shape(cfg, shape, *args, **kwargs):
    return get_quadrature_by_name(
        cfg, "{}-{}".format(quadrature_sect, name), *args, **kwargs
    )

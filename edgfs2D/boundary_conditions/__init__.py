# -*- coding: utf-8 -*-
from edgfs2D.utils.nputil import subclass_where

boundary_conditions_sect = "boundary-conditions"


def get_bc_by_cls_and_name(cfg, name, cls, *args, **kwargs):
    sect_name = "{}-{}".format(boundary_conditions_sect, name)
    icKind = cfg.lookup(sect_name, "kind")
    return subclass_where(cls, kind=icKind)(
        cfg.get_section(sect_name), *args, **kwargs
    )

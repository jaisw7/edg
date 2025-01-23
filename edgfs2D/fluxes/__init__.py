# -*- coding: utf-8 -*-
from edgfs2D.utils.nputil import subclass_where

flux_sect = "flux"


def get_flux_by_cls_and_name(cfg, name, cls, *args, **kwargs):
    sect_name = "{}-{}".format(flux_sect, name)
    fluxKind = cfg.lookup(sect_name, "kind")
    return subclass_where(cls, kind=fluxKind)(
        cfg.get_section(sect_name), *args, **kwargs
    )

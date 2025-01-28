# -*- coding: utf-8 -*-
from edgfs2D.utils.nputil import subclass_where

entropy_flux_sect = "entropy-flux"


def get_eflux_by_cls_and_name(cfg, name, cls, *args, **kwargs):
    sect_name = "{}-{}".format(entropy_flux_sect, name)
    if not cfg.has_section(sect_name):
        return None
    efluxKind = cfg.lookup(sect_name, "kind")
    return subclass_where(cls, kind=efluxKind)(
        cfg.get_section(sect_name), *args, **kwargs
    )

# -*- coding: utf-8 -*-
from edgfs2D.std.scattering.base import BaseScatteringModel
from edgfs2D.std.scattering.fast_spectral import FastSpectral
from edgfs2D.utils.nputil import subclass_where

scattering_sect = "scattering-model"


def get_scattering_model_by_name(cfg, name, *args, **kwargs):
    smKind = cfg.lookup(name, "kind")
    return subclass_where(BaseScatteringModel, kind=smKind)(cfg, name, *args, **kwargs)


def get_scattering_model(cfg, *args, **kwargs):
    return get_scattering_model_by_name(cfg, scattering_sect, *args, **kwargs)

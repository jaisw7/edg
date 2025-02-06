# -*- coding: utf-8 -*-
from edgfs2D.scattering.base import BaseScatteringModel
from edgfs2D.scattering.fast_spectral import (  # noqa
    FastSpectral,
    PenalizedFastSpectral,
    PenalizedFastSpectralMixingRegime,
)
from edgfs2D.scattering.relaxation_models import (  # noqa
    BaseRelaxationModel,
    BgkRelaxation,
    BgkRelaxationMixingRegime,
)
from edgfs2D.utils.nputil import subclass_where

scattering_sect = "scattering-model"
relaxation_sect = "relaxation-model"


def get_scattering_model_by_name_cls(cls, cfg, name, *args, **kwargs):
    smKind = cfg.lookup(name, "kind")
    smSect = cfg.get_section(name)
    return subclass_where(cls, kind=smKind)(smSect, *args, **kwargs)


def get_scattering_model(cfg, *args, **kwargs):
    return get_scattering_model_by_name_cls(
        BaseScatteringModel, cfg, scattering_sect, *args, **kwargs
    )


def get_relaxation_model(cfg, *args, **kwargs):
    return get_scattering_model_by_name_cls(
        BaseRelaxationModel, cfg, relaxation_sect, *args, **kwargs
    )

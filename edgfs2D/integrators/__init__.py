# -*- coding: utf-8 -*-
from edgfs2D.integrators.base import BaseIntegrator
from edgfs2D.integrators.euler import EulerIntegrator  # noqa
from edgfs2D.integrators.lserk import LserkIntegrator  # noqa
from edgfs2D.utils.nputil import subclass_where

integrator_sect = "integrator"


def get_integrator_by_name(cfg, name, *args, **kwargs):
    integKind = cfg.lookup(name, "kind")
    return subclass_where(BaseIntegrator, kind=integKind)(
        cfg.get_section(name), *args, **kwargs
    )


def get_integrator(cfg, *args, **kwargs):
    return get_integrator_by_name(cfg, integrator_sect, *args, **kwargs)

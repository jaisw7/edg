# -*- coding: utf-8 -*-
import re

from edgfs2D.plugins.base import BasePlugin
from edgfs2D.plugins.nancheck import NanCheckPlugin
from edgfs2D.plugins.residual import ResidualPlugin
from edgfs2D.utils.nputil import subclass_where

plugins_sect = "plugin"


def get_plugin_sections(cfg):
    for sect in cfg.sections():
        m = re.match(rf"{plugins_sect}-(.+?)$", sect)
        if m:
            yield m


def get_plugin_for_solver(cfg, solver, name, *args, **kwargs):
    sect = cfg.get_section("{}-{}".format(plugins_sect, name))
    cls = subclass_where(BasePlugin, kind=name)
    if solver in cls.allowed_solvers:
        return cls(sect, *args, **kwargs)
    else:
        return None

# -*- coding: utf-8 -*-
from edgfs2D.std.limiters.base import BaseLimiter
from edgfs2D.utils.nputil import subclass_where

limiter_sect = "limiter"


def get_limiter_by_name(cfg, name, *args, **kwargs):
    limiterKind = cfg.lookup(name, "kind")
    return subclass_where(BaseLimiter, kind=limiterKind)(cfg, *args, **kwargs)


def get_limiter(cfg, *args, **kwargs):
    return get_limiter_by_name(cfg, limiter_sect, *args, **kwargs)

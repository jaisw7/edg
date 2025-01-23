# -*- coding: utf-8 -*-
from edgfs2D.std.writers.base import BaseWriter
from edgfs2D.utils.nputil import subclass_where

writer_sect = "writer"


def get_writer_by_name(cfg, name, *args, **kwargs):
    writerKind = cfg.lookup(name, "kind")
    return subclass_where(BaseWriter, kind=writerKind)(cfg, *args, **kwargs)


def get_writer(cfg, *args, **kwargs):
    return get_writer_by_name(cfg, writer_sect, *args, **kwargs)

# -*- coding: utf-8 -*-

from edgfs2D.physical_mesh.readers.base import (  # noqa
    BaseReader,
    NodalMeshAssembler,
)
from edgfs2D.physical_mesh.readers.gmsh import GmshReader  # noqa
from edgfs2D.utils.nputil import subclass_where


def get_reader_by_name(name, *args, **kwargs):
    return subclass_where(BaseReader, name=name)(*args, **kwargs)

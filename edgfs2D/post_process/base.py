# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

from edgfs2D.physical_mesh.dg_mesh import DgMesh


class BasePostProcessor(object, metaclass=ABCMeta):
    kind = None

    def __init__(self, dgmesh: DgMesh, *args, **kwargs):
        self._dgmesh = dgmesh

    @abstractmethod
    def execute(self):
        pass

    @property
    def dgmesh(self):
        return self._dgmesh

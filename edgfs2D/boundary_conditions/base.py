from abc import ABCMeta, abstractmethod


class BaseBoundaryCondition(object, metaclass=ABCMeta):
    def __init__(self, cfg, nodes, normals, **kwargs):
        self._cfg = cfg
        self._nodes = nodes
        self._normals = normals

    @property
    def nodes(self):
        return self._nodes

    @property
    def normals(self):
        return self._normals

    @abstractmethod
    def apply(self, *args, **kwargs):
        pass

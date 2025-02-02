from abc import ABCMeta, abstractmethod

import torch

from edgfs2D.distribution_mesh.dgdist_mesh import DgDistMesh
from edgfs2D.utils.dictionary import SubDictionary


class BaseScatteringModel(object, metaclass=ABCMeta):
    kind = None
    allowed_solvers = None

    def __init__(
        self, cfg: SubDictionary, distmesh: DgDistMesh, *args, **kwargs
    ):
        self._cfg = cfg
        self._distmesh = distmesh

    @property
    def vmesh(self):
        return self._distmesh.vmesh

    @property
    def dgmesh(self):
        return self._distmesh.dgmesh

    @abstractmethod
    def solve(
        self,
        curr_time: torch.float64,
        element_data: torch.Tensor,
        out: torch.Tensor,
    ):
        pass

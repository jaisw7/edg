from abc import ABCMeta, abstractmethod

import torch

from edgfs2D.utils.dictionary import SubDictionary
from edgfs2D.utils.util import to_torch_device
from edgfs2D.velocity_mesh.base import BaseVelocityMesh


class BaseScatteringModel(object, metaclass=ABCMeta):
    kind = None

    def __init__(
        self, cfg: SubDictionary, vmesh: BaseVelocityMesh, *args, **kwargs
    ):
        self._cfg = cfg
        self._vmesh = vmesh

    @property
    def vmesh(self):
        return self._vmesh

    @abstractmethod
    def solve(
        self,
        curr_time: torch.float64,
        element_data: torch.Tensor,
        out: torch.Tensor,
    ):
        pass

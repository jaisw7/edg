from abc import ABCMeta, abstractmethod, abstractproperty

import torch
from typing_extensions import Callable

from edgfs2D.fields.types import FieldData
from edgfs2D.time.physical_time import PhysicalTime
from edgfs2D.utils.dictionary import SubDictionary

RhsFunction = Callable[[torch.float64, FieldData], FieldData]


class BaseIntegrator(object, metaclass=ABCMeta):
    kind = None

    def __init__(self, cfg: SubDictionary, *args, **kwargs):
        pass

    @abstractproperty
    def order(self):
        pass

    @abstractproperty
    def num_steps(self):
        pass

    @abstractmethod
    def integrate_at_step(
        self, step: int, time: PhysicalTime, u: FieldData, rhs: RhsFunction
    ) -> FieldData:
        pass

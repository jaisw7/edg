from abc import ABCMeta, abstractmethod

from edgfs2D.solvers.base import BaseSolver
from edgfs2D.utils.dictionary import SubDictionary


class BasePlugin(object, metaclass=ABCMeta):
    kind = None
    allowed_solvers = None

    @abstractmethod
    def __init__(self, cfg: SubDictionary, solver: BaseSolver):
        pass

    @abstractmethod
    def __call__(self):
        pass

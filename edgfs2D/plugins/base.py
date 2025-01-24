from abc import ABCMeta, abstractmethod

import numpy as np

from edgfs2D.solvers.base import BaseSolver
from edgfs2D.utils.dictionary import SubDictionary


class BasePlugin(object, metaclass=ABCMeta):
    kind = None
    allowed_solvers = None

    @abstractmethod
    def __init__(self, cfg: SubDictionary, solver: BaseSolver):
        self._cfg = cfg
        self._solver = solver

    def get_nsteps(self):
        if not (self._cfg.has_option("nsteps") or self._cfg.has_option("time")):
            raise ValueError(
                "either nsteps or time must be provided for plugin",
                self.kind,
            )

        if self._cfg.has_option("nsteps"):
            return self._cfg.lookupint("nsteps")
        else:
            time = self._cfg.lookupfloat("time")
            return int(np.ceil(time / self._solver.time.dt))

    @abstractmethod
    def __call__(self):
        pass

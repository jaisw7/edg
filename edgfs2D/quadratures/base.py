from abc import ABCMeta, abstractmethod

import numpy as np


class BaseQuadrature(object, metaclass=ABCMeta):
    kind = None

    def __init__(self, cfg, name, *args, **kwargs):
        self.cfg = cfg

    @property
    def Nq(self):
        return self._Nq

    @property
    def z(self):
        return self._z

    @property
    def w(self):
        return self._w

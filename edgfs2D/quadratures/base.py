from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np


class BaseQuadrature(object, metaclass=ABCMeta):
    kind = None

    @abstractmethod
    def __init__(self, cfg, name, *args, **kwargs):
        self.cfg = cfg

    @abstractproperty
    def Nq(self):
        pass

    @abstractproperty
    def z(self):
        pass

    @abstractproperty
    def w(self):
        pass

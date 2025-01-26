from abc import ABCMeta, abstractmethod, abstractproperty


class BaseFlux(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg

    @abstractmethod
    def apply(self, *args, **kwargs):
        pass

    @abstractproperty
    def velocity(self):
        pass

from abc import ABCMeta, abstractmethod


class BaseBoundaryCondition(object, metaclass=ABCMeta):
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg

    @abstractmethod
    def apply(self, *args, **kwargs):
        pass

from abc import ABCMeta, abstractmethod


class BaseBoundaryCondition(object, metaclass=ABCMeta):
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg

    @abstractmethod
    def apply(self, *args, **kwargs):
        pass

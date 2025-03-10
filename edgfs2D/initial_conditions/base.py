from abc import ABCMeta, abstractmethod


class BaseInitialCondition(object, metaclass=ABCMeta):
    def __init__(self, cfg, **kwargs):
        self._cfg = cfg

    @abstractmethod
    def apply(self, *args, **kwargs):
        pass

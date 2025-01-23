from abc import ABCMeta, abstractmethod, abstractproperty


class BaseSolver(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, cfg, *args, **kwargs):
        pass

    @abstractproperty
    def curr_fields(self):
        pass

    @abstractproperty
    def prev_fields(self):
        pass

    @abstractproperty
    def time(self):
        pass

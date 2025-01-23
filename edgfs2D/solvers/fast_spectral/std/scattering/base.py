from abc import ABCMeta, abstractmethod


class BaseScatteringModel(object, metaclass=ABCMeta):
    kind = None

    def __init__(self, cfg, name, nondim, velocitymesh, *args, **kwargs):
        self.cfg = cfg
        self.sect = name
        self.nondim = nondim
        self.vm = velocitymesh

    @abstractmethod
    def solve(self, d_arr_in, d_arr_out, elem, upt):
        pass

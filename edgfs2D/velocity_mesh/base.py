from abc import ABCMeta, abstractmethod, abstractproperty


class BaseVelocityMesh(object, metaclass=ABCMeta):

    def __init__(self, cfg, nondim, *args, **kwargs):
        self._cfg = cfg
        self._nondim = nondim

    @abstractmethod
    def _construct_velocity_mesh(self, nondim):
        pass

    @abstractproperty
    def num_points(self):
        pass

    @abstractproperty
    def extents(self):
        pass

    @abstractproperty
    def shape(self):
        pass

    @abstractproperty
    def points(self):
        pass

    @abstractproperty
    def weights(self):
        pass

    @property
    def nondim(self):
        return self._nondim

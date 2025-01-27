from abc import ABCMeta, abstractmethod, abstractproperty
from pathlib import Path

from edgfs2D.fields.types import FieldData


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

    @abstractproperty
    def mesh(self):
        pass

    @abstractmethod
    def error_norm(self, err: FieldData):
        pass

    @abstractmethod
    def write(self, path: Path):
        pass


class MomentMixin:

    @abstractmethod
    def write_moment(self, path: Path):
        pass

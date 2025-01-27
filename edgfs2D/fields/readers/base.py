from abc import ABCMeta, abstractmethod
from pathlib import Path

from edgfs2D.fields.types import Shape


class BaseFieldReader(object, metaclass=ABCMeta):
    def __init__(self, path: Path):
        if not path.exists():
            raise ValueError(f"file path {path.absolute()} not valid")
        self._path = path

    @abstractmethod
    def read_metadata(self, key):
        pass

    @abstractmethod
    def read_field_names(self):
        pass

    @abstractmethod
    def read_field_data(self, fieldname: str, shape: Shape):
        pass

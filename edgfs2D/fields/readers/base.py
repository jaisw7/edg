from abc import ABCMeta, abstractmethod
from pathlib import Path


class BaseFieldReader(object, metaclass=ABCMeta):
    def __init__(self, path: Path):
        if not path.exists():
            raise ValueError(f"file path {path.absolute()} not valid")
        self._path = path

    @abstractmethod
    def read_metadata(self, key):
        pass

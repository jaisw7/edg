from abc import ABCMeta, abstractmethod
from pathlib import Path

from typing_extensions import override

from edgfs2D.fields.types import FieldData


class BaseFieldWriter(object, metaclass=ABCMeta):
    def __init__(self, path: Path):
        self._path = path

    @abstractmethod
    def write_metadata(self, key, val):
        pass

    @override
    def write_fields(self, data: FieldData):
        pass

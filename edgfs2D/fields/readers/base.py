from abc import ABCMeta
from pathlib import Path

import h5py


class BaseFieldReader(object, metaclass=ABCMeta):
    def __init__(self, path: Path):
        if not path.exists():
            raise ValueError(f"file path {path.absolute()} not valid")
        self._path = path

    def get_metadata(self, key):
        with h5py.File(self._path, "r") as h5f:
            return h5f.attrs.get(key)

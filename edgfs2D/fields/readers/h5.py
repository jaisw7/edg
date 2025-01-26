from abc import ABCMeta
from pathlib import Path

import h5py
from typing_extensions import override

from edgfs2D.fields.readers.base import BaseFieldReader


class H5FieldReader(BaseFieldReader):
    mode = "r"

    def __init__(self, path: Path):
        if not path.exists():
            raise ValueError(f"file path {path.absolute()} not valid")
        self._path = path

    @override
    def read_metadata(self, key):
        with h5py.File(self._path, self.mode) as h5f:
            return h5f.attrs.get(key)

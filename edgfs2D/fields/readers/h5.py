from abc import ABCMeta
from pathlib import Path

import h5py
from typing_extensions import override

from edgfs2D.fields.readers.base import BaseFieldReader
from edgfs2D.fields.types import Shape


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

    @override
    def read_field_names(self):
        with h5py.File(self._path, self.mode) as h5f:
            return [str(key) for key in h5f.keys()]

    @override
    def read_field(self, fieldname: str):
        with h5py.File(self._path, self.mode) as h5f:
            return {shape: data[:] for shape, data in h5f[fieldname].items()}

    @override
    def read_field_data(self, fieldname: str, shape: Shape):
        with h5py.File(self._path, self.mode) as h5f:
            return h5f[fieldname][shape][:]

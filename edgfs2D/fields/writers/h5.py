from pathlib import Path

import h5py
from typing_extensions import override

from edgfs2D.fields.types import FieldData
from edgfs2D.fields.writers.base import BaseFieldWriter


class H5FieldWriter(BaseFieldWriter):
    mode = "a"
    extn = ".h5"
    compression = "gzip"

    def __init__(self, path: Path):
        self._path = (
            path
            if str(path).endswith(self.extn)
            else Path(str(path) + self.extn)
        )

        # clear contents
        with h5py.File(self._path, "w") as h5f:  # noqa
            pass

    @override
    def write_metadata(self, key, val):
        with h5py.File(self._path, self.mode) as h5f:
            h5f.attrs[key] = val

    @override
    def write_fields(self, data: FieldData):
        with h5py.File(self._path, self.mode) as h5f:
            for key, val in data.items():
                group = h5f.create_group(key)
                for shape, soln in val.items():
                    group.create_dataset(
                        shape,
                        data=soln.cpu().numpy(),
                        compression=self.compression,
                    )

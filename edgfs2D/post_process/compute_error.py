# -*- coding: utf-8 -*-
from argparse import ArgumentParser, FileType

import numpy as np
import pyvista as pv
from loguru import logger
from typing_extensions import override

from edgfs2D.post_process.base import BasePostProcessor


class ComputeErrorPostProcessor(BasePostProcessor):
    kind = "compute_error"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pargs = self.parse_args()
        self._vf1 = pargs.vf1.name
        self._vf2 = pargs.vf2.name

    @override
    def parse_args(self):
        parser = ArgumentParser(
            description="Compute Error between data stored in two vtu files"
        )
        parser.add_argument("vf1", type=FileType("r"), help="first vtu file")
        parser.add_argument("vf2", type=FileType("r"), help="second vtu file")
        return parser.parse_args(self.args)

    @override
    def execute(self):
        file1 = pv.read(self._vf1)
        file2 = pv.read(self._vf2)

        # Check if the number of points is the same
        if file1.number_of_points != file2.number_of_points:
            print("The files have different numbers of points")
        else:
            if np.abs(file1.points - file2.points).max() > 1e-10:
                raise RuntimeError("The files have different point coordinates")

        # Check if the number of cells is the same
        if file1.number_of_cells != file2.number_of_cells:
            print("The files have different numbers of cells.")
        else:
            if not all(
                np.array_equal(c1, c2)
                for c1, c2 in zip(file1.cells, file2.cells)
            ):
                raise RuntimeError("The files have different cell connectivity")

        # Check if the cell types are the same
        if not np.array_equal(file1.celltypes, file2.celltypes):
            raise RuntimeError("The files have different cell types")

        # Check if the data arrays are same
        if file1.array_names != file2.array_names:
            raise RuntimeError("Data between the two files is not consistent")

        for field in file1.array_names:
            data1, data2 = file1[field], file2[field]
            mse = ((data1 - data2) ** 2).mean()
            logger.info("Field: {}, Mean Squared Error: {}", field, mse)

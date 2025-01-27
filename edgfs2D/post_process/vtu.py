# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import numpy as np
from typing_extensions import override

from edgfs2D.fields.readers.h5 import H5FieldReader
from edgfs2D.physical_mesh.dg_mesh import DgMesh
from edgfs2D.post_process.base import BasePostProcessor


class VtuPostProcessor(BasePostProcessor):
    kind = "vtu"
    extn = ".vtu"

    def __init__(self, dgmesh: DgMesh, *args, **kwargs):
        super().__init__(dgmesh, *args, **kwargs)

    @override
    def execute(self):
        time = self.dgmesh.pmesh.time
        vtufilename = time.args.soln.with_suffix(self.extn)

        with open(vtufilename, "w") as f:
            writeln = lambda s: f.write((s + "\n"))
            self.write_header(writeln)
            self.write_unstructured_grid(writeln)
            self.write_footer(writeln)

    def write_header(self, writeln):
        byte_order = {
            "little": "LittleEndian",
            "big": "BigEndian",
        }
        writeln(rf'<?xml version="1.0"?>')
        writeln(rf'<VTKFile type="UnstructuredGrid"')
        writeln(rf'         version="0.1"')
        writeln(rf'         byte_order="{byte_order[sys.byteorder]}">')

    def write_footer(self, writeln):
        writeln(rf"</VTKFile>")

    def write_unstructured_grid(self, writeln):
        writeln(rf"<UnstructuredGrid>")
        self.write_piece(writeln)
        writeln(rf"</UnstructuredGrid>")

    def write_piece(self, writeln):
        mesh = self.dgmesh
        time = mesh.pmesh.time
        basis = mesh.get_basis_at_shapes
        shape = next(iter(basis.keys()))
        reader = H5FieldReader(time.args.soln)
        fields = reader.read_field_names()

        if len(basis.keys()) != 1 or shape != "tri":
            raise RuntimeError("only tri elements supported as of now")

        # define uniform nodes on the basis elements
        basis = basis[shape]
        celldata = CellData(2 * basis.num_nodes)
        points = celldata.nodes()

        # interpolate to new nodes
        interp_op = basis.interpolation_op(points)
        points = basis.interpolate(mesh.get_element_nodes[shape], interp_op)
        points = np.pad(points, [(0, 0), (0, 0), (0, 1)], "constant")

        npts = points.shape[0] * points.shape[1]
        ncells = 0

        # write piece header
        writeln(rf'<Piece NumberOfPoints="{npts}" NumberOfCells="{ncells}">')

        # write points
        writeln(r"<Points>")
        writeln(r'<DataArray type="Float64"')
        writeln(r'           Name="Points"')
        writeln(r'           NumberOfComponents="3">')
        writeln(" ".join(map(str, points.ravel())))
        writeln(r"</DataArray>")
        writeln(r"</Points>")

        # write cells and connectivity
        # self.write_cells(writeln, pts)

        # write point data
        writeln(r"<PointData>")
        for field in fields:
            data = reader.read_field_data(field, shape)
            data_new = basis.interpolate(data, interp_op)
            writeln(rf'<DataArray type="Float64"')
            writeln(rf'           Name="{field}">')
            writeln(" ".join(map(str, data_new.ravel())))
            writeln(r"</DataArray>")
        writeln(r"</PointData>")

        # write piece footer
        writeln(rf"</Piece>")


class CellData(object):

    def __init__(self, n):
        self.n = n

    def offsets(self):
        return np.cumsum(3 * np.ones(self.n**2, dtype=int))

    def types(self):
        return 5 * np.ones(self.n**2, dtype=int)

    def connectivity(self, n):
        n = self.n
        conlst = []

        for row in range(n, 0, -1):
            # Lower and upper indices
            l = (n - row) * (n + row + 3) // 2
            u = l + row + 1

            # Base offsets
            off = [l, l + 1, u, u + 1, l + 1, u]

            # Generate current row
            subin = np.ravel(np.arange(row - 1)[..., None] + off)
            subex = [ix + row - 1 for ix in off[:3]]

            # Extent list
            conlst.extend([subin, subex])

        return np.hstack(conlst)

    def nodes(self):
        n = self.n
        pts = np.linspace(-1, 1, n + 1)
        return np.array(
            [(p, q) for i, q in enumerate(pts) for p in pts[: (n + 1 - i)]]
        )

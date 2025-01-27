# -*- coding: utf-8 -*-
import base64
import sys
import zlib
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

        with open(vtufilename, "wb") as f:
            writeln = lambda s: f.write((s + "\n").encode("utf-8"))
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
        writeln(rf'         compressor="vtkZLibDataCompressor"')
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
        nodes = celldata.nodes()

        # interpolate to new nodes
        interp_op = basis.interpolation_op(nodes)
        points = basis.interpolate(mesh.get_element_nodes[shape], interp_op)
        points = np.pad(points, [(0, 0), (0, 0), (0, 1)], "constant")

        # information about interpolated data
        npts, neles, ncells = *points.shape[:2], celldata.num_cells()

        # cell information
        conn = celldata.connectivity()
        connectivity = np.tile(conn, (neles, 1))
        connectivity += (np.arange(neles) * len(nodes))[:, None]
        offsets = np.tile(celldata.offsets(), (neles, 1))
        offsets += (np.arange(neles) * len(conn))[:, None]
        types = np.tile(celldata.types(), neles)

        # write piece header
        writeln(rf'<Piece NumberOfPoints="{npts * neles}"')
        writeln(rf'       NumberOfCells="{ncells * neles}">')

        # write points
        writeln(r"<Points>")
        self.write_data_array(writeln, "Points", points.swapaxes(0, 1), 3)
        writeln(r"</Points>")

        # write cells
        writeln(r"<Cells>")
        self.write_data_array(writeln, "connectivity", connectivity, 1)
        self.write_data_array(writeln, "offsets", offsets, 1)
        self.write_data_array(writeln, "types", types, 1)
        writeln(r"</Cells>")

        # write point data
        writeln(r"<PointData>")
        for field in fields:
            data = reader.read_field_data(field, shape)
            interp_data = basis.interpolate(data, interp_op)
            self.write_data_array(writeln, field, interp_data.swapaxes(0, 1), 1)
        writeln(r"</PointData>")

        # write piece footer
        writeln(rf"</Piece>")

    def _chunk_it(self, array, n):
        for start in range(0, len(array), n):
            yield array[start : start + n]

    def text_writer_uncompressed(self, writeln, data):
        data_bytes = data.tobytes()
        header = np.array(len(data_bytes), dtype=np.uint32)
        writeln(base64.b64encode(header.tobytes() + data_bytes).decode())

    def text_writer_compressed(self, writeln, data):
        max_block_size = 32768
        data_bytes = data.tobytes()

        # round up
        num_blocks = -int(-len(data_bytes) // max_block_size)
        last_block_size = len(data_bytes) - (num_blocks - 1) * max_block_size

        compressed_blocks = [
            zlib.compress(block)
            for block in self._chunk_it(data_bytes, max_block_size)
        ]

        # collect header
        header = np.array(
            [num_blocks, max_block_size, last_block_size]
            + [len(b) for b in compressed_blocks],
            dtype=np.uint32,
        )
        writeln(
            base64.b64encode(header.tobytes()).decode()
            + base64.b64encode(b"".join(compressed_blocks)).decode()
        )

    def write_data_array(self, writeln, name, data, num_cmpts):
        data_type = {
            np.dtype("float64"): "Float64",
            np.dtype("int64"): "Int64",
            np.dtype("uint8"): "UInt8",
            np.dtype("uint32"): "UInt32",
        }
        writeln(rf'<DataArray type="{data_type.get(data.dtype)}"')
        writeln(rf'           format="binary"')
        writeln(rf'           Name="{name}"')
        writeln(rf'           NumberOfComponents="{num_cmpts}">')
        self.text_writer_compressed(writeln, data.ravel())
        writeln(rf"</DataArray>")


class CellData(object):

    def __init__(self, n):
        self.n = n

    def num_cells(self):
        return self.n**2

    def offsets(self):
        return np.cumsum(3 * np.ones(self.n**2, dtype=np.int64))

    def types(self):
        return 5 * np.ones(self.n**2, dtype=np.uint8)

    def connectivity(self):
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

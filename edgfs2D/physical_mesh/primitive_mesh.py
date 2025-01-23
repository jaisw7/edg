import re
from argparse import Namespace

from edgfs2D.time.physical_time import PhysicalTime
from edgfs2D.utils.dictionary import Dictionary

msect = "mesh"


class PrimitiveMesh(object):
    """Defines a primitive mesh"""

    def loadgmsh(self):
        meshfile = self.time.args.mesh
        from edgfs2D.physical_mesh.readers import get_reader_by_name

        reader = get_reader_by_name("gmsh", meshfile)
        xmesh = reader.get_mesh()

        get = lambda prefix: {
            key: xmesh[key]
            for key in filter(lambda v: v.startswith(prefix), xmesh.keys())
        }

        extract = lambda prefix, key: re.match(
            r"{0}_(\w+)_\w+".format(prefix), key
        ).group(1)

        # define vertex
        self._vertex = {
            extract("spt", key): val for key, val in get("spt_").items()
        }

        # import numpy as np
        # import torch

        # p = np.array(
        #     [
        #         [0.0, 0.0],
        #         [1.0, 0.0],
        #         [1.0, 1.0],
        #         [0.0, 1.0],
        #         [0.5, 0.0],
        #         [1.0, 0.5],
        #         [0.5, 1.0],
        #         [0.0, 0.5],
        #         [0.5, 0.5],
        #     ]
        # )
        # vtx = self._vertex["tri"]

        # e2v = np.zeros((8, 3), dtype=int)
        # for e in range(vtx.shape[1]):
        #     v1 = vtx[0, e, :].reshape(-1, 2)
        #     v2 = vtx[1, e, :].reshape(-1, 2)
        #     v3 = vtx[2, e, :].reshape(-1, 2)
        #     # print(p.shape, v1.shape)

        #     tolerance = 1e-6
        #     i1 = np.where(np.all(np.abs(p - v1) < tolerance, axis=1))[0][0]
        #     i2 = np.where(np.all(np.abs(p - v2) < tolerance, axis=1))[0][0]
        #     i3 = np.where(np.all(np.abs(p - v3) < tolerance, axis=1))[0][0]
        #     e2v[e, :] = [i1, i2, i3]
        #     # print(i1, i2, i3)

        # print(torch.from_numpy(e2v))
        # exit(0)

        # define element to element internal connectivity
        self._conn_internal = {
            key: val.astype("U4,i4,i1,i1").tolist()
            for key, val in get("con_").items()
        }

        # define boundary connectivity
        self._conn_bnd = {
            extract("bcon", key): val.astype("U4,i4,i1,i1").tolist()
            for key, val in get("bcon_").items()
        }

        # define element shapes in the domain
        self._ele_shapes = self._vertex.keys()

    _meshTypes = {"gmsh": loadgmsh}

    def __init__(self, global_cfg: Dictionary, time: PhysicalTime):
        self.cfg = global_cfg.get_section(msect)
        self.time = time

        mt = self.cfg.lookupordefault("kind", "gmsh")
        assert mt in self._meshTypes, "Valid mesh:" + str(
            self._meshTypes.keys()
        )
        self._meshTypes[mt](self)

    @property
    def vertex(self):
        return self._vertex

    @property
    def connectivity_internal(self):
        return self._conn_internal

    @property
    def connectivity_boundaries(self):
        return self._conn_bnd

    @property
    def element_shapes(self):
        return self._ele_shapes

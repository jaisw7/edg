import re

from edgfs2D.time.physical_time import PhysicalTime
from edgfs2D.utils.dictionary import Dictionary

msect = "mesh"


class PrimitiveMesh(object):
    """Defines a primitive mesh"""

    def loadgmsh(self):
        meshfile = self._time.args.mesh
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

        # define mesh uuid
        self._uuid = xmesh["mesh_uuid"]

    _meshTypes = {"gmsh": loadgmsh}

    def __init__(self, global_cfg: Dictionary, time: PhysicalTime):
        self._cfg = global_cfg.get_section(msect)
        self._time = time

        mt = self._cfg.lookupordefault("kind", "gmsh")
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

    @property
    def uuid(self):
        return self._uuid

    @property
    def time(self):
        return self._time

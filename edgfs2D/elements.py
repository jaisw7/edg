class Elements(object):
    """Defines elements and their associated metrics"""

    def __init__(self, cfg, mesh):
        self.cfg = cfg
        self.args = args

        mt = cfg.lookupordefault(msect, "kind", "gmsh")
        assert mt in self._meshTypes, "Valid mesh:" + str(self._meshTypes.keys())

        self._meshType = self._meshTypes[mt]

        # define 1D mesh (We just need a sorted point distribution)
        xmesh = self._meshType(self)

        # non-dimensionalize the domain
        for key in filter(lambda v: v.startswith("spt_"), xmesh.keys()):
            xmesh[key] /= nondim.H0

        self._xmesh = xmesh

    @property
    def xmesh(self):
        return self._xmesh

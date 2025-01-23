import numpy as np
from typing_extensions import override

from edgfs2D.fields.dgfield import DgField
from edgfs2D.fields.types import FieldData
from edgfs2D.physical_mesh.dg_mesh import DgMesh
from edgfs2D.solvers.advection.create_boundary_conditions import (
    get_boundary_condition,
)
from edgfs2D.solvers.advection.create_flux import get_flux
from edgfs2D.solvers.advection.create_initial_conditions import (
    get_initial_condition,
)
from edgfs2D.utils.dictionary import Dictionary


class AdvField(DgField):
    nfields = 1
    fields = ["u"]
    kind = "advection"

    def __init__(self, cfg: Dictionary, dgmesh: DgMesh):
        super().__init__(cfg, dgmesh, self.nfields)

        # define boundary conditions
        self._bnds: Dict[str, BaseBoundaryCondition] = {}
        for kind, x in self.dgmesh.get_boundary_nodes.items():
            self._bnds[kind] = get_boundary_condition(
                self.cfg, "{}-{}".format(self.kind, kind), x
            )

        # define flux
        self._flux = get_flux(self.cfg, self.kind)
        self._velocity = self._flux.velocity

    def apply_initial_condition(self, u: FieldData):
        ic = get_initial_condition(self.cfg, self.kind)
        super()._apply_initial_condition(u, ic)

    def apply_boundary_condition(self, uf: FieldData):
        super()._apply_boundary_condition(uf, self._bnds)

    def compute_flux(self, ul: FieldData, ur: FieldData):
        super()._compute_flux(ul, ur, self._flux)

    def convect(self, gradu: FieldData) -> FieldData:
        return super()._convect(gradu, self._velocity)

    def lift_jump(
        self, fl: FieldData, fr: FieldData, uf: FieldData, out: FieldData
    ):
        return super()._lift_jump(fl, fr, uf, self._velocity, out)

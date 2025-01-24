import numpy as np
import torch
from typing_extensions import override

from edgfs2D.fields.dgfield import DgField
from edgfs2D.fields.types import FieldData, FieldDataTuple
from edgfs2D.physical_mesh.dg_mesh import DgMesh
from edgfs2D.solvers.fast_spectral.create_boundary_conditions import (
    get_boundary_condition,
)
from edgfs2D.solvers.fast_spectral.create_flux import get_flux
from edgfs2D.solvers.fast_spectral.create_initial_conditions import (
    get_initial_condition,
)
from edgfs2D.utils.dictionary import Dictionary


class FsField(DgField):
    nfields = 1
    fields = ["f"]
    kind = "fast-spectral"

    def __init__(self, cfg: Dictionary, dgmesh: DgMesh):
        super().__init__(cfg, dgmesh, self.nfields)

        # define boundary conditions
        self._bnds: Dict[str, BaseBoundaryCondition] = {}
        for kind, _ in self.dgmesh.get_boundary_interfaces.items():
            self._bnds[kind] = get_boundary_condition(
                self.cfg, "{}-{}".format(self.kind, kind)
            )

        # define flux
        self._flux = get_flux(self.cfg, self.kind)
        self._velocity = self._flux.velocity

    def apply_initial_condition(self, u: FieldData):
        ic = get_initial_condition(self.cfg, self.kind)
        super()._apply_initial_condition(u, ic)

    def internal_traces(self, uf: FieldData) -> FieldDataTuple:
        return super()._internal_traces(uf)

    def boundary_traces(
        self, curr_time: torch.float64, uf: FieldData
    ) -> FieldDataTuple:
        return super()._boundary_traces(curr_time, uf, self._bnds)

    def compute_internal_flux(self, ul: FieldData, ur: FieldData):
        super()._compute_flux(
            ul, ur, self._internal_interface_normals[0], self._flux
        )

    def compute_boundary_flux(self, ul: FieldData, ur: FieldData):
        super()._compute_flux(
            ul, ur, self._boundary_interface_normals, self._flux
        )

    def convect(self, gradu: FieldData) -> FieldData:
        return super()._convect(gradu, self._velocity)

    def lift_jump(
        self,
        fl: FieldData,
        fr: FieldData,
        fb: FieldData,
        uf: FieldData,
        out: FieldData,
    ):
        return super()._lift_jump(fl, fr, fb, uf, self._velocity, out)

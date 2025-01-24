import numpy as np
import torch

from edgfs2D.fields.dgfield import DgField
from edgfs2D.fields.types import FieldData, FieldDataList
from edgfs2D.integrators.base import BaseIntegrator
from edgfs2D.physical_mesh.dg_mesh import DgMesh
from edgfs2D.solvers.advection.create_fields import FsField
from edgfs2D.solvers.base import BaseSolver
from edgfs2D.time.physical_time import PhysicalTime
from edgfs2D.utils.dictionary import Dictionary


class FastSpectralSolver(BaseSolver):

    def __init__(
        self,
        cfg: Dictionary,
        time: PhysicalTime,
        dgmesh: DgMesh,
        intg: BaseIntegrator,
    ):
        self._time = time
        self._dgmesh = dgmesh
        self._intg = intg
        self._fs = FsField(cfg, dgmesh)

        # load plugins
        self._time.load_plugins(self)

        # tensors for storage of solutions
        self._u0 = self._fs.create_new_field()
        self._u1 = self._fs.create_new_field()

        # apply initial condition
        self._fs.apply_initial_condition(self._u1)

    def rhs(self, curr_time: torch.float64, u: FieldData):
        fs = self._fs

        # compute convective derivative
        gradu = fs.grad(u)
        conv = fs.convect(gradu)
        conv.mul_(-1)

        # traces at boundaries
        uf = fs.surface_data(u)
        ul, ur = fs.internal_traces(uf)
        ulb, urb = fs.boundary_traces(curr_time, uf)

        # fluxes
        fs.compute_internal_flux(ul, ur)
        fs.compute_boundary_flux(ulb, urb)

        # lift
        fs.lift_jump(ul, ur, ulb, uf, conv)

        return conv

    def solve(self):
        intg = self._intg
        time = self._time

        u_ex = self._u1.clone()

        while time.run():
            self._u0.copy_(self._u1)

            for i in range(intg.num_steps):
                u_new = intg.integrate_at_step(i, time, self._u1, self.rhs)
                self._u1.copy_(u_new)

            time.increment()

    @property
    def curr_fields(self) -> FieldDataList:
        return [self._u1]

    @property
    def prev_fields(self) -> FieldDataList:
        return [self._u0]

    @property
    def time(self) -> PhysicalTime:
        return self._time

    @property
    def mesh(self) -> DgMesh:
        return self._dgmesh

    def error_norm(self, err: FieldData):
        return self._fs.error(err)

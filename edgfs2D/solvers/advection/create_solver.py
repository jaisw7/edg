from pathlib import Path

import torch
from typing_extensions import override

from edgfs2D.fields.types import FieldData, FieldDataList
from edgfs2D.integrators.base import BaseIntegrator
from edgfs2D.physical_mesh.dg_mesh import DgMesh
from edgfs2D.solvers.advection.create_fields import AdvField
from edgfs2D.solvers.base import BaseSolver
from edgfs2D.time.physical_time import PhysicalTime
from edgfs2D.utils.dictionary import Dictionary


class AdvSolver(BaseSolver):
    field_name = "u"

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
        self._advf = AdvField(cfg, dgmesh)

        # tensors for storage of solutions
        self._u0 = self._advf.create_new_field()
        self._u1 = self._advf.create_new_field()

        # apply initial condition
        if time.is_restart:
            self._u1 = self._advf.read_field(time.args.soln, self.field_name)
        else:
            self._advf.apply_initial_condition(self._u1)

        # load plugins
        self._time.load_plugins(self)

    def eval_derivate(self, u: FieldData):
        advf = self._advf

        if advf.is_eflux_enabled:
            eflux = advf.compute_entropy_flux(u, u)
            conv = advf.convect_eflux(eflux)
        else:
            gradu = advf.grad(u)
            conv = advf.convect(gradu)

        return conv

    def rhs(self, curr_time: torch.float64, u: FieldData):
        advf = self._advf

        # compute convective derivative
        conv = self.eval_derivate(u)
        conv.mul_(-1)

        # traces at boundaries
        uf = advf.surface_data(u)
        ul, ur = advf.internal_traces(uf)
        ulb, urb = advf.boundary_traces(curr_time, uf)

        # fluxes
        advf.compute_internal_flux(ul, ur)
        advf.compute_boundary_flux(ulb, urb)

        # lift
        advf.lift_jump(ul, ur, ulb, uf, conv)

        return conv

    def solve(self):
        intg = self._intg
        time = self._time

        while time.run():
            self._u0.copy_(self._u1)

            for i in range(intg.num_steps):
                u_new = intg.integrate_at_step(i, time, self._u1, self.rhs)
                self._u1.copy_(u_new)

            time.increment()

    @override
    @property
    def curr_fields(self) -> FieldDataList:
        return [self._u1]

    @override
    @property
    def prev_fields(self) -> FieldDataList:
        return [self._u0]

    @override
    @property
    def time(self) -> PhysicalTime:
        return self._time

    @override
    @property
    def mesh(self) -> DgMesh:
        return self._dgmesh

    @override
    def error_norm(self, err: FieldData):
        return self._advf.error(err)

    @override
    def write(self, path: Path):
        writer = self._advf.write_metadata(path)
        writer.write_fields(FieldData({self.field_name: self.curr_fields[0]}))

import numpy as np
import torch

from edgfs2D.fields.dgfield import DgField
from edgfs2D.fields.types import FieldData, FieldDataList
from edgfs2D.integrators.base import BaseIntegrator
from edgfs2D.physical_mesh.dg_mesh import DgMesh
from edgfs2D.solvers.advection.create_fields import AdvField
from edgfs2D.solvers.base import BaseSolver
from edgfs2D.time.physical_time import PhysicalTime
from edgfs2D.utils.dictionary import Dictionary


class AdvSolver(BaseSolver):

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

        # load plugins
        self._time.load_plugins(self)

        # tensors for storage of solutions
        self._u0 = self._advf.create_new_field()
        self._u1 = self._advf.create_new_field()

        # apply initial condition
        self._advf.apply_initial_condition(self._u1)

    def rhs(self, curr_time: torch.float64, u: FieldData):
        advf = self._advf

        # compute convective derivative
        gradu = advf.grad(u)
        conv = advf.convect(gradu)
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

        u_ex = self._u1.clone()

        while time.run():
            self._u0.copy_(self._u1)

            for i in range(intg.num_steps):
                u_new = intg.integrate_at_step(i, time, self._u1, self.rhs)
                self._u1.copy_(u_new)

            time.increment()

        x = torch.from_numpy(self._dgmesh._element_nodes["tri"])
        # u_ex["tri"] = (
        #     torch.sin(torch.pi * (x[..., 0] - time.time))
        #     * torch.cos(torch.pi * (x[..., 1] - time.time))
        # ).unsqueeze(-1)

        err = (self._u1["tri"].squeeze() - u_ex["tri"].squeeze()).numpy()
        M = np.linalg.inv(self._dgmesh.get_basis_at_shapes["tri"]._iH)
        error = (
            np.sqrt(
                np.tensordot(
                    self._dgmesh.get_element_jacobian_det["tri"],
                    np.matmul(M, err**2),
                )
            )
            / 4
        )
        print(error)

    @property
    def curr_fields(self) -> FieldDataList:
        return [self._u1]

    @property
    def prev_fields(self) -> FieldDataList:
        return [self._u0]

    @property
    def time(self) -> PhysicalTime:
        return self._time

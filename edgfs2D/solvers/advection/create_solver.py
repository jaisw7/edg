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

# np.set_printoptions(suppress=True, linewidth=2000, precision=3)


pe = lambda v: print(v[next(iter(v.keys()))][..., 0].numpy())
pex = lambda v: pe(v) + exit(0)


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
        # print(u["tri"][..., 0].numpy()) + exit(0)

        # compute convective derivative
        gradu = advf.grad(u)
        conv = advf.convect(gradu)
        conv.mul_(-1)

        # print(conv["tri"][..., 0].numpy()) + exit(0)

        # pex(u)
        # pe(u)
        uf = advf.surface_data(u)
        # pe(uf)
        # pex(uf)
        ul, ur = advf.traces(uf)
        # pex(ul)
        # pex(ur)
        advf.compute_flux(ul, ur)
        # pe(ul) + pex(ur)
        # pe(uf)
        z = FieldData.zeros_like(conv)
        advf.lift_jump(ul, ur, uf, z)
        # exit(0)
        # pe(ul) + pex(ur)
        # pex(z)
        conv.add_(z)

        # print(z["tri"][..., 0].numpy()) + exit(0)
        # print(conv["tri"][..., 0].numpy()) + exit(0)
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

            # if time.step == 1:
            # print(time.time)
            # print(self._u1["tri"][..., 0].numpy())
            # exit(0)
            time.increment()

            # t = time.time
            # x = self._advf._element_nodes["tri"]
            # ux = (
            #     torch.sin(torch.pi * (x[..., 0] - t))
            #     * torch.cos(torch.pi * (x[..., 1] - t))
            # ).unsqueeze(-1)
            # print(t, "norm: ", torch.dist(self._u1["tri"], ux).item())

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

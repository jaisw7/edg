from functools import cached_property

import torch

from edgfs2D.fields.types import FieldData
from edgfs2D.solvers.fast_spectral.formulations.base import BaseFormulation


class ImexFastSpectralSolver(BaseFormulation):
    formulation = "imex"
    num_moment_fields = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # tensors for storage of moments
        self._m0 = self._fs.create_new_field(self.num_moment_fields)
        self._m1 = self._m0.clone()

    @cached_property
    def moments_op(self):
        vpoints = self._distmesh.vmesh.points
        magv = torch.sum(vpoints**2, dim=0)
        out = torch.vstack((torch.ones_like(magv), vpoints, magv)).T
        out.mul_(self._distmesh.vmesh.weights)
        return out

    def moments(self, u: FieldData, out: FieldData):
        for shape in out.keys():
            torch.matmul(u[shape], self.moments_op, out=out[shape])

    def construct_maxwellian(self, m: torch.Tensor, out: torch.Tensor):
        vpoints = self._distmesh.vmesh.points
        rho = m[..., 0]
        u = m[..., 1:4] / rho.unsqueeze(-1)
        T = (m[..., 4] - rho * (u**2).sum(dim=-1)).div_(rho).div_(1.5)
        out.copy_(
            ((vpoints.unsqueeze(0).unsqueeze(0) - u.unsqueeze(-1)) ** 2)
            .sum(dim=-2)
            .mul_(-1)
        )
        out.div_(T.unsqueeze(-1)).exp_()
        out.mul_(rho.unsqueeze(-1))
        out.div_(torch.pi**1.5).div_(T.unsqueeze(-1) ** 1.5)

    def solve(self):
        time = self._time
        nu = self._fs.create_new_field(1)
        u0, u1 = self._u0, self._u1
        m0, m1 = self._m0, self._m1
        M = u1.clone()

        # compute initial moments
        self.moments(u1, m1)

        # compute maxwellian from initial moments
        for shape in m1.keys():
            self.construct_maxwellian(m1[shape], M[shape])

        while time.run():
            tlocal, dt = time.time, time.dt
            u0.copy_(u1)
            m0.copy_(m1)

            # compute transport = -T(u0)
            tpt = self.transport(tlocal, u0)

            # compute updated moments = m0 + dt * m1
            self.moments(tpt, m1)
            for shape in m1.keys():
                m1[shape].mul_(dt).add_(m0[shape])

            # add scattering = -T(u0) + Q(u0,u0)/Kn
            for shape in m1.keys():
                self._sm.solve(tlocal, u0[shape], tpt[shape], nu[shape])

            # add penalty = -T(u0) + Q(u0,u0)/Kn - nu*(M(m0) - u0)/Kn
            for shape in m1.keys():
                M[shape].sub_(u0[shape]).mul_(nu[shape]).mul_(-1)
                tpt[shape].add_(M[shape])

            # scale nu by time-step
            for shape in m1.keys():
                nu[shape].mul_(dt)

            # compute maxwellian from updated moments
            for shape in m1.keys():
                # u1 = (nu*dt/Kn) * M(m1)
                self.construct_maxwellian(m1[shape], M[shape])
                torch.mul(M[shape], nu[shape], out=u1[shape])

            # update
            for shape in m1.keys():
                tpt[shape].mul_(dt)
                u1[shape].add_(u0[shape]).add_(tpt[shape])
                u1[shape].div_(1 + nu[shape])

            time.increment()

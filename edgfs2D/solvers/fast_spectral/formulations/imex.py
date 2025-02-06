import torch

from edgfs2D.scattering import get_relaxation_model
from edgfs2D.solvers.fast_spectral.formulations.base import BaseFormulation


class ImexFastSpectralSolver(BaseFormulation):
    formulation = "imex"
    num_moment_fields = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # relaxation model
        self._rm = get_relaxation_model(self._cfg, self._distmesh)

        # tensors for storage of moments
        self._m0 = self._fs.create_new_field(self._rm.num_fields)
        self._m1 = self._m0.clone()

    def solve(self):
        time = self._time
        nu = self._fs.create_new_field(1)
        u0, u1 = self._u0, self._u1
        m0, m1 = self._m0, self._m1
        M = u1.clone()

        # compute initial moments
        self._rm.moments(u1, m1)

        # compute maxwellian from initial moments
        self._rm.construct_distribution(m1, M, nu)

        while time.run():
            tlocal, dt = time.time, time.dt
            u0.copy_(u1)
            m0.copy_(m1)

            # compute transport = -T(u0)
            tpt = self.transport(tlocal, u0)

            # compute updated moments = m0 + dt * m1
            self._rm.moments(tpt, m1)
            for shape in m1.keys():
                m1[shape].mul_(dt).add_(m0[shape])

            # add scattering = -T(u0) + Q(u0,u0)/Kn
            for shape in m1.keys():
                self._sm.solve(tlocal, u0[shape], tpt[shape])

            # add penalty = -T(u0) + Q(u0,u0)/Kn - nu*(M(m0) - u0)/Kn
            for shape in m1.keys():
                M[shape].sub_(u0[shape]).mul_(nu[shape]).mul_(-1)
                tpt[shape].add_(M[shape])

            # construct updated maxwellian
            self._rm.construct_distribution(m1, M, nu)

            # compute maxwellian from updated moments
            for shape in m1.keys():
                # u1 = (nu*dt/Kn) * M(m1)
                torch.mul(M[shape], nu[shape].mul(dt), out=u1[shape])

            # update
            for shape in m1.keys():
                tpt[shape].mul_(dt)
                u1[shape].add_(u0[shape]).add_(tpt[shape])
                u1[shape].div_(1 + nu[shape].mul(dt))

            time.increment()

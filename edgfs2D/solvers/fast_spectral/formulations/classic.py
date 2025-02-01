from typing_extensions import override

from edgfs2D.distribution_mesh.dgdist_mesh import DgDistMesh
from edgfs2D.integrators import get_integrator
from edgfs2D.solvers.fast_spectral.formulations.base import BaseFormulation
from edgfs2D.utils.dictionary import Dictionary


class ClassicFastSpectralSolver(BaseFormulation):
    field_name = "u"
    formulation = "classic"

    def __init__(self, cfg: Dictionary, distmesh: DgDistMesh):
        super().__init__(cfg, distmesh)
        self._intg = get_integrator(cfg)

    @override
    def solve(self):
        intg = self._intg
        time = self._time
        u0, u1 = self._u0, self._u1

        while time.run():
            u0.copy_(u1)

            for i in range(intg.num_steps):
                u1.copy_(intg.integrate_at_step(i, time, u1, self.rhs))

            time.increment()

from edgfs2D.fields.types import FieldData
from edgfs2D.integrators.base import BaseIntegrator, RhsFunction
from edgfs2D.time.physical_time import PhysicalTime
from edgfs2D.utils.dictionary import SubDictionary


class EulerIntegrator(BaseIntegrator):
    kind = "euler"

    def __init__(self, cfg: SubDictionary, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._cfg = cfg

    @property
    def order(self):
        return 1

    @property
    def num_steps(self):
        return 1

    def integrate_at_step(
        self, step: int, time: PhysicalTime, u: FieldData, rhs: RhsFunction
    ) -> FieldData:
        rhs_val = rhs(time.time, u)
        u_new = FieldData()

        for shape in u.keys():
            u_new[shape] = u[shape] + time.dt * rhs_val[shape]

        return u_new

from edgfs2D.fields.types import FieldData
from edgfs2D.integrators.base import BaseIntegrator, RhsFunction
from edgfs2D.time.physical_time import PhysicalTime
from edgfs2D.utils.dictionary import SubDictionary

rk4a = [
    0.0,
    -567301805773.0 / 1357537059087.0,
    -2404267990393.0 / 2016746695238.0,
    -3550918686646.0 / 2091501179385.0,
    -1275806237668.0 / 842570457699.0,
]

rk4b = [
    1432997174477.0 / 9575080441755.0,
    5161836677717.0 / 13612068292357.0,
    1720146321549.0 / 2090206949498.0,
    3134564353537.0 / 4481467310338.0,
    2277821191437.0 / 14882151754819.0,
]

rk4c = [
    0.0,
    1432997174477.0 / 9575080441755.0,
    2526269341429.0 / 6820363962896.0,
    2006345519317.0 / 3224310063776.0,
    2802321613138.0 / 2924317926251.0,
]


class LserkIntegrator(BaseIntegrator):
    kind = "lserk"

    def __init__(self, cfg: SubDictionary, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._cfg = cfg
        self._uprev = None

    @property
    def order(self):
        return 4

    @property
    def num_steps(self):
        return 5

    def u_prev(self, u: FieldData):
        if self._uprev:
            return self._uprev
        self._uprev = u.clone_as_zeros()
        return self._uprev

    def integrate_at_step(
        self, step: int, time: PhysicalTime, u: FieldData, rhs: RhsFunction
    ) -> FieldData:
        u_prev = self.u_prev(u)
        tlocal = time.time + rk4c[step] * time.dt

        rhs_val = rhs(tlocal, u)
        u_new = FieldData()

        for shape in u.keys():
            u_prev[shape].mul_(rk4a[step]).add_(rhs_val[shape], alpha=time.dt)
            u_new[shape] = u[shape] + rk4b[step] * u_prev[shape]

        return u_new

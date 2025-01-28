from functools import cached_property

import numpy as np
from loguru import logger
from typing_extensions import override

from edgfs2D.initial_conditions import get_ic_by_cls_and_name
from edgfs2D.initial_conditions.base import BaseInitialCondition
from edgfs2D.utils.dictionary import SubDictionary
from edgfs2D.utils.util import to_torch_device
from edgfs2D.velocity_mesh.base import BaseVelocityMesh


class FastSpectralInitialCondition(BaseInitialCondition):
    kind = None

    def __init__(
        self, cfg: SubDictionary, vmesh: BaseVelocityMesh, *args, **kwargs
    ):
        super().__init__(cfg, *args, **kwargs)
        self._vmesh = vmesh

    @property
    def vmesh(self):
        return self._vmesh


class MaxwellianInitialCondition(FastSpectralInitialCondition):
    kind = "fast-spectral-maxwellian"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._rhoini = 1.0
        if not ("read_rho" in kwargs):
            self._rhoini = self._cfg.lookupfloat("rho") / self.vmesh.nondim.rho0

        self._Tini = self._cfg.lookupfloat("T") / self.vmesh.nondim.T0
        self._uini = self._cfg.lookupfloat_list("u") / self.vmesh.nondim.u0
        self._uini = self._uini.reshape((3, 1))
        self._perform_precomputation()

    @cached_property
    def init_vals(self):
        return to_torch_device(self._init_vals, self._cfg)

    @cached_property
    def u(self):
        return to_torch_device(self._uini, self._cfg)

    def _perform_precomputation(self):
        vm = self.vmesh

        # define maxwellian
        self._init_vals = (self._rhoini / (np.pi * self._Tini) ** 1.5) * np.exp(
            -np.sum((vm.points - self._uini) ** 2, axis=0) / self._Tini
        )

        # test the distribution support
        cv = self.vmesh.points
        vsize = self.vmesh.num_points
        cw = self.vmesh.weights
        T0 = self.vmesh.nondim.T0
        rho0 = self.vmesh.nondim.rho0
        molarMass0 = self.vmesh.nondim.molarMass0
        u0 = self.vmesh.nondim.u0
        mr = molarMass0 / molarMass0
        mcw = mr * cw

        ele_sol = np.zeros(5)
        soln = self._init_vals

        # non-dimensional mass density
        ele_sol[0] = np.sum(soln) * mcw

        # non-dimensional velocities
        ele_sol[1] = np.tensordot(soln, cv[0, :], axes=(0, 0)) * mcw
        ele_sol[1] /= ele_sol[0]
        ele_sol[2] = np.tensordot(soln, cv[1, :], axes=(0, 0)) * mcw
        ele_sol[2] /= ele_sol[0]
        ele_sol[3] = np.tensordot(soln, cv[2, :], axes=(0, 0)) * mcw
        ele_sol[3] /= ele_sol[0]

        # peculiar velocity
        cx = cv[0, :] - ele_sol[1]
        cy = cv[1, :] - ele_sol[2]
        cz = cv[2, :] - ele_sol[3]
        cSqr = cx * cx + cy * cy + cz * cz

        # non-dimensional temperature
        ele_sol[4] = np.sum(soln * cSqr, axis=0) * (2.0 / 3.0 * mcw * mr)
        ele_sol[4] /= ele_sol[0]

        if not (
            np.allclose(self._rhoini * rho0, ele_sol[0] * rho0, atol=1e-5)
            and np.allclose(self._uini[0, 0] * u0, ele_sol[1] * u0, atol=1e-5)
            and np.allclose(self._uini[1, 0] * u0, ele_sol[2] * u0, atol=1e-5)
            and np.allclose(self._uini[2, 0] * u0, ele_sol[3] * u0, atol=1e-5)
            and np.allclose(self._Tini * T0, ele_sol[4] * T0, atol=1e-5)
        ):
            logger.info("Bulk properties not conserved!")
            logger.info("Calculated bulk properties based on provided inputs:")
            logger.info("density {}", self._rhoini * rho0, ele_sol[0] * rho0)
            logger.info("x-velocity {}", self._uini[0, 0] * u0, ele_sol[1] * u0)
            logger.info("y-velocity {}", self._uini[1, 0] * u0, ele_sol[2] * u0)
            logger.info("z-velocity {}", self._uini[2, 0] * u0, ele_sol[3] * u0)
            logger.info("temperature: {}", self._Tini * T0, ele_sol[4] * T0)
            raise ValueError("Check velocity mesh parameters")

    @override
    def apply(self, x: np.array):
        return np.broadcast_to(
            self._init_vals, (*x.shape[:2], len(self._init_vals.squeeze()))
        ).copy()


def get_initial_condition(cfg, name, *args, **kwargs):
    return get_ic_by_cls_and_name(
        cfg, name, FastSpectralInitialCondition, *args, **kwargs
    )

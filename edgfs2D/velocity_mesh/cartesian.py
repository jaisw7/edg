from functools import cached_property

import numpy as np
from loguru import logger
from typing_extensions import override

from edgfs2D.utils.util import to_torch_device
from edgfs2D.velocity_mesh.base import BaseVelocityMesh


class Cartesian(BaseVelocityMesh):
    kind = "cartesian"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._construct_velocity_mesh()

    def _construct_velocity_mesh(self):
        # define the velocity mesh
        self._Nv = self._cfg.lookupint("Nv")
        self._vsize = self._Nv**3

        _cmax = self._cfg.lookupfloat("cmax")
        _Tmax = self._cfg.lookupfloat("Tmax")
        _dev = self._cfg.lookupfloat("dev")

        # normalize maximum bulk velocity
        _cmax /= self._nondim.u0

        # normalize maximum bulk temperature
        _Tmax /= self._nondim.T0

        # define the length of the velocity mesh
        self._L = _cmax + _dev * np.sqrt(_Tmax)
        self._S = self._L * 2.0 / (3.0 + np.sqrt(2.0))
        self._R = 2 * self._S
        logger.info("velocityMesh: ({} {})", -self._L, self._L)

        self._dev = _dev

        # define the weight of the velocity mesh
        self._cw = (2.0 * self._L / self._Nv) ** 3
        c0 = np.linspace(
            -self._L + self._L / self._Nv,
            self._L - self._L / self._Nv,
            self._Nv,
        )
        self._cv = np.zeros((3, self._vsize), dtype=self._cfg.dtype)
        for l in range(self._vsize):
            I = int(l / (self._Nv * self._Nv))
            J = int((l % (self._Nv * self._Nv)) / self._Nv)
            K = int((l % (self._Nv * self._Nv)) % self._Nv)
            self._cv[0, l] = c0[I]
            self._cv[1, l] = c0[J]
            self._cv[2, l] = c0[K]

    @override
    @property
    def num_points(self):
        return self._vsize

    @override
    @property
    def extents(self):
        return self._L

    @override
    @property
    def shape(self):
        return (self._Nv, self._Nv, self._Nv)

    @override
    @cached_property
    def points(self):
        return to_torch_device(self._cv, self._cfg)

    @override
    @property
    def weights(self):
        return self._cw

import numpy as np

from edgfs2D.std.velocity_mesh.base import BaseVelocityMesh


class Cartesian(BaseVelocityMesh):
    kind = "cartesian"

    def __init__(self, cfg, name, *args, **kwargs):
        super().__init__(cfg, name, *args, **kwargs)
        self._construct_velocity_mesh(self.nondim)

    def _construct_velocity_mesh(self, nondim):
        sect = self.sect

        # define the velocity mesh
        self._Nv = self.cfg.lookupint(sect, "Nv")
        self._vsize = self._Nv**3

        _cmax = self.cfg.lookupfloat(sect, "cmax")
        _Tmax = self.cfg.lookupfloat(sect, "Tmax")
        _dev = self.cfg.lookupfloat(sect, "dev")

        # normalize maximum bulk velocity
        _cmax /= nondim.u0

        # normalize maximum bulk temperature
        _Tmax /= nondim.T0

        # define the length of the velocity mesh
        self._L = _cmax + _dev * np.sqrt(_Tmax)
        self._S = self._L * 2.0 / (3.0 + np.sqrt(2.0))
        self._R = 2 * self._S
        print("velocityMesh: (%s %s)" % (-self._L, self._L))

        self._dev = _dev

        # define the weight of the velocity mesh
        self._cw = (2.0 * self._L / self._Nv) ** 3
        c0 = np.linspace(
            -self._L + self._L / self._Nv, self._L - self._L / self._Nv, self._Nv
        )
        self._cv = np.zeros((3, self._vsize), dtype=self.cfg.dtype)
        for l in range(self._vsize):
            I = int(l / (self._Nv * self._Nv))
            J = int((l % (self._Nv * self._Nv)) / self._Nv)
            K = int((l % (self._Nv * self._Nv)) % self._Nv)
            self._cv[0, l] = c0[I]
            self._cv[1, l] = c0[J]
            self._cv[2, l] = c0[K]

    def Nv(self):
        return self._vsize

    def L(self):
        return self._L

    def shape(self):
        return (self._Nv, self._Nv, self._Nv)

    def points(self):
        return self._cv

    def weights(self):
        return self._cw

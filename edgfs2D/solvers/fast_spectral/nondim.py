# -*- coding: utf-8 -*-
import numpy as np

from edgfs2D.utils.dictionary import Dictionary

nondim_sect = "non-dim"


class NondimParams(object):
    R0 = 8.3144598
    NA = 6.0221409e23

    def __init__(self, cfg: Dictionary):
        # read the non-dimensional variables (length, temperature, density)
        self._H0 = cfg.lookupfloat(nondim_sect, "H0")  # meters
        self._T0 = cfg.lookupfloat(nondim_sect, "T0")  # K
        self._rho0 = cfg.lookupfloat(nondim_sect, "rho0")  # kg/m^3
        self._molarMass0 = cfg.lookupfloat(nondim_sect, "molarMass0")  # kg/mol
        self._n0 = self._rho0 / self._molarMass0 * self.NA
        self._u0 = np.sqrt(2 * self.R0 / self._molarMass0 * self._T0)
        print("n0, u0, t0: (%s %s %s)" % (self._n0, self._u0, self._H0 / self._u0))

    @property
    def H0(self):
        return self._H0

    @property
    def T0(self):
        return self._T0

    @property
    def rho0(self):
        return self._rho0

    @property
    def molarMass0(self):
        return self._molarMass0

    @property
    def n0(self):
        return self._n0

    @property
    def u0(self):
        return self._u0

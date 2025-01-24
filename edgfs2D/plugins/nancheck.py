# -*- coding: utf-8 -*-

import torch

from edgfs2D.plugins.base import BasePlugin
from edgfs2D.solvers.base import BaseSolver
from edgfs2D.utils.dictionary import SubDictionary


class NanCheckPlugin(BasePlugin):
    kind = "nancheck"
    allowed_solvers = ["AdvSolver"]

    """Check for solution blowup"""

    def __init__(self, cfg: SubDictionary, solver: BaseSolver):
        super().__init__(cfg, solver)
        self._nsteps = super().get_nsteps()

    def __call__(self):
        time = self._solver.time
        solver = self._solver

        if time.should_output(self._nsteps) and any(
            any(torch.isnan(f).any() for _, f in field.items())
            for field in solver.curr_fields
        ):
            raise RuntimeError("NaNs detected at t = {0}".format(time.time))

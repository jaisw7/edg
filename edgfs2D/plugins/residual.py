# -*- coding: utf-8 -*-

import torch

from edgfs2D.plugins.base import BasePlugin
from edgfs2D.solvers.base import BaseSolver
from edgfs2D.utils.dictionary import SubDictionary


class ResidualPlugin(BasePlugin):
    kind = "residual"
    allowed_solvers = ["AdvSolver"]

    """Write norm"""

    def __init__(self, cfg: SubDictionary, solver: BaseSolver):
        super().__init__(cfg, solver)
        self._nsteps = super().get_nsteps()

    def __call__(self):
        time = self._solver.time
        solver = self._solver

        if time.should_output(self._nsteps):

            print(f"residual at t = {time.time: 0.6g}")
            for i, (curr, prev) in enumerate(
                zip(solver.curr_fields, solver.prev_fields)
            ):
                print(
                    f"field: {i}, norm: ",
                    sum(
                        torch.dist(c, p)
                        for c, p in zip(curr.values(), prev.values())
                    ).item(),
                )

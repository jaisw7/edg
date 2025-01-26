# -*- coding: utf-8 -*-

from pathlib import Path

import torch
from loguru import logger

from edgfs2D.plugins.base import BasePlugin
from edgfs2D.solvers.base import BaseSolver
from edgfs2D.utils.dictionary import SubDictionary


class SolutionWriterPlugin(BasePlugin):
    kind = "solutionwriter"
    allowed_solvers = ["AdvSolver", "FastSpectralSolver"]

    """Write current solution fields for the solver"""

    def __init__(self, cfg: SubDictionary, solver: BaseSolver):
        super().__init__(cfg, solver)
        self._nsteps = super().get_nsteps()
        self._basedir, self._basename = super().get_basename()

    def __call__(self):
        time = self._solver.time
        solver = self._solver

        if time.should_output(self._nsteps):
            filename = self._basedir.joinpath(
                self._basename.format(t=time.time)
            )
            self._solver.write(filename)
            logger.info(
                f"written solution in directory {self._basedir} at time =",
                f"{time.time: 0.6g}",
            )

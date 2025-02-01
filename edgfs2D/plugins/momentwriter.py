# -*- coding: utf-8 -*-


from loguru import logger

from edgfs2D.plugins.base import BasePlugin
from edgfs2D.solvers.base import BaseSolver
from edgfs2D.utils.dictionary import SubDictionary


class MomentWriterPlugin(BasePlugin):
    kind = "momentwriter"
    allowed_solvers = ["ClassicFastSpectralSolver", "ImexFastSpectralSolver"]

    """Write moments for the solver"""

    def __init__(self, cfg: SubDictionary, solver: BaseSolver):
        super().__init__(cfg, solver)
        self._nsteps = super().get_nsteps()
        self._basedir, self._basename = super().get_basename()
        self.__call__()

    def __call__(self):
        time = self._solver.time

        if time.step == 0 or time.should_output(self._nsteps):
            filename = self._basedir.joinpath(
                self._basename.format(t=time.time)
            )
            self._solver.write_moment(filename)
            logger.info(
                "written moment in directory {} at time {}",
                self._basedir,
                f"{time.time:0.6g}",
            )

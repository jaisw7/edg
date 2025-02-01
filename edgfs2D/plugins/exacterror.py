# -*- coding: utf-8 -*-

from loguru import logger

from edgfs2D.fields.types import FieldData
from edgfs2D.plugins.base import BasePlugin
from edgfs2D.solvers.base import BaseSolver
from edgfs2D.utils.dictionary import SubDictionary
from edgfs2D.utils.nputil import npeval
from edgfs2D.utils.util import to_torch_device


class ExactErrorPlugin(BasePlugin):
    kind = "exacterror"
    allowed_solvers = ["AdvSolver"]

    """Compare with exact solution"""

    def __init__(self, cfg: SubDictionary, solver: BaseSolver):
        super().__init__(cfg, solver)
        self._nsteps = super().get_nsteps()
        self._expr = self._cfg.lookupexpr("exact-solution")

    def _get_exact(self, time) -> FieldData:
        mesh = self._solver.mesh
        exact_sol = FieldData()
        for shape in mesh.get_basis_at_shapes.keys():
            locals = {
                "x": mesh._element_nodes[shape][..., 0].squeeze(),
                "y": mesh._element_nodes[shape][..., 1].squeeze(),
                "t": time,
            }
            exact_sol[shape] = to_torch_device(
                npeval(self._expr, locals), self._cfg
            ).unsqueeze(-1)
        return exact_sol

    def __call__(self):
        time = self._solver.time
        solver = self._solver

        if time.should_output(self._nsteps):

            logger.info(
                "comparing numerical solution with exact solution at time = {}",
                f"{time.time:0.6g}",
            )
            curr = solver.curr_fields[0]
            exact = self._get_exact(time.time)
            exact.sub_(curr)
            error = solver.error_norm(exact)
            logger.info("error: {}", sum(error.values()))

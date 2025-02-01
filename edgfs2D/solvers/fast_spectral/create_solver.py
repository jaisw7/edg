from edgfs2D.solvers.fast_spectral.formulations.base import BaseFormulation
from edgfs2D.solvers.fast_spectral.formulations.classic import (  # noqa
    ClassicFastSpectralSolver,
)
from edgfs2D.solvers.fast_spectral.formulations.imex import (  # noqa
    ImexFastSpectralSolver,
)
from edgfs2D.utils.nputil import subclass_where

solver_sect = "solver"


def get_solver_formulation(cfg, *args, **kwargs):
    formulationKind = cfg.lookupordefault(solver_sect, "formulation", "classic")
    return subclass_where(BaseFormulation, formulation=formulationKind)(
        cfg, *args, **kwargs
    )

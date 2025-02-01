solver_sect = "solver"
from edgfs2D.solvers.fast_spectral.formulations.base import BaseFormulation
from edgfs2D.solvers.fast_spectral.formulations.classic import (
    ClassicFastSpectralSolver,
)
from edgfs2D.solvers.fast_spectral.formulations.imex import (
    ImexFastSpectralSolver,
)
from edgfs2D.utils.nputil import subclass_where


def get_solver_formulation(cfg, *args, **kwargs):
    formulationKind = cfg.lookupordefault(solver_sect, "formulation", "classic")
    return subclass_where(BaseFormulation, formulation=formulationKind)(
        cfg, *args, **kwargs
    )

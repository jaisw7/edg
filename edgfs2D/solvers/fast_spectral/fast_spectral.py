# -*- coding: utf-8 -*-

"""
Entropy Stable discontinous galerkin solver for Boltzmann equation
"""

import numpy as np
import torch

from edgfs2D.distribution_mesh.dgdist_mesh import DgDistMesh
from edgfs2D.initialize import initialize
from edgfs2D.integrators import get_integrator
from edgfs2D.physical_mesh.dg_mesh import DgMesh
from edgfs2D.physical_mesh.nondim_mesh import NondimMesh
from edgfs2D.solvers.fast_spectral.create_solver import FastSpectralSolver
from edgfs2D.solvers.fast_spectral.nondim import NondimParams
from edgfs2D.time.physical_time import PhysicalTime
from edgfs2D.velocity_mesh import get_velocity_mesh


def main():
    # read the inputs
    cfg, args = initialize()

    # define physical time
    time = PhysicalTime(cfg, args)

    # define non-dimensional parameters
    nondim = NondimParams(cfg)

    # define non-dimensional mesh
    nmesh = NondimMesh(cfg, time, nondim)

    # define discontinous galerkin mesh
    dgmesh = DgMesh(cfg, nmesh)

    # define velocity mesh
    vmesh = get_velocity_mesh(cfg, nondim)

    # define distribution mesh
    distmesh = DgDistMesh(dgmesh, vmesh)

    # define integrator
    intg = get_integrator(cfg)

    # create solver
    solver = FastSpectralSolver(cfg, time, distmesh, intg)
    solver.solve()


def __main__():
    main()

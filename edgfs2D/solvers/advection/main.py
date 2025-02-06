# -*- coding: utf-8 -*-

"""
Entropy Stable discontinous galerkin solver for advection equation
"""
from loguru import logger

from edgfs2D.integrators import get_integrator
from edgfs2D.physical_mesh.dg_mesh import DgMesh
from edgfs2D.physical_mesh.primitive_mesh import PrimitiveMesh
from edgfs2D.solvers.advection.create_solver import AdvSolver
from edgfs2D.solvers.initialize import initialize
from edgfs2D.time.physical_time import PhysicalTime


def main():
    logger.add("advection_{time}.log")

    # read the inputs
    cfg, args = initialize()

    # define physical time
    time = PhysicalTime(cfg, args)

    # define primitive mesh
    pmesh = PrimitiveMesh(cfg, time)

    # define DG mesh
    dgmesh = DgMesh(cfg, pmesh)

    # define integrator
    intg = get_integrator(cfg)

    # create solver
    solver = AdvSolver(cfg, time, dgmesh, intg)
    solver.solve()


def __main__():
    main()

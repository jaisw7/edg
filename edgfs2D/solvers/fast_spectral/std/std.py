# -*- coding: utf-8 -*-

"""
Entropy Stable DGFS on curvilinear triangular grids
"""

import torch as t

from edgfs2D.initialize import initialize
from edgfs2D.mesh.dg_mesh import DgMesh
from edgfs2D.mesh.nondim_mesh import NondimMesh
from edgfs2D.std.nondim import NondimParams

# from edgfs2D.basis import get_basis_by_shape
# from edgfs2D.std.velocity_mesh import get_velocity_mesh
# from edgfs2D.std.scattering import get_scattering_model



def main():
    # read the inputs
    cfg, args = initialize()

    # non-dimensionalization parameters
    nondim = NondimParams(cfg)

    # define non-dimensional mesh
    nmesh = NondimMesh(cfg, args, nondim)

    # define DG mesh
    dgmesh = DgMesh(cfg, nmesh)

    # define DG field to hold value of distribution function
    # u = DgScalarField(cfg, dgmesh)

    # define velocity mesh
    # vm = get_velocity_mesh(cfg, nondim)

    # define scattering model
    # sm = get_scattering_model(cfg, nondim, vm)

    # # define time-integration
    # integ = get_integrator(cfg)

    # # define limiter
    # limiter = get_limiter(cfg)

    # define plugins
    # plugins = get_plugins(cfg)

    # # define solver
    # solver = get_solver(cfg, args, mesh, metric, vm, sm, integ, limiter, plugins)
    # solver.solve()


def __main__():
    main()

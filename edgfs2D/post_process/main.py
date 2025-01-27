# -*- coding: utf-8 -*-

"""
Post process solution
"""

from edgfs2D.physical_mesh.dg_mesh import DgMesh
from edgfs2D.physical_mesh.primitive_mesh import PrimitiveMesh
from edgfs2D.post_process import get_post_processor
from edgfs2D.post_process.initialize import initialize
from edgfs2D.time.physical_time import PhysicalTime


def main():
    # read the inputs
    cfg, args = initialize()

    # define physical time
    time = PhysicalTime(cfg, args)

    # define primitive mesh
    pmesh = PrimitiveMesh(cfg, time)

    # define discontinous galerkin mesh
    dgmesh = DgMesh(cfg, pmesh)

    # define post processor
    pp = get_post_processor(args.p, dgmesh)
    pp.execute()


def __main__():
    main()

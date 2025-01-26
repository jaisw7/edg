# -*- coding: utf-8 -*-
from edgfs2D.utils.nputil import subclass_where
from edgfs2D.velocity_mesh.base import BaseVelocityMesh
from edgfs2D.velocity_mesh.cartesian import Cartesian

velocity_mesh_sect = "velocity-mesh"


def get_velocity_mesh_by_name(cfg, name, *args, **kwargs):
    vmKind = cfg.lookup(name, "kind")
    vmSect = cfg.get_section(name)
    return subclass_where(BaseVelocityMesh, kind=vmKind)(
        vmSect, *args, **kwargs
    )


def get_velocity_mesh(cfg, *args, **kwargs):
    return get_velocity_mesh_by_name(cfg, velocity_mesh_sect, *args, **kwargs)


from edgfs2D.physical_mesh.dg_mesh import DgMesh
from edgfs2D.velocity_mesh.base import BaseVelocityMesh


class DgDistMesh:
    """Defines a distribution mesh as a product of physical and velocity mesh"""

    def __init__(self, dgmesh: DgMesh, vmesh: BaseVelocityMesh):
        self._dgmesh = dgmesh
        self._vmesh = vmesh

    @property
    def dgmesh(self):
        return self._dgmesh

    @property
    def vmesh(self):
        return self._vmesh

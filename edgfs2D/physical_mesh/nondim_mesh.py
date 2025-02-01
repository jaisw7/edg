
from edgfs2D.physical_mesh.primitive_mesh import PrimitiveMesh
from edgfs2D.solvers.fast_spectral.nondim import NondimParams
from edgfs2D.time.physical_time import PhysicalTime
from edgfs2D.utils.dictionary import Dictionary


class NondimMesh(PrimitiveMesh):
    """Defines a non-dimensional mesh"""

    def __init__(
        self, global_cfg: Dictionary, time: PhysicalTime, nondim: NondimParams
    ):
        super().__init__(global_cfg, time)

        vertex = self.vertex
        for key, val in vertex.items():
            val /= nondim.H0

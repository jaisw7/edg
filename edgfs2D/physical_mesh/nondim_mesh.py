from argparse import Namespace

from edgfs2D.physical_mesh.primitive_mesh import PrimitiveMesh
from edgfs2D.std.nondim import NondimParams
from edgfs2D.utils.dictionary import Dictionary


class NondimMesh(PrimitiveMesh):
    """Defines a non-dimensional mesh"""

    def __init__(self, cfg: Dictionary, args: Namespace, nondim: NondimParams):
        super().__init__(cfg, args)

        vertex = self.vertex
        for key, val in vertex.items():
            val /= nondim.H0

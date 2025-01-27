import re
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
import torch
from pkg_resources import resource_listdir, resource_string

from edgfs2D.fields.types import FieldData, FieldDataList
from edgfs2D.utils.dictionary import Dictionary


class BaseBasis(object, metaclass=ABCMeta):
    kind = None
    shape = None

    def __init__(self, cfg: Dictionary, name: str, *args, **kwargs):
        self.cfg = cfg

    @abstractproperty
    def degree(self):
        """Degree of basis"""
        pass

    @abstractproperty
    def num_nodes(self):
        """Number of solution points (nodes) on each element"""
        pass

    @abstractproperty
    def quad_nodes(self):
        """Quadrature nodes associated with the basis/shape"""
        pass

    @abstractproperty
    def quad_weights(self):
        """Quadrature weights associated with the basis/shape"""
        pass

    @abstractmethod
    def vertex_to_element_nodes(self, vertex: np.array):
        """Given vertices of element, compute nodes"""
        pass

    @abstractmethod
    def face_nodes(self, element_nodes: np.array):
        """Nodes associated with faces of the given element"""
        pass

    @abstractproperty
    def face_node_ids(self):
        """Id of nodes associated with faces of the given element.
        e.g., if there are three points associated with each face of triangle,
        then id of these face nodes will be [[0,1,2], [3,4,5], [6,7,8]]"""
        pass

    @abstractproperty
    def num_face_nodes(self):
        """Number of face nodes associated with faces"""
        pass

    @abstractmethod
    def element_geometrical_metrics(self, nodes: np.array):
        """Given nodes, compute inverse jacobian matrix, jacobian matrix and its determinant"""
        pass

    @abstractmethod
    def surface_geometrical_metrics(self, nodes: np.array):
        """Given element inverse jacobian matrix, compute jacobian determinant and normals at faces"""
        pass

    @abstractmethod
    def scale_surface_jacobian_det(
        self, surface_jac_det: np.array, element_jac_det: np.array
    ):
        """Given element and surface jacobian determinant, scale surface jacobian determinant"""
        pass

    @abstractmethod
    def grad(
        self, element_data: torch.Tensor, element_jac: torch.Tensor
    ) -> torch.Tensor:
        """Given element_data compute gradient"""
        pass

    @abstractmethod
    def surface_data(self, element_data: torch.Tensor) -> torch.Tensor:
        """Given element_data, compute surface data"""
        pass

    @abstractmethod
    def convect(
        self, grad_element_data: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Given grad(element_data) and velocity, compute convection"""
        pass

    @abstractmethod
    def lift(self, surface_data: torch.Tensor) -> torch.Tensor:
        """Given grad(element_data) and velocity, compute convection"""
        pass

    @abstractmethod
    def error(
        self, element_data: torch.Tensor, element_jac_det: torch.Tensor
    ) -> torch.float64:
        """Given error at element solution points, compute jacobian weighted error"""
        pass

    @abstractmethod
    def interpolation_op(self, nodes: np.ndarray):
        """Compute interpolation operators at the location of given nodes.
        Note: The nodes must lie within the basis element"""
        pass

    @abstractmethod
    def interpolate(self, element_data: np.ndarray, interp_op: np.ndarray):
        """Given element_data and interpolation operator, compute interpolated solition"""
        pass


class StoredBasis(object):
    @classmethod
    def _iter_rules(cls):
        rpaths = getattr(cls, "_rpaths", None)
        if rpaths is None:
            cls._rpaths = rpaths = resource_listdir(__name__, cls.shape_type)

        for path in rpaths:
            m = re.match(r"([a-zA-Z0-9\-~+]+)-p(\d+)\.pb$", path)
            if m:
                yield (path, m.group(1), int(m.group(2)))

    def __init__(self, shape=None, name=None, deg=None):
        if not deg:
            raise ValueError("Must specify deg")

        best = None
        for rpath, rname, rqdeg in self._iter_rules():
            # See if this basis fulfils the required criterion
            if (not name or name == rname) and (not deg or deg == rqdeg):
                best = (rpath, rqdeg)
                break

        if not best:
            raise ValueError("No suitable basis found")

        # Load the rule
        self._rule = resource_string(__name__, "{}/{}".format(shape, best[0]))

    @property
    def rule(self):
        return self._rule


def get_basis_for_shape(shape=None, name=None, deg=None):
    class BasisForShape(StoredBasis):
        shape_type = shape

    return BasisForShape(shape, name, deg).rule

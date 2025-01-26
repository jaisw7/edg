from functools import cached_property

import numpy as np

from edgfs2D.basis import get_basis_by_shape
from edgfs2D.fields.types import FieldData, FieldDataTuple
from edgfs2D.physical_mesh.primitive_mesh import PrimitiveMesh
from edgfs2D.utils.dictionary import Dictionary
from edgfs2D.utils.util import fuzzysort


class DgMesh:
    """Defines a discontinous galerkin mesh"""

    def __init__(self, cfg: Dictionary, pmesh: PrimitiveMesh):
        self.cfg = cfg
        self.pmesh = pmesh

        # define basis
        self._define_basis_at_shapes()

        # define coordinates of solution nodes in every element
        self._define_element_nodes()

        # define geometric metrics for all elements and their faces
        self._define_geometrical_metrics()

        # define connectivity for internal faces
        self._define_connectivity_internal_faces()

        # define connectivity for boundary faces
        self._define_connectivity_boundary_faces()

    @property
    def uuid(self):
        return self.pmesh.uuid

    @property
    def time(self):
        return self.pmesh.time

    @property
    def get_basis_at_shapes(self):
        return self._basis

    @property
    def get_element_nodes(self):
        return self._element_nodes

    @cached_property
    def get_element_data_sizes(self):
        return {shape: x.shape[:2] for shape, x in self._element_nodes.items()}

    @cached_property
    def get_internal_face_data_sizes(self):
        return {shape: x[0].shape[:2] for shape, x in self._nodes_lhs.items()}

    @cached_property
    def get_boundary_face_data_sizes(self):
        return {shape: x[0].shape[:2] for shape, x in self._nodes_bnd.items()}

    @property
    def get_boundary_nodes(self):
        return self._nodes_bnd

    @property
    def get_element_jacobian_mat(self):
        return self._element_jac_mat

    @property
    def get_element_jacobian_det(self):
        return self._element_jac_det

    @property
    def get_internal_interfaces(self) -> FieldDataTuple:
        return (self._node_ids_lhs, self._node_ids_rhs)

    @property
    def get_internal_interface_normals(self) -> FieldData:
        return (self._normals_lhs, self._normals_rhs)

    @property
    def get_internal_interface_scaled_jacobian_det(self) -> FieldData:
        return (self._scaled_jac_det_lhs, None)

    @property
    def get_boundary_interfaces(self) -> FieldData:
        return self._node_ids_bnd

    @property
    def get_boundary_interface_nodes(self) -> FieldData:
        return self._nodes_bnd

    @property
    def get_boundary_interface_normals(self) -> FieldData:
        return self._normals_bnd

    @property
    def get_boundary_interface_scaled_jacobian_det(self) -> FieldData:
        return self._scaled_jac_det_bnd

    def _define_basis_at_shapes(self):
        self._basis = {
            shape: get_basis_by_shape(self.cfg, shape)
            for shape in self.pmesh.element_shapes
        }

    def _define_element_nodes(self):
        self._element_nodes = {
            shape: self._basis[shape].vertex_to_element_nodes(vtx)
            for shape, vtx in self.pmesh.vertex.items()
        }

    def _define_geometrical_metrics(self):
        self._element_jac_mat, self._element_jac_det = {}, {}
        self._surface_scaled_jac_det, self._surface_normal = {}, {}
        for shape, nodes in self._element_nodes.items():
            ijac, jac, det = self._basis[shape].element_geometrical_metrics(
                nodes
            )
            self._element_jac_mat[shape] = jac
            self._element_jac_det[shape] = det

            sdet, snormal = self._basis[shape].surface_geometrical_metrics(ijac)
            self._surface_scaled_jac_det[shape] = self._basis[
                shape
            ].scale_surface_jacobian_det(sdet, det)
            self._surface_normal[shape] = snormal

    @cached_property
    def _get_face_nodes(self):
        return {
            shape: self._basis[shape].face_nodes(element_nodes)
            for shape, element_nodes in self._element_nodes.items()
        }

    @cached_property
    def _get_sorted_face_node_ids(self):
        return {
            shape: [
                [
                    np.array(fuzzysort(n.tolist(), ids))
                    for n in face_nodes.transpose(1, 2, 0)
                ]
                for ids in self._basis[shape].face_node_ids
            ]
            for shape, face_nodes in self._get_face_nodes.items()
        }

    def _get_interface_face_node_ids(self, shape, eidx, fidx):
        num_face_nodes = self._basis[shape].num_face_nodes[fidx]
        rmap = self._get_sorted_face_node_ids[shape][fidx][eidx]
        cmap = (eidx,) * num_face_nodes
        return (rmap, cmap)

    def _get_interface_face_nodes(self, shape, eidx, fidx):
        ids = self._get_sorted_face_node_ids[shape][fidx][eidx]
        return (self._get_face_nodes[shape][ids, eidx],)

    def _get_interface_face_scaled_jacobian_det(self, shape, eidx, fidx):
        ids = self._get_sorted_face_node_ids[shape][fidx][eidx]
        return (self._surface_scaled_jac_det[shape][ids, eidx],)

    def _get_interface_face_normals(self, shape, eidx, fidx):
        ids = self._get_sorted_face_node_ids[shape][fidx][eidx]
        return (self._surface_normal[shape][ids, eidx],)

    def _get_interface_property(self, interface, getter):
        func = getattr(self, getter)
        vm = [func(shape, eidx, fidx) for shape, eidx, fidx, _ in interface]
        vm = [np.concatenate(m)[...] for m in zip(*vm)]
        return vm

    def _define_connectivity_internal_faces(self):
        self._node_ids_lhs, self._node_ids_rhs = {}, {}
        self._scaled_jac_det_lhs = {}
        self._normals_lhs, self._normals_rhs = {}, {}

        for key, conn in self.pmesh.connectivity_internal.items():
            lhs, rhs = conn

            get = lambda inter, me: self._get_interface_property(inter, me)
            get_both = lambda me: map(lambda inter: get(inter, me), (lhs, rhs))
            flatten = lambda a: map(lambda v: v[0], a)

            self._node_ids_lhs[key], self._node_ids_rhs[key] = get_both(
                "_get_interface_face_node_ids"
            )

            self._scaled_jac_det_lhs[key] = get(
                lhs, "_get_interface_face_scaled_jacobian_det"
            )[0]

            self._normals_lhs[key], self._normals_rhs[key] = flatten(
                get_both("_get_interface_face_normals")
            )

    def _define_connectivity_boundary_faces(self):
        self._node_ids_bnd = {}
        self._nodes_bnd = {}
        self._scaled_jac_det_bnd = {}
        self._normals_bnd = {}

        for key, lhs in self.pmesh.connectivity_boundaries.items():
            self._node_ids_bnd[key] = self._get_interface_property(
                lhs, "_get_interface_face_node_ids"
            )

            self._nodes_bnd[key] = self._get_interface_property(
                lhs, "_get_interface_face_nodes"
            )[0]

            self._scaled_jac_det_bnd[key] = self._get_interface_property(
                lhs, "_get_interface_face_scaled_jacobian_det"
            )[0]

            self._normals_bnd[key] = self._get_interface_property(
                lhs, "_get_interface_face_normals"
            )[0]

from functools import cached_property

import numpy as np
import torch
from typing_extensions import override

from edgfs2D.basis.base import BaseBasis, get_basis_for_shape
from edgfs2D.proto.sbp_pb2 import SBP
from edgfs2D.quadratures.jacobi import ortho_basis_at
from edgfs2D.utils.dictionary import Dictionary
from edgfs2D.utils.util import torch_map

np.set_printoptions(precision=3, linewidth=1e10)


# The nodal DG due to D. Del Rey Fernández, J. Hicken, and D. Zingg.
class FernandezHickenZingg(BaseBasis):
    kind = "fernandez-hicken-zingg"
    shape = "tri"
    dim = 2
    faces = 3

    def __init__(self, cfg: Dictionary, name: str, *args, **kwargs):
        super().__init__(cfg, name, *args, **kwargs)

        # degree of basis
        self._degree = degree = cfg.lookupint(name, "degree")

        # proto-buf data
        pb_data = get_basis_for_shape(shape=self.shape, name="fhz", deg=degree)
        sbp = SBP()
        sbp.ParseFromString(pb_data)
        assert sbp.degree == degree

        # define number of nodes
        self._num_nodes = sbp.numnodes

        # define vertex for shapes
        self._vtx = np.array(sbp.vtx).reshape(-1, self.dim)
        assert self._vtx.shape[0] == 3

        # define cubature weights
        cw = np.array(sbp.cw).flatten()
        assert cw.shape[0] == self.num_nodes

        # define inverse mass matrix
        self._iH = np.diag(1 / cw)

        # define derivative matrices
        Q = [np.array(Q.values).reshape(-1, self.num_nodes) for Q in sbp.Q]
        self._D = np.stack([self._iH @ q for q in Q])
        assert self._D.shape[1] == self.num_nodes
        assert self._D.shape[2] == self.num_nodes

        # define quadrature point
        self._qz = np.array(sbp.qz).reshape(-1, self.dim)
        assert self._qz.shape[0] == self.num_nodes

        # define vertex for shapes
        self._qw = np.array(sbp.qw).flatten()
        assert self._qw.shape[0] == self.num_nodes

        # define face weights
        self._sqw = np.array(sbp.sqw).flatten()
        self._sqz = np.array(sbp.sqz).flatten()

        # extract surface operators
        self._define_surface_operators()

    @override
    @property
    def degree(self):
        return self._degree

    @override
    @property
    def num_nodes(self):
        return self._num_nodes

    @override
    @property
    def quad_nodes(self):
        return self._qz

    @override
    @property
    def quad_weights(self):
        return self._qw

    @override
    def vertex_to_element_nodes(self, vertex: np.array):
        r, s = self._qz[:, 0], self._qz[:, 1]
        return 0.5 * (
            np.tensordot(-(r + s), vertex[0, ...], axes=0)
            + np.tensordot((1 + r), vertex[1, ...], axes=0)
            + np.tensordot((1 + s), vertex[2, ...], axes=0)
        )

    @override
    def element_geometrical_metrics(self, nodes: np.array):
        # compute inverse jacobian matrix i.e., dx / dxi
        ijac = np.tensordot(self._D, nodes, axes=(2, 0))

        # compute determinant of jacobian
        det = (
            ijac[0, ..., 0] * ijac[1, ..., 1]
            - ijac[1, ..., 0] * ijac[0, ..., 1]
        )
        assert np.all(det > 1e-8), "element jacobian determinant is too small"

        # compute jacobian i.e., dxi / dx
        jac = ijac.copy()
        jac[0, :, :, 0] = ijac[1, :, :, 1] / det  # dr / dx
        jac[0, :, :, 1] = -ijac[1, :, :, 0] / det  # dr / dy
        jac[1, :, :, 0] = -ijac[0, :, :, 1] / det  # ds / dx
        jac[1, :, :, 1] = ijac[0, :, :, 0] / det  # ds / dy

        return (ijac, jac, det)

    @override
    @cached_property
    def face_nodes_mask_all(self):
        return np.hstack(self._fpts_mask).ravel()

    @override
    @cached_property
    def face_nodes_mask_shape(self):
        return np.cumsum([len(m) for m in self._fpts_mask])

    @override
    def surface_geometrical_metrics(self, ijac: np.array):
        mask = self.face_nodes_mask_all
        N0, N1, N2 = self.face_nodes_mask_shape
        num_ele = ijac.shape[2]

        # extract components of inverse jacobian matrix
        xr = ijac[0, ..., 0]
        yr = ijac[0, ..., 1]
        xs = ijac[1, ..., 0]
        ys = ijac[1, ..., 1]

        # interpolate geometric metrics to surface
        fxr, fxs, fyr, fys = map(lambda x: x[mask, :], (xr, xs, yr, ys))

        # face points
        num_fpts = len(mask)

        # define normals
        snormal = np.zeros((num_fpts, num_ele, self.dim))

        # populate normals
        snormal[:N0, :, 0] = fyr[:N0, :]
        snormal[:N0, :, 1] = -fxr[:N0, :]
        snormal[N0:N1, :, 0] = fys[N0:N1, :] - fyr[N0:N1, :]
        snormal[N0:N1, :, 1] = -fxs[N0:N1, :] + fxr[N0:N1, :]
        snormal[N1:N2, :, 0] = -fys[N1:N2, :]
        snormal[N1:N2, :, 1] = fxs[N1:N2, :]

        # populate surface jacobian determinant
        sdet = np.sqrt(
            snormal[..., 0] * snormal[..., 0]
            + snormal[..., 1] * snormal[..., 1]
        )
        assert np.all(sdet > 1e-8), "surface jacobian determinant is too small"

        # normalize surface normal
        snormal[..., 0] /= sdet
        snormal[..., 1] /= sdet

        return (sdet, snormal)

    @override
    def scale_surface_jacobian_det(
        self, surface_jac_det: np.array, element_jac_det: np.array
    ):
        mask = self.face_nodes_mask_all
        surface_jac_det[:] /= element_jac_det[mask, ...]
        return surface_jac_det

    @override
    def face_nodes(self, element_nodes: np.array):
        return self.surface_data(element_nodes)

    @override
    @cached_property
    def num_face_nodes(self):
        return [len(m) for m in self._fpts_mask]

    @override
    @cached_property
    def face_node_ids(self):
        ids = np.cumsum([0] + self.num_face_nodes)
        return [np.arange(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]

    def _define_surface_operators(self):
        r, s = self._qz[:, 0], self._qz[:, 1]

        # mask of surface points
        tol = 1e-10
        self._fpts_mask = mask = [
            np.where(abs(1 + s) < tol)[0],
            np.where(abs(r + s) < tol)[0],
            np.where(abs(1 + r) < tol)[0],
        ]

        N0, N1, N2 = np.cumsum([len(m) for m in mask])
        Nqf = N2

        # since surface quadrature is sufficiently accurate, this is exact
        def weight(point):
            V1D = ortho_basis_at(self.degree + 2, point.ravel()).T
            return np.diag(np.linalg.inv(np.matmul(V1D, V1D.T)).sum(axis=0))

        Emat = np.zeros((self._num_nodes, Nqf))
        Emat[mask[0], 0:N0] = weight(r[mask[0]])
        Emat[mask[1], N0:N1] = weight(r[mask[1]])
        Emat[mask[2], N1:N2] = weight(s[mask[2]])
        self._L = np.matmul(self._iH, Emat)

    @cached_property
    def grad_op(self):
        return torch.from_numpy(self._D).to(self.cfg.device)

    @override
    def grad(
        self, element_data: torch.Tensor, element_jac: torch.Tensor
    ) -> torch.Tensor:
        gradu = torch.tensordot(self.grad_op, element_data, dims=1)
        ur, us = gradu[0], gradu[1]
        rx = element_jac[0, ..., 0].unsqueeze(-1)
        ry = element_jac[0, ..., 1].unsqueeze(-1)
        sx = element_jac[1, ..., 0].unsqueeze(-1)
        sy = element_jac[1, ..., 1].unsqueeze(-1)
        gu0 = rx * ur + sx * us
        gradu[1] = ry * ur + sy * us
        gradu[0] = gu0
        return gradu

    @override
    def surface_data(self, element_data: torch.Tensor) -> torch.Tensor:
        return element_data[self.face_nodes_mask_all, ...]

    @override
    def convect(
        self, grad_element_data: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        return torch.tensordot(velocity, grad_element_data, dims=1)

    @cached_property
    def lift_op(self):
        return torch.from_numpy(self._L).to(self.cfg.device)

    @override
    def lift(self, surface_data: torch.Tensor):
        return torch.tensordot(self.lift_op, surface_data, dims=1)

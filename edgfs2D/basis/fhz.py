from functools import cached_property, reduce

import numpy as np
import torch
from typing_extensions import override

from edgfs2D.basis.base import BaseBasis, get_basis_for_shape
from edgfs2D.proto.sbp_pb2 import SBP
from edgfs2D.quadratures.jacobi import ortho_basis_at, tri_northo_basis
from edgfs2D.utils.dictionary import Dictionary
from edgfs2D.utils.util import to_torch_device


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
        pb = SBP()
        pb.ParseFromString(pb_data)
        assert pb.degree == degree

        # define number of nodes
        self._num_nodes = pb.numnodes

        # define vertex for shapes
        self._vtx = np.array(pb.vtx).reshape(-1, self.dim)
        assert self._vtx.shape[0] == 3

        # define cubature weights
        cw = np.array(pb.cw).flatten()
        assert cw.shape[0] == self.num_nodes

        # define inverse mass matrix
        self._iH = np.diag(1 / cw)
        self._H = np.linalg.inv(self._iH)

        # define derivative matrices
        Q = [np.array(Q.values).reshape(-1, self.num_nodes) for Q in pb.Q]
        self._D = np.stack([self._iH @ q for q in Q])
        assert self._D.shape[1] == self.num_nodes
        assert self._D.shape[2] == self.num_nodes

        # define quadrature point
        self._qz = np.array(pb.qz).reshape(-1, self.dim)
        assert self._qz.shape[0] == self.num_nodes

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
        assert np.all(det > 0), "element jacobian determinant is too small"

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
        return to_torch_device(self._D, self.cfg)

    @override
    @torch.compile
    def grad(
        self, element_data: torch.Tensor, element_jac: torch.Tensor
    ) -> torch.Tensor:
        gradu = torch.tensordot(self.grad_op, element_data, dims=1)
        ur, us = gradu[0], gradu[1]
        rx = element_jac[0, 0, ..., 0].unsqueeze(-1).unsqueeze(0)
        ry = element_jac[0, 0, ..., 1].unsqueeze(-1).unsqueeze(0)
        sx = element_jac[1, 0, ..., 0].unsqueeze(-1).unsqueeze(0)
        sy = element_jac[1, 0, ..., 1].unsqueeze(-1).unsqueeze(0)
        gu0 = rx * ur + sx * us
        gradu[1] = ry * ur + sy * us
        gradu[0] = gu0
        return gradu

    @override
    @torch.compile
    def surface_data(self, element_data: torch.Tensor) -> torch.Tensor:
        return element_data[self.face_nodes_mask_all, ...]

    @override
    @torch.compile
    def convect(
        self, grad_element_data: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        return (
            velocity[0] * grad_element_data[0]
            + velocity[1] * grad_element_data[1]
        )

    @cached_property
    def lift_op(self):
        return to_torch_device(self._L, self.cfg)

    @override
    @torch.compile
    def lift(self, surface_data: torch.Tensor):
        return torch.tensordot(self.lift_op, surface_data, dims=1)

    @cached_property
    def mass_op(self):
        return to_torch_device(self._H, self.cfg)

    @override
    @torch.compile
    def error(self, error_data: torch.Tensor, element_jac_det: torch.Tensor):
        return (
            torch.sqrt(
                torch.tensordot(
                    element_jac_det,
                    torch.tensordot(self.mass_op, error_data**2, dims=1),
                    dims=[[0, 1], [0, 1]],
                )
            )
            .squeeze()
            .item()
            / 4
        )

    def rstoab(self, r, s):
        VSMALL = 1e-15
        a = 2 * (1.0 + r) / (1.0 - s + VSMALL) - 1.0
        return a, s

    def basis(self, r, s):
        p = self.degree
        B = np.zeros((int((p + 1) * (p + 2) / 2), len(r)))
        a, b = self.rstoab(r, s)

        idx = [(i, j) for i in range(p + 1) for j in range(p - i + 1)]
        for k, (i, j) in enumerate(idx):
            B[k, :] = tri_northo_basis(a, b, i, j).ravel()
        return B

    @cached_property
    def vdm(self):
        return self.basis(self._qz[:, 0], self._qz[:, 1]).T

    @override
    def interpolation_op(self, nodes: np.ndarray):
        V = self.vdm
        mul = lambda *args: reduce(np.matmul, args)
        Fwd = mul(np.linalg.inv(mul(V.T, self._H, V)), V.T, self._H)
        Vr = self.basis(nodes[:, 0], nodes[:, 1]).T
        return mul(Vr, Fwd)

    @override
    @torch.compile
    def interpolate(self, element_data: np.ndarray, interp_op: np.ndarray):
        return np.tensordot(interp_op, element_data, axes=1)

    @override
    @torch.compile
    def convect_eflux(
        self, eflux_data: torch.Tensor, element_jac: torch.Tensor
    ) -> torch.Tensor:
        f1s, f2s = eflux_data[0], eflux_data[1]
        Dr = self.grad_op[0][..., None, None]
        Ds = self.grad_op[1][..., None, None]

        rx = element_jac[0, 0, ..., 0].unsqueeze(-1).unsqueeze(0)
        ry = element_jac[0, 0, ..., 1].unsqueeze(-1).unsqueeze(0)
        sx = element_jac[1, 0, ..., 0].unsqueeze(-1).unsqueeze(0)
        sy = element_jac[1, 0, ..., 1].unsqueeze(-1).unsqueeze(0)

        return 2 * (
            rx * (Dr * f1s).sum(axis=1)
            + sx * (Ds * f1s).sum(axis=1)
            + ry * (Dr * f2s).sum(axis=1)
            + sy * (Ds * f2s).sum(axis=1)
        )

    @override
    @torch.compile
    def convect_contravariant(
        self,
        element_data: torch.Tensor,
        velocity: torch.Tensor,
        element_ijac: torch.Tensor,
        element_jac_det: torch.Tensor,
    ) -> torch.Tensor:
        element_data.shape
        f1, f2 = element_data * velocity[0], element_data * velocity[1]
        Dr, Ds = self.grad_op[0], self.grad_op[1]

        xr = element_ijac[0, ..., 0].unsqueeze(-1)
        xs = element_ijac[1, ..., 0].unsqueeze(-1)
        yr = element_ijac[0, ..., 1].unsqueeze(-1)
        ys = element_ijac[1, ..., 1].unsqueeze(-1)

        t1 = ys * f1 - xs * f2
        t2 = -yr * f1 + xr * f2

        res = (torch.tensordot(Dr, t1, dims=1)).add_(
            torch.tensordot(Ds, t2, dims=1)
        )
        res.div_(element_jac_det[..., None])

        return res

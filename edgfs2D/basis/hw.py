from functools import cached_property

import numpy as np
import torch
from typing_extensions import override

from edgfs2D.basis.base import BaseBasis, get_basis_for_shape
from edgfs2D.basis.fhz import FernandezHickenZingg
from edgfs2D.proto.hw_pb2 import HW
from edgfs2D.quadratures.jacobi import ortho_basis_at, tri_jac_northo_basis
from edgfs2D.utils.dictionary import Dictionary

np.set_printoptions(suppress=True, linewidth=2000, precision=4)


# The nodal DG due to Jan S. Hesthaven and Tim Warburton
class HesthavenWarburton(FernandezHickenZingg):
    kind = "hesthaven-warburton"

    def __init__(self, cfg: Dictionary, name: str, *args, **kwargs):
        BaseBasis.__init__(self, cfg, name, *args, **kwargs)

        # degree of basis
        self._degree = degree = cfg.lookupint(name, "degree")

        # proto-buf data
        pb_data = get_basis_for_shape(shape=self.shape, name="hw", deg=degree)
        pb = HW()
        pb.ParseFromString(pb_data)
        assert pb.degree == degree

        # define number of nodes
        self._num_nodes = pb.numnodes

        # define vertex for shapes
        self._vtx = np.array(pb.vtx).reshape(-1, self.dim)
        assert self._vtx.shape[0] == 3

        # define quadrature point
        self._qz = np.array(pb.qz).reshape(-1, self.dim)
        assert self._qz.shape[0] == self.num_nodes

        # define inverse mass matrix
        V = self.vdm
        Vr, Vs = self.grad_vdm
        self._iH = np.matmul(V, V.T)
        self._H = np.linalg.inv(self._iH)

        # define derivative matrices
        Dr = np.linalg.solve(V.T, Vr.T).T
        Ds = np.linalg.solve(V.T, Vs.T).T
        self._D = np.stack([Dr, Ds])

        # extract surface operators
        self._define_surface_operators()

    @override
    @torch.compile
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
    @torch.compile
    def convect_eflux(
        self, eflux_data: torch.Tensor, element_jac: torch.Tensor
    ) -> torch.Tensor:
        raise RuntimeError("not implemented")

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
        def edge_mat(point):
            V1D = ortho_basis_at(self.degree + 1, point.ravel()).T
            return np.linalg.inv(np.matmul(V1D, V1D.T))

        Emat = np.zeros((self._num_nodes, Nqf))
        Emat[mask[0], 0:N0] = edge_mat(r[mask[0]])
        Emat[mask[1], N0:N1] = edge_mat(r[mask[1]])
        Emat[mask[2], N1:N2] = edge_mat(s[mask[2]])
        self._L = np.matmul(self._iH, Emat)

    def grad_basis(self, r, s):
        p = self.degree
        dBr = np.zeros((int((p + 1) * (p + 2) / 2), len(r)))
        dBs = np.zeros((int((p + 1) * (p + 2) / 2), len(r)))
        a, b = self.rstoab(r, s)

        k = 0
        for i in range(p + 1):
            for j in range(p - i + 1):
                dBr[k, :], dBs[k, :] = map(
                    np.ravel, tri_jac_northo_basis(a, b, i, j)
                )
                k += 1
        return dBr, dBs

    @cached_property
    def grad_vdm(self):
        dBr, dBs = self.grad_basis(self._qz[:, 0], self._qz[:, 1])
        return (dBr.T, dBs.T)

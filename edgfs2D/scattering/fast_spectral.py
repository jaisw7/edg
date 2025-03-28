from math import gamma

import numpy as np
import torch
from loguru import logger

from edgfs2D.quadratures.jacobi import zwgj
from edgfs2D.scattering.base import BaseScatteringModel
from edgfs2D.sphericaldesign import get_sphquadrule
from edgfs2D.utils.dictionary import SubDictionary
from edgfs2D.utils.nputil import ndrange, npeval
from edgfs2D.utils.util import to_torch_device


# Simplified VHS model for GLL based nodal collocation schemes
class FastSpectral(BaseScatteringModel):
    kind = "fast-spectral-vhs"
    allowed_solvers = ["ClassicFastSpectralSolver"]

    def __init__(self, cfg: SubDictionary, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.load_parameters()
        self.compute_quadratures()
        self.perform_precomputation()
        logger.info("scattering-model: finished computation")

    def load_parameters(self):
        nd = self.vmesh.nondim
        omega = self._cfg.lookupfloat("omega")
        dRef = self._cfg.lookupfloat("dRef")
        Tref = self._cfg.lookupfloat("Tref")

        alpha = 1.0
        self._gamma = 2.0 * (1 - omega)

        invKn = (
            nd.H0
            * np.sqrt(2.0)
            * np.pi
            * nd.n0
            * dRef
            * dRef
            * pow(Tref / nd.T0, omega - 0.5)
        )

        self._prefactor = (
            invKn
            * alpha
            / (pow(2.0, 2 - omega + alpha) * gamma(2.5 - omega) * np.pi)
        )
        self._omega = omega
        logger.info("Kn {}, prefactor {}", 1 / invKn, self._prefactor)

    def compute_quadratures(self):
        # spherical quadrature for integration on sphere
        self._ssrule = self._cfg.lookup("spherical_rule")
        self._M = self._cfg.lookupint("M")
        srule = get_sphquadrule(
            "symmetric", rule=self._ssrule, npts=2 * self._M
        )
        self._sz = to_torch_device(srule.pts[0 : self._M, :], self._cfg)
        self._sw = 2 * np.pi / self._M

        # support of velocity mesh
        self._R = self.vmesh.extents * 4 / (3.0 + np.sqrt(2.0))

        # quadrature in rho direction
        self._Nrho = self._cfg.lookupint("Nrho")
        self._qz, self._qw = zwgj(self._Nrho, 0.0, 0.0)
        # scale the quadrature from [-1, 1] to [0, R]
        self._qz = to_torch_device((self._R / 2) * (1 + self._qz), self._cfg)
        self._qw = to_torch_device(
            ((self._R - 0.0) / 2.0) * self._qw, self._cfg
        )

    def perform_precomputation(self):
        # Precompute aa, bb1, bb2 (required for kernel)
        vm = self.vmesh
        N = vm.shape[0]
        L = vm.extents
        qz = self._qz
        qw = self._qw
        sw = self._sw
        vsize = vm.num_points
        gamma = self._gamma

        # compute l: permutation for reindexing FFT modes
        l0 = np.concatenate((np.arange(0, N / 2), np.arange(-N / 2, 0)))
        perm = np.zeros((3, vsize))
        for idv in range(vsize):
            I = int(idv / (N * N))
            J = int((idv % (N * N)) / N)
            K = int((idv % (N * N)) % N)
            perm[0, idv] = l0[I]
            perm[1, idv] = l0[J]
            perm[2, idv] = l0[K]
        self._l = to_torch_device(perm, self._cfg)

        # precompute aa
        fac = torch.pi / L / 2
        self._aa = (fac * self._qz)[:, None, None] * torch.tensordot(
            self._sz, self._l, dims=1
        )[None, ...]

        # precompute bb
        cSqr = (self._l**2).sum(dim=0).sqrt_()
        eps = 1e-15 if self._cfg.ttype == torch.float64 else 1e-6
        term = fac * torch.outer(qz, cSqr) + eps
        pi = torch.pi
        common = (8 * pi * qw * (qz ** (gamma + 2))).unsqueeze(-1)
        self._bb1 = sw * common * (torch.sin(term) / term)
        term = 2 * fac * torch.outer(qz, cSqr) + eps
        self._bb2 = ((2 * pi) * common * (torch.sin(term) / term)).sum(dim=0)

    def solve(
        self,
        curr_time: torch.float64,
        element_data: torch.Tensor,
        out: torch.Tensor,
    ):
        for k, e in ndrange(*element_data.shape[:2]):
            out[k, e, :].add_(
                self.solve_at_point(element_data[k, e, :], self._prefactor)
            )

    def solve_at_point(self, f0: torch.Tensor, prefactor):
        # convert real valued tensor to complex valued tensor
        shape = self.vmesh.shape
        mnshape = (self._M, self._Nrho, *shape)
        fft3 = lambda f: torch.fft.fftn(f, norm="ortho", dim=(-3, -2, -1))
        ifft3s = lambda f: torch.fft.ifftn(
            f, norm="ortho", dim=(-3, -2, -1), out=f
        )

        # compute forward FFT of f | Ftf = fft(f)
        Ftf = fft3(f0.reshape(shape))

        # compute t1_{pqr} = cos(a_{pqr})*FTf_r; t2_{pqr} = sin(a_{pqr})*FTf_r
        t1 = (torch.cos(self._aa) * (Ftf.ravel()[None, None, :])).reshape(
            mnshape
        )
        t2 = (torch.sin(self._aa) * (Ftf.ravel()[None, None, :])).reshape(
            mnshape
        )

        # compute t2 = ifft(t1)^2 + ifft(t2)^2
        ifft3s(t1).pow_(2)
        ifft3s(t2).pow_(2)
        t2.add_(t1)

        # compute t1 = fft(t2)
        t1 = fft3(t2).reshape(self._Nrho, self._M, -1)

        # compute fC_r = b1_p*t1_r
        fC = ((self._bb1.unsqueeze(1) * t1).sum(dim=(0, 1))).reshape(shape)

        # inverse fft| fC = iff(fC)  [Gain computed]
        ifft3s(fC)

        # compute FTf_r = b2_r*FTf_r
        Ftf.mul_(self._bb2.reshape(shape))

        # inverse fft| FTf = iff(FTf)
        ifft3s(Ftf)

        # output
        return (fC - Ftf.mul_(f0.reshape(shape))).real.mul_(prefactor).ravel()


class PenalizedFastSpectral(FastSpectral):
    kind = "penalized-fast-spectral-vhs"
    allowed_solvers = ["ImexFastSpectralSolver"]

    def solve(
        self,
        curr_time: torch.float64,
        element_data: torch.Tensor,
        out: torch.Tensor,
    ):
        for k, e in ndrange(*element_data.shape[:2]):
            out[k, e, :].add_(
                self.solve_at_point(element_data[k, e, :], self._prefactor)
            )


class PenalizedFastSpectralMixingRegime(FastSpectral):
    kind = "penalized-fast-spectral-vhs-mixing-regime"
    allowed_solvers = ["ImexFastSpectralSolver"]

    def load_parameters(self):
        alpha = 1.0
        self._omega = omega = self._cfg.lookupfloat("omega")
        self._gamma = 2.0 * (1 - omega)

        nodes = self.dgmesh.get_element_nodes["tri"]
        vars = {"x": nodes[..., 0], "y": nodes[..., 1]}
        invKn = 1.0 / npeval(self._cfg.lookupexpr("Kn-expr"), vars)

        self._prefactor = to_torch_device(
            invKn
            * alpha
            / (pow(2.0, 2 - omega + alpha) * gamma(2.5 - omega) * np.pi),
            self._cfg,
        )
        Kn = 1 / invKn
        logger.info("Kn: ({}, {})", Kn.min(), Kn.max())

    def solve(
        self,
        curr_time: torch.float64,
        element_data: torch.Tensor,
        out: torch.Tensor,
    ):
        prefactor = self._prefactor
        for k, e in ndrange(*element_data.shape[:2]):
            out[k, e, :].add_(
                self.solve_at_point(element_data[k, e, :], prefactor[k, e])
            )

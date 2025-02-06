from functools import cached_property

import torch
from loguru import logger

from edgfs2D.fields.types import FieldData
from edgfs2D.scattering.base import BaseScatteringModel
from edgfs2D.utils.dictionary import SubDictionary
from edgfs2D.utils.nputil import npeval
from edgfs2D.utils.util import to_torch_device


class BaseRelaxationModel(BaseScatteringModel):
    pass


# Bhatnagar-Gross-Krook model
class BgkRelaxation(BaseRelaxationModel):
    kind = "bgk-relaxation"
    allowed_solvers = ["ImexFastSpectralSolver"]
    num_fields = 5

    def __init__(self, cfg: SubDictionary, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.load_parameters()
        logger.info("scattering-model: finished computation")

    def load_parameters(self):
        ndim = self.vmesh.nondim
        omega = self._cfg.lookupfloat("omega")
        muRef = self._cfg.lookupfloat("muRef")
        Tref = self._cfg.lookupfloat("Tref")

        t0 = ndim.H0 / ndim.u0
        visc = muRef * ((ndim.T0 / Tref) ** omega)
        p0 = ndim.n0 * ndim.R0 / ndim.NA * ndim.T0

        self._prefactor = t0 * p0 / visc
        self._omega = omega
        self._Pr = 1
        logger.info("prefactor {}", self._prefactor)

    @cached_property
    def moments_op(self):
        vpoints = self.vmesh.points
        magv = torch.sum(vpoints**2, dim=0)
        out = torch.vstack((torch.ones_like(magv), vpoints, magv)).T
        out.mul_(self.vmesh.weights)
        return out

    def distribution_op(
        self, m: torch.Tensor, out: torch.Tensor, nu: torch.Tensor
    ):
        vpoints = self._distmesh.vmesh.points
        rho = m[..., 0]
        u = m[..., 1:4] / rho.unsqueeze(-1)
        T = (m[..., 4] - rho * (u**2).sum(dim=-1)).div_(rho).div_(1.5)
        nu[..., 0] = (rho * T * (1 - self._omega)).mul_(self._prefactor)
        out.copy_(
            ((vpoints.unsqueeze(0).unsqueeze(0) - u.unsqueeze(-1)) ** 2)
            .sum(dim=-2)
            .mul_(-1)
        )
        out.div_(T.unsqueeze(-1)).exp_()
        out.mul_(rho.unsqueeze(-1))
        out.div_(torch.pi**1.5).div_(T.unsqueeze(-1) ** 1.5)

    def moments(self, u: FieldData, out: FieldData):
        for shape in out.keys():
            torch.matmul(u[shape], self.moments_op, out=out[shape])

    def construct_distribution(
        self, m: FieldData, out: FieldData, nu: FieldData
    ):
        for shape in out.keys():
            self.distribution_op(m[shape], out[shape], nu[shape])

    def solve(self):
        raise RuntimeError("not implemented")


class BgkRelaxationMixingRegime(BgkRelaxation):
    kind = "bgk-relaxation-mixing-regime"
    allowed_solvers = ["ImexFastSpectralSolver"]

    def load_parameters(self):
        self._omega = omega = self._cfg.lookupfloat("omega")

        nodes = self.dgmesh.get_element_nodes["tri"]
        vars = {"x": nodes[..., 0], "y": nodes[..., 1]}
        invKn = 1.0 / npeval(self._cfg.lookupexpr("Kn-expr"), vars)

        self._prefactor = to_torch_device(100 * invKn, self._cfg)
        Kn = 1 / invKn
        logger.info("Kn: ({}, {})", Kn.min(), Kn.max())

import torch
from typing_extensions import override

from edgfs2D.entropy_fluxes import get_eflux_by_cls_and_name
from edgfs2D.entropy_fluxes.base import BaseEntropyFlux
from edgfs2D.utils.dictionary import SubDictionary
from edgfs2D.utils.nputil import ndrange
from edgfs2D.velocity_mesh.base import BaseVelocityMesh


class FastSpectralEntropyFlux(BaseEntropyFlux):
    pass


class BoltzmannEntropyFlux(FastSpectralEntropyFlux):
    kind = "fast-spectral-boltzmann-entropy"

    def __init__(
        self, cfg: SubDictionary, vm: BaseVelocityMesh, *args, **kwargs
    ):
        super().__init__(cfg, *args, **kwargs)
        self._velocity = vm.points

    @override
    @property
    def velocity(self):
        return self._velocity

    def log_mean(self, x: torch.Tensor, y: torch.Tensor):
        # Ismail and Roe, J. Comput. Phys. 228 (15) (2009) 5410–5436
        eps = torch.finfo(x.dtype).eps
        xi = torch.abs(x / (y + eps))
        f = (xi - 1) / (xi + 1)
        u = f * f
        return (x + y) / torch.where(
            u < 1e-2,
            ((210 + u * (70 + u * (42 + u * 30)))) / 105,
            torch.log(xi + eps) / (f + eps),
        )

    @override
    def apply(self, ul: torch.Tensor, ur: torch.Tensor):
        npts = ul.shape[0]
        fs = torch.zeros((2, npts, *ul.shape), dtype=ul.dtype, device=ul.device)
        for i, j in ndrange(npts, npts):
            flux = self.log_mean(ul[i, ...], ur[j, ...])
            fs[0, i, j, ...] = self.velocity[0] * flux
            fs[1, i, j, ...] = self.velocity[1] * flux
        return fs


def get_eflux(cfg, name, *args, **kwargs):
    return get_eflux_by_cls_and_name(
        cfg, name, FastSpectralEntropyFlux, *args, **kwargs
    )

import torch
from typing_extensions import override

from edgfs2D.fluxes import get_flux_by_cls_and_name
from edgfs2D.fluxes.base import BaseFlux
from edgfs2D.utils.dictionary import SubDictionary
from edgfs2D.velocity_mesh.base import BaseVelocityMesh


class FastSpectralFlux(BaseFlux):
    pass


class LaxFriedrichsFlux(FastSpectralFlux):
    kind = "fast-spectral-lax-friedrichs"

    def __init__(
        self, cfg: SubDictionary, vm: BaseVelocityMesh, *args, **kwargs
    ):
        super().__init__(cfg, *args, **kwargs)
        self._velocity = vm.points

    @override
    @property
    def velocity(self):
        return self._velocity

    @override
    @torch.compile
    def apply(self, ul: torch.Tensor, ur: torch.Tensor, nl: torch.Tensor):
        # As per Hasthaven pp. 170, Ch. 6
        dim = nl.shape[1]
        nu = torch.tensordot(nl, self._velocity[:dim, :], dims=1)
        C = nu.abs()
        flux = 0.5 * (nu * (ul + ur) + C * (ul - ur))

        # update flux on left interface
        ul.copy_(flux)
        # update flux on right interface
        ur.copy_(-flux)


def get_flux(cfg, name, *args, **kwargs):
    return get_flux_by_cls_and_name(
        cfg, name, FastSpectralFlux, *args, **kwargs
    )

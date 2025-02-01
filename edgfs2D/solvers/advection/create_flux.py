import torch
from typing_extensions import override

from edgfs2D.fluxes import get_flux_by_cls_and_name
from edgfs2D.fluxes.base import BaseFlux
from edgfs2D.utils.dictionary import SubDictionary


class AdvFlux(BaseFlux):
    pass


class LaxFriedrichsFlux(AdvFlux):
    kind = "advection-lax-friedrichs"

    def __init__(self, cfg: SubDictionary, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._velocity = (
            torch.from_numpy(cfg.lookupfloat_list("velocity"))
            .to(dtype=cfg.ttype, device=cfg.device)
            .reshape(-1, 1)
        )

    @property
    def velocity(self):
        return self._velocity

    @override
    def apply(self, ul: torch.Tensor, ur: torch.Tensor, nl: torch.Tensor):
        # As per Hasthaven pp. 170, Ch. 6
        nu = torch.tensordot(nl, self._velocity, dims=1)
        C = nu.abs().max()
        flux = 0.5 * (nu * (ul + ur) + C * (ul - ur))

        # update flux on left interface
        ul.copy_(flux)
        # update flux on right interface
        ur.copy_(-flux)


def get_flux(cfg, name, *args, **kwargs):
    return get_flux_by_cls_and_name(cfg, name, AdvFlux, *args, **kwargs)

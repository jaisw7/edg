import torch
from typing_extensions import override

from edgfs2D.entropy_fluxes import get_eflux_by_cls_and_name
from edgfs2D.entropy_fluxes.base import BaseEntropyFlux
from edgfs2D.utils.dictionary import SubDictionary
from edgfs2D.utils.util import torch_map


class AdvEntropyFlux(BaseEntropyFlux):
    pass


class TwoPointEntropyFlux(AdvEntropyFlux):
    kind = "advection-two-point-eflux"

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
    def apply(self, ul: torch.Tensor, ur: torch.Tensor):
        npts = ul.shape[0]
        fs = torch.zeros((2, npts, *ul.shape), dtype=ul.dtype, device=ul.device)
        for i in range(npts):
            fs[0, i, ...] = 0.5 * self._velocity[0] * (ul[i, :] + ur)
            fs[1, i, ...] = 0.5 * self._velocity[1] * (ul[i, :] + ur)
        return fs


def get_eflux(cfg, name, *args, **kwargs):
    return get_eflux_by_cls_and_name(cfg, name, AdvEntropyFlux, *args, **kwargs)

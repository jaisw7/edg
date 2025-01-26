import torch
from typing_extensions import override

from edgfs2D.boundary_conditions import get_bc_by_cls_and_name
from edgfs2D.boundary_conditions.base import BaseBoundaryCondition


class AdvBoundaryCondition(BaseBoundaryCondition):
    pass


class SinCosBoundaryCondition(AdvBoundaryCondition):
    kind = "advection-sincos"

    @override
    def apply(
        self,
        curr_time: torch.float64,
        ul: torch.Tensor,
    ):
        xl = self._nodes
        return (
            torch.sin(torch.pi * (xl[..., 0] - curr_time))
            * torch.cos(torch.pi * (xl[..., 1] - curr_time))
        ).unsqueeze(-1)


def get_boundary_condition(cfg, name, *args, **kwargs):
    return get_bc_by_cls_and_name(
        cfg, name, AdvBoundaryCondition, *args, **kwargs
    )

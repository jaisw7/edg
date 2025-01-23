import numpy as np
from typing_extensions import override

from edgfs2D.initial_conditions import get_ic_by_cls_and_name
from edgfs2D.initial_conditions.base import BaseInitialCondition


class AdvInitialCondition(BaseInitialCondition):
    pass


class BumpInitialCondition(AdvInitialCondition):
    kind = "advection-bump"

    @override
    def apply(self, x: np.array):
        usol = np.ones_like(x[..., 0])[..., np.newaxis]
        r = np.sqrt((x[..., 0] - 0.5) ** 2 + (x[..., 1] - 0.5) ** 2).ravel()
        ids = np.where(r < 0.5)[0]
        usol.ravel()[ids] = 1 - ((4 * (r[ids] ** 2) - 1) ** 5)
        return usol


class SinCosInitialCondition(AdvInitialCondition):
    kind = "advection-sincos"

    @override
    def apply(self, x: np.array):
        return (np.sin(np.pi * x[..., 0]) * np.cos(np.pi * x[..., 1]))[
            ..., np.newaxis
        ]


def get_initial_condition(cfg, name, *args, **kwargs):
    return get_ic_by_cls_and_name(
        cfg, name, AdvInitialCondition, *args, **kwargs
    )

import numpy as np
from typing_extensions import override

from edgfs2D.initial_conditions import get_ic_by_cls_and_name
from edgfs2D.initial_conditions.base import BaseInitialCondition


class FastSpectralInitialCondition(BaseInitialCondition):
    pass


class MaxwellianInitialCondition(FastSpectralInitialCondition):
    kind = "fast-spectral-maxwellian"

    @override
    def apply(self, x: np.array):
        return (np.sin(np.pi * x[..., 0]) * np.cos(np.pi * x[..., 1]))[
            ..., np.newaxis
        ]


def get_initial_condition(cfg, name, *args, **kwargs):
    return get_ic_by_cls_and_name(
        cfg, name, FastSpectralInitialCondition, *args, **kwargs
    )

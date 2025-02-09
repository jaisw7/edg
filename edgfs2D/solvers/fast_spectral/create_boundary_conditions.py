from functools import cached_property

import numpy as np
import torch
from typing_extensions import override

from edgfs2D.boundary_conditions import get_bc_by_cls_and_name
from edgfs2D.boundary_conditions.base import BaseBoundaryCondition
from edgfs2D.solvers.fast_spectral.create_initial_conditions import (
    MaxwellianInitialCondition,
)
from edgfs2D.utils.dictionary import SubDictionary
from edgfs2D.utils.util import fuzzysort
from edgfs2D.velocity_mesh.base import BaseVelocityMesh


class FastSpectralBoundaryCondition(BaseBoundaryCondition):
    kind = None

    @override
    def __init__(
        self,
        cfg: SubDictionary,
        nodes: torch.Tensor,
        normals: torch.Tensor,
        vmesh: BaseVelocityMesh,
        **kwargs
    ):
        super().__init__(cfg, nodes, normals, **kwargs)
        self._vmesh = vmesh

    @property
    def vmesh(self):
        return self._vmesh

    @cached_property
    def vpoints(self):
        return self._vmesh.points

    @cached_property
    def vweights(self):
        assert isinstance(
            self._vmesh.weights, (np.float32, np.float64)
        ), "needs scalar"
        return self._vmesh.weights


class DiffuseWallBoundaryCondition(FastSpectralBoundaryCondition):
    kind = "fast-spectral-diffuse-wall"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ic = MaxwellianInitialCondition(self._cfg, self._vmesh, read_rho=False)
        self._f0 = ic.init_vals
        self._u = ic.u
        _ = self.trace_ids

    @cached_property
    def trace_ids(self):
        """Find inflow and outflow locations"""
        nl = self.normals
        if torch.max(torch.abs(nl - nl.mean(dim=0))) > 1e-5:
            raise ValueError("all normals must be identical")

        nu = torch.tensordot(
            nl[0, ...], self.vpoints[:2, ...] - self._u[:2, ...], dims=1
        )
        pos = torch.where(nu >= 0, 1, 0)
        neg = torch.where(nu < 0, 1, 0)
        return (
            pos,
            neg,
            pos * nu * self.vweights,
            -(neg * nu * self.vweights * self._f0).sum(),
        )

    @override
    @torch.compile
    def apply(
        self,
        curr_time: torch.float64,
        ul: torch.Tensor,
    ):
        pos_id, neg_id, pos_val, neg_val = self.trace_ids

        # compute number density
        nden = (pos_val * ul).sum(axis=1) / neg_val

        # update flux
        ur = pos_id * ul
        ur.add_(torch.outer(nden, neg_id * self._f0))

        return ur


class SpecularWallBoundaryCondition(FastSpectralBoundaryCondition):
    kind = "fast-spectral-specular-wall"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _ = self.trace_ids

    @cached_property
    def trace_ids(self):
        """Find inflow and outflow locations"""
        nl = self.normals
        if torch.max(torch.abs(nl - nl.mean(dim=0))) > 1e-5:
            raise ValueError("all normals must be identical")

        Nv = self.vmesh.num_points
        norm = torch.hstack((nl[0, :], torch.zeros(1, device=self._cfg.device)))
        cr = self.vpoints - 2 * torch.outer(
            norm, torch.tensordot(norm, self.vpoints, dims=1)
        )
        sorted_idx = torch.arange(0, Nv, device=self._cfg.device)
        f0 = torch.tensor(fuzzysort(cr, sorted_idx), device=self._cfg.device)
        if not torch.all(torch.sort(f0)[0] == sorted_idx):
            raise ValueError("Non-unique map")
        return f0

    @override
    @torch.compile
    def apply(
        self,
        curr_time: torch.float64,
        ul: torch.Tensor,
    ):
        idx = self.trace_ids
        return ul[..., idx]


class DiffuseCurvedWallBoundaryCondition(FastSpectralBoundaryCondition):
    kind = "fast-spectral-diffuse-curved-wall"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ic = MaxwellianInitialCondition(self._cfg, self._vmesh, read_rho=False)
        self._f0 = ic.init_vals
        self._u = ic.u
        _ = self.trace_ids

    @cached_property
    def trace_ids(self):
        """Find inflow and outflow locations"""
        nl = self.normals

        nu = torch.tensordot(
            nl, self.vpoints[:2, ...] - self._u[:2, ...], dims=1
        )
        pos = torch.where(nu >= 0, 1, 0)
        neg = torch.where(nu < 0, 1, 0)
        return (
            pos,
            neg,
            pos * nu * self.vweights,
            -(neg * nu * self.vweights * self._f0).sum(axis=1),
        )

    @override
    @torch.compile
    def apply(
        self,
        curr_time: torch.float64,
        ul: torch.Tensor,
    ):
        pos_id, neg_id, pos_val, neg_val = self.trace_ids

        # compute number density
        nden = (pos_val * ul).sum(axis=1) / neg_val

        # update flux
        ur = pos_id * ul
        ur.add_(nden.unsqueeze(-1) * neg_id * self._f0)

        return ur


def get_boundary_condition(cfg, name, *args, **kwargs):
    return get_bc_by_cls_and_name(
        cfg, name, FastSpectralBoundaryCondition, *args, **kwargs
    )

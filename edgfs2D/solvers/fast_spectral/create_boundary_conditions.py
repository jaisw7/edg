from functools import cached_property

import numpy as np
import torch
from loguru import logger
from typing_extensions import override

from edgfs2D.boundary_conditions import get_bc_by_cls_and_name
from edgfs2D.boundary_conditions.base import BaseBoundaryCondition
from edgfs2D.solvers.fast_spectral.create_initial_conditions import (
    MaxwellianInitialCondition,
)
from edgfs2D.utils.dictionary import SubDictionary
from edgfs2D.utils.nputil import npeval
from edgfs2D.utils.util import fuzzysort, to_torch_device
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
        **kwargs,
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


class DiffuseWallBoundaryCondition(FastSpectralBoundaryCondition):
    kind = "fast-spectral-diffuse-wall"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ic = MaxwellianInitialCondition(self._cfg, self._vmesh, read_rho=False)
        self._f0 = ic.init_vals
        self._u = ic.u
        _ = self.trace_ids

    def compute_trace_ids(self, nl):
        """Find inflow and outflow locations"""
        nu = torch.tensordot(
            nl, self.vpoints[:2, ...] - self._u[:2, ...], dims=1
        )
        pos = torch.where(nu >= 0, 1, 0)
        neg = torch.where(nu < 0, 1, 0)
        return (
            nu,
            pos * nu * self.vweights,
            -(neg * nu * self.vweights * self._f0).sum(axis=1),
        )

    @cached_property
    def trace_ids(self):
        nl = self.normals
        if torch.max(torch.abs(nl - nl.mean(dim=0))) > 1e-5:
            raise ValueError("all normals must be identical")
        return self.compute_trace_ids(nl[0, ...].unsqueeze(0))

    @override
    @torch.compile
    def apply(
        self,
        curr_time: torch.float64,
        ul: torch.Tensor,
    ):
        nu, pos_val, neg_val = self.trace_ids

        # compute number density
        nden = (pos_val * ul).sum(axis=1) / neg_val

        # update flux
        ur = torch.where(nu >= 0, ul, nden.unsqueeze(-1) * self._f0)

        return ur


class DiffuseCurvedWallBoundaryCondition(DiffuseWallBoundaryCondition):
    kind = "fast-spectral-diffuse-curved-wall"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def trace_ids(self):
        return self.compute_trace_ids(self.normals)


class DiffuseCurvedWallExprBoundaryCondition(DiffuseWallBoundaryCondition):
    kind = "fast-spectral-diffuse-curved-wall-expr"

    def t(self, expr, ndim_var, shape, vars={}):
        val = npeval(f"({expr}) / {ndim_var}", vars)
        val = val if isinstance(val, np.ndarray) else np.full(shape, val)
        val = val.reshape(shape)
        return val

    def arc_length(self, nodes):
        idx = np.arange(0, nodes.shape[0])
        srtidx = fuzzysort(nodes.swapaxes(0, 1), idx)
        nodes = np.vstack(([0, 0], nodes[srtidx, :]))
        deltas = np.diff(nodes, axis=0)
        lengths = np.linalg.norm(deltas, axis=1).cumsum()
        lengths[srtidx] = lengths[idx]
        logger.info("arc-lengths: ({}, {})", min(lengths), max(lengths))
        return lengths

    def __init__(self, *args, **kwargs):
        FastSpectralBoundaryCondition.__init__(self, *args, **kwargs)

        nodes = self.nodes.cpu().numpy()
        shape = (nodes.shape[0], 1)
        vars = {"x": nodes[..., 0], "y": nodes[..., 1]}
        vars.update({"s": self.arc_length(nodes)})
        vars.update(self._cfg.section_values(self._cfg.dtype))
        ndim = self.vmesh.nondim

        u = np.array(
            [
                self.t(expr, ndim.u0, shape, vars)
                for expr in self._cfg.lookupexpr_list("u")
            ]
        )
        T = self.t(self._cfg.lookupexpr("T"), ndim.T0, shape, vars)

        init_vals = (1 / (np.pi * T) ** 1.5) * np.exp(
            -np.sum((np.expand_dims(self.vmesh._cv, axis=1) - u) ** 2, axis=0)
            / T
        )

        self._u = to_torch_device(u, self._cfg)
        self._f0 = to_torch_device(init_vals, self._cfg)
        _ = self.trace_ids

    @cached_property
    def trace_ids(self):
        nl = self.normals
        nu = (
            nl.swapaxes(0, 1).unsqueeze(-1)
            * (self.vpoints[:2, ...].unsqueeze(1) - self._u[:2, ...])
        ).sum(axis=0)
        pos = torch.where(nu >= 0, 1, 0)
        neg = torch.where(nu < 0, 1, 0)
        return (
            nu,
            pos * nu * self.vweights,
            -(neg * nu * self.vweights * self._f0).sum(axis=1),
        )


class DiffuseInletBoundaryCondition(FastSpectralBoundaryCondition):
    kind = "fast-spectral-inlet"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ic = MaxwellianInitialCondition(self._cfg, self._vmesh)
        self._f0 = ic.init_vals
        self._u = ic.u
        _ = self.trace_ids

    @cached_property
    def trace_ids(self):
        nl = self.normals
        return torch.tensordot(
            nl, self.vpoints[:2, ...] - self._u[:2, ...], dims=1
        )

    @override
    @torch.compile
    def apply(
        self,
        curr_time: torch.float64,
        ul: torch.Tensor,
    ):
        nu = self.trace_ids
        return torch.where(nu >= 0, ul, self._f0.unsqueeze(0))


def get_boundary_condition(cfg, name, *args, **kwargs):
    return get_bc_by_cls_and_name(
        cfg, name, FastSpectralBoundaryCondition, *args, **kwargs
    )

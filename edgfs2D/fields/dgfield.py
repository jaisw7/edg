from abc import ABCMeta
from functools import cached_property

import numpy as np
import torch
from typing_extensions import Dict

from edgfs2D.boundary_conditions.base import BaseBoundaryCondition
from edgfs2D.fields.types import FieldData, FieldDataTuple
from edgfs2D.fluxes.base import BaseFlux
from edgfs2D.initial_conditions.base import BaseInitialCondition
from edgfs2D.physical_mesh.dg_mesh import DgMesh
from edgfs2D.utils.dictionary import Dictionary
from edgfs2D.utils.util import to_torch, torch_map

np.set_printoptions(suppress=True, linewidth=2000, precision=2)


class DgField(object, metaclass=ABCMeta):
    kind = None
    """Defines a discontinous Galerkin field"""

    def __init__(self, cfg: Dictionary, dgmesh: DgMesh, nvars: int):
        self.cfg = cfg
        self.dgmesh = dgmesh
        self.nvars = nvars

    def _alloc_torch(self, size):
        return torch.empty(
            [*size, self.nvars],
            dtype=self.cfg.ttype,
            device=self.cfg.device,
        )

    def _to_device(self, data: FieldData):
        return {
            shape: to_torch(d).to(dtype=self.cfg.ttype, device=self.cfg.device)
            for shape, d in data.items()
        }

    def _apply_initial_condition(self, u: FieldData, ic: BaseInitialCondition):
        for shape, x in self.dgmesh.get_element_nodes.items():
            u[shape].copy_(torch.from_numpy(ic.apply(x)))

    @cached_property
    def _element_jac_mat(self):
        return self._to_device(self.dgmesh.get_element_jacobian_mat)

    @cached_property
    def _element_nodes(self):
        return self._to_device(self.dgmesh.get_element_nodes)

    @cached_property
    def _internal_interface_normals(self):
        return (
            self._to_device(self.dgmesh.get_internal_interface_normals[0]),
            self._to_device(self.dgmesh.get_internal_interface_normals[1]),
        )

    @cached_property
    def _internal_interface_scaled_jacobian_det(self):
        return (
            self._to_device(
                self.dgmesh.get_internal_interface_scaled_jacobian_det[0]
            ),
            None,
        )

    def create_new_field(self) -> FieldData:
        element_data = FieldData()
        for shape, size in self.dgmesh.get_element_data_sizes.items():
            element_data[shape] = self._alloc_torch(size)
        return element_data

    def grad(self, u: FieldData) -> FieldData:
        grad_data = FieldData()
        element_jac = self._element_jac_mat
        for shape, basis in self.dgmesh.get_basis_at_shapes.items():
            grad_data[shape] = basis.grad(u[shape], element_jac[shape])
        return grad_data

    def surface_data(self, u: FieldData) -> FieldData:
        surface_data = FieldData()
        for shape, basis in self.dgmesh.get_basis_at_shapes.items():
            surface_data[shape] = basis.surface_data(u[shape])
        return surface_data

    @property
    def _get_shape(self):
        assert (
            len(self.dgmesh.get_basis_at_shapes.keys()) == 1
        ), "Only one type of element supported as of now"
        return next(iter(self.dgmesh.get_basis_at_shapes.keys()))

    def traces(self, uf: FieldData) -> FieldDataTuple:
        shape = self._get_shape
        trace_lhs, trace_rhs = FieldData(), FieldData()
        lhs, rhs = self.dgmesh.get_internal_interfaces
        for key, (fids, eids) in lhs.items():
            trace_lhs[key] = uf[shape][fids, eids, ...]
        for key, (fids, eids) in rhs.items():
            trace_rhs[key] = uf[shape][fids, eids, ...]
        return (trace_lhs, trace_rhs)

    def _apply_boundary_condition(
        self, uf: FieldData, bc: Dict[str, BaseBoundaryCondition]
    ):
        shape = self._get_shape
        trace_bnd = FieldData()
        bnd = self.dgmesh.get_boundary_interface
        for bid, (fids, eids) in bnd.items():
            trace_bnd[bid] = bc[shape].apply(uf[shape][fids, eids, ...])

        return trace_bnd

    def _compute_flux(self, ul: FieldData, ur: FieldData, flux: BaseFlux):
        nl, _ = self._internal_interface_normals
        for shape in ul.keys():
            flux.apply(ul[shape], ur[shape], nl[shape])

    def _convect(self, gradu: FieldData, velocity: torch.Tensor) -> FieldData:
        data = FieldData()
        for shape, basis in self.dgmesh.get_basis_at_shapes.items():
            data[shape] = basis.convect(gradu[shape], velocity)
        return data

    def _lift_jump(
        self,
        fl: FieldData,
        fr: FieldData,
        uf: FieldData,
        velocity: torch.Tensor,
        out: torch.Tensor,
    ) -> FieldData:
        pe = lambda v: print(v[next(iter(v.keys()))].squeeze().numpy())
        pex = lambda v: pe(v) + exit(0)

        shape = self._get_shape
        lhs, rhs = self.dgmesh.get_internal_interfaces
        nl, nr = self._internal_interface_normals
        basis = self.dgmesh.get_basis_at_shapes[shape]
        sdetl, _ = self._internal_interface_scaled_jacobian_det

        # print(torch.norm(fl[next(iter(fl.keys()))] + fr[next(iter(fr.keys()))]))

        # pe(fl)
        # pe(fr)

        # pe(nl)

        for key, (fids, eids) in lhs.items():
            # print(uf[shape][fids, eids, ...].squeeze().numpy())
            # print(fl[key].squeeze().numpy())
            # print(torch.tensordot(nl[key], velocity, dims=1).squeeze().numpy())
            # print(
            #     (
            #         torch.tensordot(nl[key], velocity, dims=1).unsqueeze(-1)
            #         * uf[shape][fids, eids, ...]
            #     )
            #     .squeeze()
            #     .numpy()
            # )
            fl[key] = (
                torch.tensordot(nl[key], velocity, dims=1).unsqueeze(-1)
                * uf[shape][fids, eids, ...]
                - fl[key]
            )

        for key, (fids, eids) in rhs.items():
            # print(uf[shape][fids, eids, ...].squeeze().numpy())
            # print(fr[key].squeeze().numpy())
            fr[key] = (
                torch.tensordot(nr[key], velocity, dims=1).unsqueeze(-1)
                * uf[shape][fids, eids, ...]
                - fr[key]
            )

        # print(torch.norm(fl[next(iter(fl.keys()))] + fr[next(iter(fr.keys()))]))

        # pe(fl)
        # pex(fr)

        for key, (fids, eids) in lhs.items():
            uf[shape][fids, eids, ...] = fl[key] * sdetl[key].unsqueeze(-1)

        for key, (fids, eids) in rhs.items():
            uf[shape][fids, eids, ...] = fr[key] * sdetl[key].unsqueeze(-1)

        out[shape].add_(basis.lift(uf[shape]))

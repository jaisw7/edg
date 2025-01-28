from abc import ABCMeta
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from typing_extensions import Dict

from edgfs2D.boundary_conditions.base import BaseBoundaryCondition
from edgfs2D.fields.readers.h5 import H5FieldReader
from edgfs2D.fields.types import FieldData, FieldDataTuple, Shape
from edgfs2D.fields.writers.h5 import H5FieldWriter
from edgfs2D.fluxes.base import BaseFlux
from edgfs2D.initial_conditions.base import BaseInitialCondition
from edgfs2D.physical_mesh.dg_mesh import DgMesh
from edgfs2D.utils.dictionary import Dictionary
from edgfs2D.utils.util import to_torch, torch_map


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
        return FieldData(
            {
                shape: to_torch(d).to(
                    dtype=self.cfg.ttype, device=self.cfg.device
                )
                for shape, d in data.items()
            }
        )

    def _to_device_interfaces(self, data: FieldData):
        return {
            shape: (
                torch.from_numpy(fid).to(device=self.cfg.device),
                torch.from_numpy(eid).to(device=self.cfg.device),
            )
            for shape, (fid, eid) in data.items()
        }

    def _apply_initial_condition(self, u: FieldData, ic: BaseInitialCondition):
        for shape, x in self.dgmesh.get_element_nodes.items():
            u[shape].copy_(torch.from_numpy(ic.apply(x)))

    @cached_property
    def _element_jac_mat(self):
        return self._to_device(self.dgmesh.get_element_jacobian_mat)

    @cached_property
    def _element_jac_det(self):
        return self._to_device(self.dgmesh.get_element_jacobian_det)

    @cached_property
    def _element_nodes(self):
        return self._to_device(self.dgmesh.get_element_nodes)

    @cached_property
    def _internal_interfaces(self):
        return (
            self._to_device_interfaces(self.dgmesh.get_internal_interfaces[0]),
            self._to_device_interfaces(self.dgmesh.get_internal_interfaces[1]),
        )

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

    @cached_property
    def _boundary_interfaces(self):
        return self._to_device_interfaces(self.dgmesh.get_boundary_interfaces)

    @cached_property
    def _boundary_interface_nodes(self):
        return self._to_device(self.dgmesh.get_boundary_interface_nodes)

    @cached_property
    def _boundary_interface_normals(self):
        return self._to_device(self.dgmesh.get_boundary_interface_normals)

    @cached_property
    def _boundary_interface_scaled_jacobian_det(self):
        return self._to_device(
            self.dgmesh.get_boundary_interface_scaled_jacobian_det
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

    def error(self, u: FieldData):
        error_data = {}
        element_jac_det = self._element_jac_det
        for shape, basis in self.dgmesh.get_basis_at_shapes.items():
            error_data[shape] = basis.error(u[shape], element_jac_det[shape])
        return error_data

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

    def _internal_traces(self, uf: FieldData) -> FieldDataTuple:
        shape = self._get_shape
        trace_lhs, trace_rhs = FieldData(), FieldData()
        lhs, rhs = self.dgmesh.get_internal_interfaces
        for key, (fids, eids) in lhs.items():
            trace_lhs[key] = uf[shape][fids, eids, ...]
        for key, (fids, eids) in rhs.items():
            trace_rhs[key] = uf[shape][fids, eids, ...]
        return (trace_lhs, trace_rhs)

    def _boundary_traces(
        self,
        curr_time: torch.float64,
        uf: FieldData,
        bc: Dict[str, BaseBoundaryCondition],
    ) -> FieldDataTuple:
        shape = self._get_shape
        trace_lhs, trace_rhs = FieldData(), FieldData()
        for key, (fids, eids) in self.dgmesh.get_boundary_interfaces.items():
            trace_lhs[key] = uf[shape][fids, eids, ...]
            trace_rhs[key] = bc[key].apply(curr_time, trace_lhs[key])
        return (trace_lhs, trace_rhs)

    def _compute_flux(
        self, ul: FieldData, ur: FieldData, nl: FieldData, flux: BaseFlux
    ):
        for key in ul.keys():
            flux.apply(ul[key], ur[key], nl[key])

    def _convect(self, gradu: FieldData, velocity: torch.Tensor) -> FieldData:
        data = FieldData()
        for shape, basis in self.dgmesh.get_basis_at_shapes.items():
            data[shape] = basis.convect(gradu[shape], velocity)
        return data

    def _add_flux(self, uf, velocity, interface, nl, fl, sdetl):
        shape = self._get_shape
        for key, (fids, eids) in interface.items():
            dim = min(nl[key].shape[1], velocity.shape[0])
            uf[shape][fids, eids, ...] = (
                torch.tensordot(nl[key], velocity[:dim], dims=1)
                * uf[shape][fids, eids, ...]
                - fl[key]
            ) * sdetl[key].unsqueeze(-1)

    def _lift_jump(
        self,
        fl: FieldData,
        fr: FieldData,
        fb: FieldData,
        uf: FieldData,
        velocity: torch.Tensor,
        out: torch.Tensor,
    ) -> FieldData:

        shape = self._get_shape
        lhs, rhs = self._internal_interfaces
        nl, nr = self._internal_interface_normals
        basis = self.dgmesh.get_basis_at_shapes[shape]
        sdetl, _ = self._internal_interface_scaled_jacobian_det

        bnd = self._boundary_interfaces
        xb = self._boundary_interface_nodes
        nb = self._boundary_interface_normals
        sdetb = self._boundary_interface_scaled_jacobian_det

        self._add_flux(uf, velocity, lhs, nl, fl, sdetl)
        self._add_flux(uf, velocity, rhs, nr, fr, sdetl)
        self._add_flux(uf, velocity, bnd, nb, fb, sdetb)

        out[shape].add_(basis.lift(uf[shape]))

    def write_metadata(self, path: Path):
        writer = H5FieldWriter(path)
        writer.write_metadata("uuid", self.dgmesh.uuid)
        writer.write_metadata("time", self.dgmesh.time.time)
        return writer

    def read_field(self, path: Path, field_name: str) -> FieldData:
        reader = H5FieldReader(path)
        return self._to_device(reader.read_field(field_name))

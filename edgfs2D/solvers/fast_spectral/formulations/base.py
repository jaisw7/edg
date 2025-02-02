from abc import abstractmethod
from pathlib import Path

import torch
from typing_extensions import override

from edgfs2D.distribution_mesh.dgdist_mesh import DgDistMesh
from edgfs2D.fields.types import FieldData, FieldDataList
from edgfs2D.scattering import get_scattering_model
from edgfs2D.solvers.base import BaseSolver, MomentMixin
from edgfs2D.solvers.fast_spectral.create_fields import FsField
from edgfs2D.solvers.fast_spectral.moments import Moments
from edgfs2D.time.physical_time import PhysicalTime
from edgfs2D.utils.dictionary import Dictionary


class BaseFormulation(BaseSolver, MomentMixin):
    formulation = None

    def __init__(self, cfg: Dictionary, distmesh: DgDistMesh):
        self._cfg = cfg
        self._time = time = distmesh.dgmesh.time
        self._distmesh = distmesh
        self._fs = FsField(cfg, distmesh)

        # tensors for storage of solutions
        self._u0 = self._fs.create_new_field()
        self._u1 = self._fs.create_new_field()

        # apply initial condition
        if time.is_restart:
            self._u1 = self._fs.read_field(time.args.soln, self.field_name)
        else:
            self._fs.apply_initial_condition(self._u1)

        # scattering model
        self._sm = get_scattering_model(cfg, distmesh)
        if type(self).__name__ not in self._sm.allowed_solvers:
            raise RuntimeError(
                f"Scattering model not supported by {type(self).__name__}"
            )

        # moments
        self._moments = Moments(self._distmesh.vmesh)

        # load plugins
        self._time.load_plugins(self)

    def eval_derivate(self, u: FieldData):
        fs = self._fs

        if fs.is_eflux_enabled:
            eflux = fs.compute_entropy_flux(u, u)
            conv = fs.convect_eflux(eflux)
        else:
            gradu = fs.grad(u)
            conv = fs.convect(gradu)

        return conv

    def transport(self, curr_time: torch.float64, u: FieldData):
        fs = self._fs

        # compute convective derivative
        conv = self.eval_derivate(u)
        conv.mul_(-1)

        # traces at boundaries
        uf = fs.surface_data(u)
        ul, ur = fs.internal_traces(uf)
        ulb, urb = fs.boundary_traces(curr_time, uf)

        # fluxes
        fs.compute_internal_flux(ul, ur)
        fs.compute_boundary_flux(ulb, urb)

        # lift
        fs.lift_jump(ul, ur, ulb, uf, conv)

        return conv

    def rhs(self, curr_time: torch.float64, u: FieldData):
        conv = self.transport(curr_time, u)

        # add scattering
        for shape in u.keys():
            self._sm.solve(curr_time, u[shape], out=conv[shape])

        return conv

    @abstractmethod
    def solve(self):
        pass

    @override
    @property
    def curr_fields(self) -> FieldDataList:
        return [self._u1]

    @override
    @property
    def prev_fields(self) -> FieldDataList:
        return [self._u0]

    @override
    @property
    def time(self) -> PhysicalTime:
        return self._time

    @override
    @property
    def mesh(self) -> DgDistMesh:
        return self._distmesh

    @override
    def error_norm(self, err: FieldData):
        return self._fs.error(err)

    @override
    def write(self, path: Path):
        writer = self._fs.write_metadata(path)
        writer.write_fields(FieldData({self.field_name: self.curr_fields[0]}))

    @override
    def write_moment(self, path: Path):
        soln = self.curr_fields[0]
        fields = self._moments.fields
        writer = self._fs.write_metadata(path)

        moments = FieldData()
        for shape in soln.keys():
            moments[shape] = self._moments(soln[shape])

        for id, name in enumerate(fields):
            field = FieldData()
            for shape in moments.keys():
                field[shape] = moments[shape][..., id]
            writer.write_fields(FieldData({name: field}))

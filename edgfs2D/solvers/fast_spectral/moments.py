from functools import lru_cache

import torch
from loguru import logger

from edgfs2D.velocity_mesh.base import BaseVelocityMesh


class Moments:
    """Moments for single-species systems"""

    fields = [
        "rho",
        "U:x",
        "U:y",
        "T",
        "Q:x",
        "Q:y",
        "P:xx",
        "P:yy",
        "P:xy",
        "p",
    ]

    @lru_cache(maxsize=None)
    def vpoints(self, dtype, device):
        return torch.from_numpy(self._vmesh.points).to(
            dtype=dtype, device=device
        )

    def __call__(self, soln: torch.Tensor):
        vm = self._vmesh
        cv = self.vpoints(soln.dtype, soln.device)
        Nv = vm.num_points
        cw = vm.weights
        T0 = vm.nondim.T0
        rho0 = vm.nondim.rho0
        molarMass0 = vm.nondim.molarMass0
        u0 = vm.nondim.u0
        mcw = cw

        Nqr, Ne = soln.shape[:2]
        Ns = len(self.fields)
        ele_sol = torch.zeros(
            (Nqr, Ne, Ns), dtype=soln.dtype, device=soln.device
        )

        # non-dimensional mass density
        ele_sol[..., 0] = soln.sum(dim=-1) * mcw

        if torch.any(torch.lt(ele_sol[..., 0], 1e-10)):
            logger.warn("density below 1e-10")
            return

        # non-dimensional velocities
        ele_sol[:, :, 1] = torch.tensordot(soln, cv[0, :], dims=1) * mcw
        ele_sol[:, :, 1] /= ele_sol[:, :, 0]
        ele_sol[:, :, 2] = torch.tensordot(soln, cv[1, :], dims=1) * mcw
        ele_sol[:, :, 2] /= ele_sol[:, :, 0]

        # peculiar velocity
        cx = cv[0, :].reshape((1, 1, Nv)) - ele_sol[:, :, 1].reshape(
            (Nqr, Ne, 1)
        )
        cy = cv[1, :].reshape((1, 1, Nv)) - ele_sol[:, :, 2].reshape(
            (Nqr, Ne, 1)
        )
        cz = cv[2, :].reshape((1, 1, Nv)) - torch.zeros((Nqr, Ne, 1))
        cSqr = cx * cx + cy * cy + cz * cz

        # non-dimensional temperature
        ele_sol[:, :, 3] = torch.einsum("...j,...j->...", soln, cSqr) * (
            2 * mcw / 3
        )
        ele_sol[:, :, 3] /= ele_sol[:, :, 0]

        # non-dimensional heat-flux
        ele_sol[:, :, 4] = (
            torch.einsum("...j,...j,...j->...", soln, cSqr, cx) * mcw
        )
        ele_sol[:, :, 5] = (
            torch.einsum("...j,...j,...j->...", soln, cSqr, cy) * mcw
        )

        # non-dimensional pressure-tensor components
        ele_sol[:, :, 6] = (
            2 * torch.einsum("...j,...j,...j->...", soln, cx, cx) * mcw
        )
        ele_sol[:, :, 7] = (
            2 * torch.einsum("...j,...j,...j->...", soln, cy, cy) * mcw
        )
        ele_sol[:, :, 8] = (
            2 * torch.einsum("...j,...j,...j->...", soln, cx, cy) * mcw
        )

        # dimensional rho, ux, uy, T, qx, qy, Pxx, Pyy, Pxy
        ele_sol[:, :, 0:9] *= torch.Tensor(
            [
                rho0,
                u0,
                u0,
                T0,
                0.5 * rho0 * (u0**3),
                0.5 * rho0 * (u0**3),
                0.5 * rho0 * (u0**2),
                0.5 * rho0 * (u0**2),
                0.5 * rho0 * (u0**2),
            ]
        ).reshape(1, 1, 9)

        # dimensional pressure
        ele_sol[:, :, 9] = (
            (vm.nondim.R0 / molarMass0) * ele_sol[:, :, 0] * ele_sol[:, :, 3]
        )

        return ele_sol

    def __init__(self, vmesh: BaseVelocityMesh):
        self._vmesh = vmesh

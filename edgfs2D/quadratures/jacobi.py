from collections.abc import Iterable
from math import gamma

import numpy as np


# The following modules are based on the "PyFR" implementation
# (See licences/LICENSE_PyFR)
def jacobi(n, a, b, z):
    j = [1]

    if n >= 1:
        j.append(((a + b + 2) * z + a - b) / 2)
    if n >= 2:
        apb, bbmaa = a + b, b * b - a * a

        for q in range(2, n + 1):
            qapbpq, apbp2q = q * (apb + q), apb + 2 * q
            apbp2qm1, apbp2qm2 = apbp2q - 1, apbp2q - 2

            aq = apbp2q * apbp2qm1 / (2 * qapbpq)
            bq = apbp2qm1 * bbmaa / (2 * qapbpq * apbp2qm2)
            cq = apbp2q * (a + q - 1) * (b + q - 1) / (qapbpq * apbp2qm2)

            # Update
            j.append((aq * z - bq) * j[-1] - cq * j[-2])

    return j


def jacobi_diff(n, a, b, z):
    dj = [0]

    if n >= 1:
        dj.extend(
            jp * (i + a + b + 2) / 2
            for i, jp in enumerate(jacobi(n - 1, a + 1, b + 1, z))
        )

    return dj


def ortho_basis_at_py(order, p):
    jp = jacobi(order - 1, 0, 0, p)
    return [np.sqrt(i + 0.5) * p for i, p in enumerate(jp)]


def jac_ortho_basis_at_py(order, p):
    djp = jacobi_diff(order - 1, 0, 0, p)
    return [(np.sqrt(i + 0.5) * p,) for i, p in enumerate(djp)]


def ortho_basis_at(order, pts):
    if len(pts) and not isinstance(pts[0], Iterable):
        pts = [(p,) for p in pts]
        return np.array([ortho_basis_at_py(order, *p) for p in pts]).T


def jac_ortho_basis_at(order, pts):
    if len(pts) and not isinstance(pts[0], Iterable):
        pts = [(p,) for p in pts]
    J = [jac_ortho_basis_at_py(order, *p) for p in pts]
    return np.array(J).swapaxes(0, 2)


def nodal_basis_at(order, pts, epts):
    return np.linalg.solve(
        ortho_basis_at(order, pts), ortho_basis_at(order, epts)
    ).T


def jac_nodal_basis_at(order, pts, epts):
    return np.linalg.solve(
        ortho_basis_at(order, pts), jac_ortho_basis_at(order, epts)
    )


def njacobi1(n, a, b, z):
    j0 = np.ones_like(z) * np.sqrt(
        pow(2, -a - b - 1) * gamma(a + b + 2) / gamma(a + 1) / gamma(b + 1)
    )
    j1 = j0

    if n >= 1:
        j1 = (
            0.5
            * j0
            * np.sqrt((a + b + 3) / (a + 1) / (b + 1))
            * ((a + b + 2) * z + (a - b))
        )

    if n >= 2:
        for q in range(2, n + 1):
            c1 = (
                2.0
                / (2 * (q - 1) + a + b)
                * np.sqrt(
                    (q - 1)
                    * ((q - 1) + a + b)
                    * ((q - 1) + a)
                    * ((q - 1) + b)
                    / (2 * (q - 1) + a + b - 1)
                    / (2 * (q - 1) + a + b + 1)
                )
            )
            c2 = (
                2.0
                / (2 * (q) + a + b)
                * np.sqrt(
                    (q)
                    * ((q) + a + b)
                    * ((q) + a)
                    * ((q) + b)
                    / (2 * (q) + a + b - 1)
                    / (2 * (q) + a + b + 1)
                )
            )
            c3 = (
                -(a**2 - b**2)
                / (2 * (q - 1) + a + b)
                / (2 * (q - 1) + a + b + 2)
            )
            j2 = (j1 * (z - c3) - c1 * j0) / c2
            j0, j1 = j1, j2

    return j1


def njacobi1_diff(n, a, b, z):
    if n == 0:
        return np.zeros_like(z)
    else:
        return np.sqrt(n * (n + a + b + 1)) * njacobi1(n - 1, a + 1, b + 1, z)


def tri_northo_basis(a, b, i, j):
    return (
        np.sqrt(2.0)
        * njacobi1(i, 0.0, 0.0, a)
        * njacobi1(j, 2 * i + 1, 0.0, b)
        * ((1 - b) ** i)
    )


def tri_jac_northo_basis(a, b, i, j, VSMALL=1e-20):
    Pa, Pb = njacobi1(i, 0.0, 0.0, a), njacobi1(j, 2 * i + 1, 0.0, b)
    dPa, dPb = njacobi1_diff(i, 0.0, 0.0, a), njacobi1_diff(
        j, 2 * i + 1, 0.0, b
    )

    # da/dr dphi/da = (2./(1-b)) * (sqrt(2)*(1-b)^i) * dPa * Pb
    dBa = np.sqrt(2.0) * dPa * Pb * ((1 - b) ** i) * (2.0 / (1 - b + VSMALL))
    if i == 1:
        dBa = np.sqrt(2.0) * dPa * Pb * 2.0

    # da/ds dphi/da + dphi/db = (1+a)/(1-b) * dphi/da + dphi/db
    dBb = (
        np.sqrt(2.0) * dPa * Pb * ((1 - b) ** i) * (1 + a) / ((1 - b + VSMALL))
    )
    if i == 1:
        dBb = np.sqrt(2.0) * dPa * Pb * (1 + a)

    # dphi/db = (sqrt(2)*(1-b)^i)*Pa*dPb - i*(sqrt(2)*(1-b)^{i-1})*Pa*Pb
    dBb += np.sqrt(2.0) * Pa * dPb * ((1 - b) ** i)
    if i > 0:
        dBb += -np.sqrt(2.0) * Pa * Pb * ((1 - b) ** (i - 1)) * i

    return dBa, dBb


"""Computation of gauss quadratures via eigenvalue decomposition.
Ref: Orthogonal Polynomials: Computation and Approximation, Walter Gautschi"""


def rjacobi(n, a, b):
    ra, rb = np.zeros(n), np.zeros(n)

    apbp2 = 2.0 + a + b
    ra[0] = (b - a) / apbp2
    rb[0] = np.power(2.0, a + b + 1) * (
        gamma(a + 1.0) * gamma(b + 1.0) / gamma(apbp2)
    )
    rb[1] = 4.0 * (a + 1.0) * (b + 1.0) / ((apbp2 + 1.0) * apbp2 * apbp2)

    # Compute other terms
    apbp2 += 2
    for i in range(1, n - 1):
        ra[i] = (b * b - a * a) / ((apbp2 - 2.0) * apbp2)
        rb[i + 1] = (
            4.0
            * (i + 1)
            * (i + 1 + a)
            * (i + 1 + b)
            * (i + 1 + a + b)
            / ((apbp2 * apbp2 - 1) * apbp2 * apbp2)
        )
        apbp2 += 2

    ra[n - 1] = (b * b - a * a) / ((apbp2 - 2.0) * apbp2)
    return ra, rb


def gauss(n, ra, rb):
    scal = rb[0]

    rb[:-1] = np.sqrt(rb[1:])
    z, V = np.linalg.eigh(np.diag(ra) + np.diag(rb[:-1], -1))
    zidx = np.argsort(z)
    z.sort()
    V = V[:, zidx]

    w = V[0, :]
    w = scal * (w**2)
    return z, w


def zwgj(n, a, b):
    ra, rb = rjacobi(n, a, b)
    z, w = gauss(n, ra, rb)
    return z, w


def zwglj(n, a, b):
    N = n - 2
    z, w = rjacobi(n, a, b)

    apb1 = a + b + 1.0
    z[n - 1] = (a - b) / (2.0 * N + apb1 + 1.0)
    w[n - 1] = (
        4.0
        * (N + a + 1.0)
        * (N + b + 1.0)
        * (N + apb1)
        / ((2.0 * N + apb1) * np.power(2 * N + apb1 + 1, 2.0))
    )

    z, w = gauss(n, z, w)
    return z, w

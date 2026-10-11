"""Shared setups for the Euler / 3-D gates (scratch, not repo tests).

Exact Riemann solver: Toro, Riemann Solvers and Numerical Methods for Fluid
Dynamics, 3rd ed. (2009), ch. 4 (pressure function 4.5-4.7, Newton on p*,
sampling 4.4).  Written here from the book; no third-party code.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.fvm.riemann_solvers import HLL, Rusanov
from zoomy_core.systemmodel.euler import compressible_euler

GAMMA = 1.4
TAGS = {1: ("left", "right"), 2: ("left", "right", "bottom", "top"),
        3: ("left", "right", "front", "back", "bottom", "top")}
# opposite tag of each boundary, for periodic pairs
OPP = {"left": "right", "right": "left", "front": "back", "back": "front",
       "bottom": "top", "top": "bottom"}


# ── exact Riemann solver (Toro ch. 4) ───────────────────────────────────────
def exact_riemann(WL, WR, xi, g=GAMMA):
    """Primitive (rho, u, p) of the exact solution at x/t = ``xi``."""
    rL, uL, pL = WL
    rR, uR, pR = WR
    cL, cR = np.sqrt(g * pL / rL), np.sqrt(g * pR / rR)
    AL, AR = 2 / ((g + 1) * rL), 2 / ((g + 1) * rR)
    BL, BR = (g - 1) / (g + 1) * pL, (g - 1) / (g + 1) * pR

    def f(p, pk, rk, ck, Ak, Bk):
        if p > pk:                                   # shock
            q = np.sqrt(Ak / (p + Bk))
            return (p - pk) * q, q * (1 - (p - pk) / (2 * (Bk + p)))
        e = (p / pk) ** ((g - 1) / (2 * g))          # rarefaction
        return (2 * ck / (g - 1) * (e - 1),
                1 / (rk * ck) * (p / pk) ** (-(g + 1) / (2 * g)))
    p = max(1e-8, 0.5 * (pL + pR) - 0.125 * (uR - uL) * (rL + rR) * (cL + cR))
    for _ in range(100):
        fL, dL = f(p, pL, rL, cL, AL, BL)
        fR, dR = f(p, pR, rR, cR, AR, BR)
        dp = (fL + fR + uR - uL) / (dL + dR)
        p_new = max(1e-12, p - dp)
        if abs(p_new - p) < 1e-14 * (p_new + p):
            p = p_new
            break
        p = p_new
    us = 0.5 * (uL + uR) + 0.5 * (f(p, pR, rR, cR, AR, BR)[0]
                                  - f(p, pL, rL, cL, AL, BL)[0])
    xi = np.atleast_1d(np.asarray(xi, float))
    out = np.zeros((3, xi.size))
    gm = (g - 1) / (g + 1)
    for i, s in enumerate(xi):
        if s <= us:                                  # left of the contact
            if p > pL:
                SL = uL - cL * np.sqrt((g + 1) / (2 * g) * p / pL + (g - 1) / (2 * g))
                out[:, i] = (WL if s <= SL else
                             (rL * (p / pL + gm) / (gm * p / pL + 1), us, p))
            else:
                cs = cL * (p / pL) ** ((g - 1) / (2 * g))
                if s <= uL - cL:
                    out[:, i] = WL
                elif s >= us - cs:
                    out[:, i] = (rL * (p / pL) ** (1 / g), us, p)
                else:
                    c = 2 / (g + 1) * (cL + (g - 1) / 2 * (uL - s))
                    out[:, i] = (rL * (c / cL) ** (2 / (g - 1)),
                                 2 / (g + 1) * (cL + (g - 1) / 2 * uL + s),
                                 pL * (c / cL) ** (2 * g / (g - 1)))
        else:
            if p > pR:
                SR = uR + cR * np.sqrt((g + 1) / (2 * g) * p / pR + (g - 1) / (2 * g))
                out[:, i] = (WR if s >= SR else
                             (rR * (p / pR + gm) / (gm * p / pR + 1), us, p))
            else:
                cs = cR * (p / pR) ** ((g - 1) / (2 * g))
                if s >= uR + cR:
                    out[:, i] = WR
                elif s <= us + cs:
                    out[:, i] = (rR * (p / pR) ** (1 / g), us, p)
                else:
                    c = 2 / (g + 1) * (cR - (g - 1) / 2 * (uR - s))
                    out[:, i] = (rR * (c / cR) ** (2 / (g - 1)),
                                 2 / (g + 1) * (-cR + (g - 1) / 2 * uR + s),
                                 pR * (c / cR) ** (2 * g / (g - 1)))
    return out


SOD = ((1.0, 0.0, 1.0), (0.125, 0.0, 0.1))


def conserved(rho, u, p, g=GAMMA):
    """(rho, rho u_d, E) from primitive rho, velocity vector u (dim, ...), p."""
    u = np.asarray(u, float)
    return np.concatenate([[rho], rho * u,
                           [p / (g - 1) + 0.5 * rho * (u ** 2).sum(axis=0)]])


def primitive(Q, g=GAMMA):
    rho = Q[0]
    u = Q[1:-1] / rho
    p = (g - 1) * (Q[-1] - 0.5 * rho * (u ** 2).sum(axis=0))
    return rho, u, p


def nsm_of(sm, order, riemann=HLL, variables="model", limiter="minmod"):
    return NumericalSystemModel.from_system_model(
        sm, riemann=riemann,
        reconstruction=ReconstructionSpec(order=order, limiter=limiter,
                                          variables=variables))


def box(dim, n, axis=0, length=1.0, thin=2):
    """Mesh: ``n`` cells of ``length`` along ``axis``, ``thin`` cells of the
    same size across (``dim`` = 1: a line)."""
    if dim == 1:
        return LSQMesh.create_1d(domain=(0.0, length), n_inner_cells=n)
    h = length / n
    ext = [(0.0, thin * h)] * dim
    ext[axis] = (0.0, length)
    cnt = [thin] * dim
    cnt[axis] = n
    dom = tuple(v for e in ext for v in e)
    if dim == 2:
        return LSQMesh.create_2d(domain=dom, nx=cnt[0], ny=cnt[1])
    return LSQMesh.create_3d(domain=dom, nx=cnt[0], ny=cnt[1], nz=cnt[2])


def sod_system(dim, axis=0, x0=0.5, bc="extrapolation"):
    """Sod tube along ``axis``; Extrapolation (or Wall) on every tag."""
    def ic(x):
        W = SOD[0] if x[axis] < x0 else SOD[1]
        u = np.zeros(dim)
        return conserved(W[0], u, W[2])
    B = BC.Wall if bc == "wall" else BC.Extrapolation
    return compressible_euler(
        dim, GAMMA, boundary_conditions=BC.BoundaryConditions(
            [B(tag=t) for t in TAGS[dim]]),
        initial_conditions=IC.UserFunction(function=ic))


def periodic_bcs(dim, open_tags=()):
    """Periodic on every tag pair; ``open_tags`` get Extrapolation."""
    bcs = []
    for t in TAGS[dim]:
        if t in open_tags:
            bcs.append(BC.Extrapolation(tag=t))
        else:
            bcs.append(BC.Periodic(tag=t, periodic_to_physical_tag=OPP[t]))
    return BC.BoundaryConditions(bcs)


def wave_exact(x, t, vel, amp=0.2, rho0=1.0):
    """rho = rho0 + amp sin(2 pi k.(x - vel t)), k = (1, ..., 1)."""
    s = sum(np.asarray(x[d]) - vel[d] * t for d in range(len(vel)))
    return rho0 + amp * np.sin(2 * np.pi * s)


def wave_system(dim, vel, p0=1.0, bcs=None):
    def ic(x):
        return conserved(wave_exact(x, 0.0, vel), np.asarray(vel), p0)
    return compressible_euler(
        dim, GAMMA, boundary_conditions=bcs or periodic_bcs(dim),
        initial_conditions=IC.UserFunction(function=ic))


def cell_volumes(mesh):
    return np.asarray(mesh.cell_volumes)[:int(mesh.n_inner_cells)]


def cell_centers(mesh):
    return np.asarray(mesh.cell_centers)[:, :int(mesh.n_inner_cells)]

"""SERGHEI LowerTriangle catchment, 10 m, on the symbolic AMReX solver (and, for
the comparison, the symbolic JAX solver on CPU): the physics of
bench_catchment.py (RainSWE, 30 mm/h, Manning n = 0.03, PositiveHLL, order 1,
linearized source, dt_max 5 s, CFL 0.5, walls), started from its coarse 1 h
state.

    python catchment_amrex.py OUT T CFG        CFG: jax | cpu1 | cpu4 | gpu1 | gpu2 | gpu4

AMReX needs a Cartesian box: the full 555 x 526 raster is meshed, the nodata
cells (-9999) get the bed b_wall = 1e4, which RainSWE reads as a reflective
no-rain wall.  jax -> OUT/jax_T.npz, AMReX -> OUT/amrex_CFG_T.npz (h on the
raster, NaN off the catchment).  AMReX prints one 'catchment ...' line;
the JAX comparison is made when OUT/jax_T.npz exists."""
import os
import sys
import time as clock
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.join(HERE, "../../../../thesis/cases/river/serghei")
sys.path.insert(0, os.path.join(CASE, "amrex"))
from rain_swe import RainSWE                                    # noqa: E402

R_O = 30.0 / 1000.0 / 3600.0                                    # 30 mm/h
B_WALL = 1.0e4
NODATA = -9999.0
DEM = os.path.join(CASE, "run/LowerTriangle_LowRes/dem_array.npy")
COARSE = os.environ.get("CATCH_COARSE", "/tmp/claude-765404697/-mnt-userdrive-Users-home-"
                        "adam-obbpb5az1dhsjzf-git-zotero-rag/b5667895-053e-413b-8d69-"
                        "8137a92fc71c/scratchpad/catch/coarse.npz")
TAGS = ("left", "right", "bottom", "top")                      # LSQMesh.create_2d


def nsm_of():
    """bench_catchment.nsm_of(), with the four tags of create_2d for its wall."""
    import zoomy_core.model.initial_conditions as IC
    from zoomy_core.fvm.riemann_solvers import PositiveHLL
    from zoomy_core.model.boundary_conditions import BoundaryConditions, Wall
    from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec, SourceTreatment
    from zoomy_core.systemmodel import SystemModel
    from zoomy_core.systemmodel.operations import guard_eigenvalue_powers
    bcs = BoundaryConditions([Wall(tag=t, momentum_field_indices=[[2, 3]],
                                   permeability=0.0, wall_slip=1.0) for t in TAGS])
    model = RainSWE(dimension=2, parameters={"g": 9.81, "n_m": 0.03, "r_o": R_O,
                                             "b_wall": B_WALL, "wet_dry_eps": 1e-3},
                    boundary_conditions=bcs)
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1), riemann=PositiveHLL,
        dt_max=5.0, source_treatment=SourceTreatment.linearized,
        extra_operations=[guard_eigenvalue_powers()])
    return nsm


def raster_state():
    """The raster (row 0 = north): bed, catchment mask, coarse 1 h state h, hu, hv."""
    Z = np.load(DEM)
    C = np.load(COARSE)
    catch = Z != NODATA
    assert C["h"].shape == Z.shape and (np.isfinite(C["h"]) == catch).all(), \
        "coarse state and DEM disagree on the catchment"
    h = np.nan_to_num(C["h"])
    return (np.where(catch, Z, B_WALL), catch, h, h * np.nan_to_num(C["u"]),
            h * np.nan_to_num(C["v"]), float(C["xll"]), float(C["yll"]), float(C["dx"]))


def mesh_and_state(nsm):
    """(mesh, Q0 in the mesh's cell order [b, h, hu, hv], raster index (i, j) of
    each mesh cell, catchment mask, dx)."""
    from zoomy_core.mesh import LSQMesh, ensure_lsq_mesh
    bed, catch, h, hu, hv, xll, yll, dx = raster_state()
    nr, nc = bed.shape
    mesh = LSQMesh.create_2d(domain=(xll, xll + nc * dx, yll, yll + nr * dx), nx=nc, ny=nr)
    mesh = ensure_lsq_mesh(mesh, nsm)
    n = int(mesh.n_inner_cells)
    assert n == nr * nc
    xc = np.asarray(mesh.cell_centers)[:2, :n]
    j = np.floor((xc[0] - xll) / dx).astype(int)               # column, west -> east
    i = nr - 1 - np.floor((xc[1] - yll) / dx).astype(int)      # row, row 0 = north
    assert i.min() >= 0 and i.max() < nr and j.min() >= 0 and j.max() < nc
    assert np.unique(i * nc + j).size == n, "cell centers do not map one-to-one onto the raster"
    assert np.allclose(xc[0], xll + (j + 0.5) * dx) and np.allclose(xc[1], yll + (nr - i - 0.5) * dx)
    Q0 = np.stack([bed[i, j], h[i, j], hu[i, j], hv[i, j]])
    return mesh, Q0, (i, j), catch, dx


def to_raster(v, ij, shape, fill=np.nan):
    r = np.full(shape, fill)
    r[ij] = v
    return r


def wall_adjacent(catch):
    """Catchment cells with a nodata cell (or the box edge) on a side."""
    pad = np.pad(catch, 1, constant_values=False)
    return catch & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])


def main(out, T, cfg):
    os.makedirs(out, exist_ok=True)
    tag = f"{T:g}"
    t0 = clock.perf_counter()
    nsm = nsm_of()
    mesh, Q0, ij, catch, dx = mesh_and_state(nsm)
    shape = catch.shape
    if cfg == "jax":
        from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
        import jax.numpy as jnp
        s = SymbolicHyperbolicSolverJax(time_end=T, cfl=0.5, mood=False)
        Q, A = s.setup(mesh, nsm)
        assert np.allclose(np.asarray(s.mesh_np.cell_centers)[:2, :Q.shape[1]],
                           np.asarray(mesh.cell_centers)[:2, :Q.shape[1]])
        Q = jnp.asarray(Q0)
        A = s.initial_aux(Q, A)
        t1 = clock.perf_counter()
        t, Qf, _ = s.run(Q, A)
        Qf = np.asarray(Qf.block_until_ready())
        wall, steps = clock.perf_counter() - t1, None
        setup = t1 - t0
    else:
        from zoomy_amrex.symsolver import SymbolicHyperbolicSolverAmrex
        s = SymbolicHyperbolicSolverAmrex(ranks=int(cfg[3:]), gpu=cfg.startswith("gpu"),
                                          time_end=T, cfl=0.5, mood=False)
        t, Qf = s.run(mesh, nsm, Q0, os.path.join(out, f"run_{cfg}_{tag}"))
        Qf = np.asarray(Qf)
        wall, steps = s.wall, s.steps
        setup = clock.perf_counter() - t0 - wall
    assert abs(float(t) - T) <= 1e-9 * max(T, 1.0), f"stopped at t = {float(t)}, not {T}"
    assert np.isfinite(Qf).all(), "non-finite final state"
    h = to_raster(Qf[1], ij, shape)
    dh = to_raster(Qf[1] - Q0[1], ij, shape)
    area, n_catch = dx * dx, int(catch.sum())
    vol = float(dh[catch].sum()) * area
    rain = R_O * n_catch * area * T
    name = "jax" if cfg == "jax" else f"amrex_{cfg}"
    h_out = np.where(catch, h, np.nan)
    np.savez(os.path.join(out, f"{name}_{tag}.npz"), h=h_out, t=float(t),
             b=to_raster(Q0[0], ij, shape), hu=to_raster(Qf[2], ij, shape),
             hv=to_raster(Qf[3], ij, shape))
    hmax = float(np.nanmax(h_out))
    ref = os.path.join(out, f"jax_{tag}.npz")
    dtxt = "n/a"
    if cfg != "jax" and os.path.exists(ref):
        hj = np.load(ref)["h"]
        d = np.abs(np.nan_to_num(h_out) - np.nan_to_num(hj))
        dtxt = f"{d.max() / np.nanmax(hj):.3e}"
        k = np.unravel_index(np.argmax(d), d.shape)
        adj = wall_adjacent(catch)
        print(f"  max|dh| {d.max():.3e} m at raster {k} (wall-adjacent: {bool(adj[k])}); "
              f"max over wall-adjacent {d[adj].max():.3e} m, over the others "
              f"{d[catch & ~adj].max():.3e} m; |sum dh| "
              f"{abs(np.nansum(h_out - hj)) * area:.3e} m3; h on the nodata cells "
              f"{np.abs(h[~catch]).max() if (~catch).any() else 0.0:.1e}", flush=True)
    print(f"  |water change - rain| / rain = {abs(vol - rain) / rain:.3e}", flush=True)
    stp = "n/a steps, n/a ms/step" if steps is None else f"{steps} steps, {1e3 * wall / steps:.1f} ms/step"
    print(f"catchment 10 m: {Q0.shape[1]} cells ({n_catch} catchment), {cfg}: setup {setup:.1f} s, "
          f"{T:g} s simulated in {wall:.2f} s ({stp}); water change {vol:.6f} m3 vs rain "
          f"{rain:.6f} m3; h_max {hmax:.4f} m; "
          f"vs JAX max|dh|/max h {dtxt}", flush=True)


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]), float(sys.argv[2]), sys.argv[3])

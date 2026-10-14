"""SERGHEI LowerTriangle catchment (thesis/cases/river/serghei) at large scale
on the distributed symbolic solver.  RainSWE (SWE + 30 mm/h rain + Manning
n = 0.03), first order, PositiveHLL, point-implicit friction, CFL 0.5,
dt_max 5 s, walls (the catchment cells only; nodata cells are not meshed).

    python bench_catchment.py coarse DIR          # 10 m, 1 h, 1 GPU -> DIR/coarse.npz
    mpiexec -n 4 python bench_catchment.py mesh DIR DEM.npy   # fine mesh file, in parallel
    mpiexec -n 4 python bench_catchment.py run DIR T [DT_WRITE]  # T s from the mapped 1 h state

The fine run starts from the coarse 1 h state mapped onto the fine cells
(depth and velocity of the coarse cell: the coarse water volume; the free
surface eta = b + h would put metres of water on 1 m beds below a barely wet
10 m hillslope cell), so dt is the dt of a flowing catchment, not of a dry
start."""
import os
import sys
import time as clock
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.join(HERE, "../../../../thesis/cases/river/serghei")
sys.path.insert(0, os.path.join(CASE, "amrex"))
from rain_swe import RainSWE                                    # noqa: E402

R_O = 30.0 / 1000.0 / 3600.0                                    # 30 mm/h
mode, out = sys.argv[1], os.path.abspath(sys.argv[2])


def nsm_of():
    import zoomy_core.model.initial_conditions as IC
    from zoomy_core.fvm.riemann_solvers import PositiveHLL
    from zoomy_core.model.boundary_conditions import BoundaryConditions, Wall
    from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec, SourceTreatment
    from zoomy_core.systemmodel import SystemModel
    from zoomy_core.systemmodel.operations import guard_eigenvalue_powers
    bcs = BoundaryConditions([Wall(tag="wall", momentum_field_indices=[[2, 3]],
                                   permeability=0.0, wall_slip=1.0)])
    model = RainSWE(dimension=2, parameters={"g": 9.81, "n_m": 0.03, "r_o": R_O,
                                             "b_wall": 1.0e4, "wet_dry_eps": 1e-3},
                    boundary_conditions=bcs)
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    return NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1), riemann=PositiveHLL,
        dt_max=5.0, source_treatment=SourceTreatment.linearized,
        extra_operations=[guard_eigenvalue_powers()])


def raster_band(Z, xll, yll, dx, r0, r1, n0, n1):
    """Cells of raster rows r0:r1 (row 0 = north) that are catchment (finite),
    as global node ids on the (nr+1) x (nc+1) node grid; and the coordinates
    of node rows n0:n1."""
    nr, nc = Z.shape
    i, j = np.nonzero(np.isfinite(Z[r0:r1]))
    i = i + r0
    nid = lambda a, b: a * (nc + 1) + b                          # noqa: E731
    cv = np.stack([nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1), nid(i, j)])
    ni, nj = np.meshgrid(np.arange(n0, n1), np.arange(nc + 1), indexing="ij")
    X = np.stack([xll + nj.ravel() * dx, yll + (nr - ni.ravel()) * dx])
    return cv, X, Z[i, j], (i, j)


if mode == "coarse":
    from zoomy_core.mesh.base_mesh import BaseMesh
    from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
    D = np.load(os.path.join(CASE, "run/LowerTriangle_LowRes/dem_array.npy"))
    hdr_file = os.environ.get("LOWRES_DEM", os.path.join(
        CASE, "testcases/Hydrology/Catchments/LowerTriangle/LowRes/input/dem.input"))
    with open(hdr_file) as f:                                    # the header only
        hdr = {k.lower(): float(v) for k, v in (f.readline().split() for _ in range(6))}
    Z = np.where(D == hdr["nodata_value"], np.nan, D)
    cv, X, b, (i, j) = raster_band(Z, hdr["xllcorner"], hdr["yllcorner"],
                                   hdr["cellsize"], 0, Z.shape[0], 0, Z.shape[0] + 1)
    mesh = BaseMesh.from_arrays(X, cv, "quad", default_tag=0, tag_names={0: "wall"})
    nsm = nsm_of()
    s = SymbolicHyperbolicSolverJax(time_end=3600.0, cfl=0.5)
    Q, A = s.setup(mesh, nsm)
    Q = Q.at[0].set(b)                                           # bed; dry start
    A = s.initial_aux(Q, A)
    t0 = clock.perf_counter()
    t, Q, A = s.run(Q, A)
    Q = np.asarray(Q)
    area = hdr["cellsize"] ** 2
    print(f"coarse 10 m: {b.size} cells, t={float(t):.0f} s in {clock.perf_counter()-t0:.0f} s; "
          f"water {Q[1].sum()*area:.1f} m3 vs r_o*A*t {R_O*b.size*area*3600:.1f} m3; "
          f"h_max {Q[1].max():.3f} m", flush=True)
    hc, u, v = (np.full(Z.shape, np.nan) for _ in range(3))
    wet = Q[1] > 1e-3
    hc[i, j] = Q[1]
    u[i, j] = np.where(wet, Q[2] / np.maximum(Q[1], 1e-3), 0.0)
    v[i, j] = np.where(wet, Q[3] / np.maximum(Q[1], 1e-3), 0.0)
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "coarse.npz"), h=hc, u=u, v=v,
             xll=hdr["xllcorner"], yll=hdr["yllcorner"], dx=hdr["cellsize"])

elif mode == "mesh":
    from mpi4py import MPI
    from zoomy_core.mesh.distributed import write_mesh_parallel
    comm = MPI.COMM_WORLD
    P, r = comm.Get_size(), comm.Get_rank()
    t0 = clock.perf_counter()
    Z = np.load(sys.argv[3], mmap_mode="r")
    xll, yll, dx = np.load(sys.argv[3].replace(".npy", "_hdr.npy"))
    nr, nc = Z.shape
    r0, r1 = r * nr // P, (r + 1) * nr // P                     # cell rows
    n0, n1 = r * (nr + 1) // P, (r + 1) * (nr + 1) // P          # node rows
    cv, X, b, (i, j) = raster_band(Z, xll, yll, dx, r0, r1, n0, n1)
    c0 = comm.exscan(cv.shape[1]) or 0
    # the coarse 1 h state on the fine bed: same free surface, same velocity
    C = np.load(os.path.join(out, "coarse.npz"))
    xc = xll + (j + 0.5) * dx
    yc = yll + (nr - i - 0.5) * dx
    nrc, ncc = C["h"].shape
    cj = np.floor((xc - C["xll"]) / C["dx"]).astype(int)
    ci = nrc - 1 - np.floor((yc - C["yll"]) / C["dx"]).astype(int)
    inside = (ci >= 0) & (ci < nrc) & (cj >= 0) & (cj < ncc)
    ci, cj = np.clip(ci, 0, nrc - 1), np.clip(cj, 0, ncc - 1)
    hc = np.where(inside, C["h"][ci, cj], np.nan)
    ok = np.isfinite(hc)                      # a catchment cell of the coarse mesh
    h = np.where(ok, hc, 0.0)
    hu = h * np.where(ok, np.nan_to_num(C["u"][ci, cj]), 0.0)
    hv = h * np.where(ok, np.nan_to_num(C["v"][ci, cj]), 0.0)
    t1 = clock.perf_counter()
    write_mesh_parallel(comm, os.path.join(out, "fine.h5"), "quad", X, n0 * (nc + 1), cv, c0,
                        default_tag=0, tag_names={0: "wall"},
                        cell_data={"b": b, "h": h, "hu": hu, "hv": hv})
    comm.Barrier()
    t2 = clock.perf_counter()
    n = comm.allreduce(cv.shape[1])
    if r == 0:
        print(f"fine mesh {dx:.2f} m: {n/1e6:.2f}M cells, ranks {P}: build {t1-t0:.1f} s, "
              f"parallel write {t2-t1:.1f} s, file {os.path.getsize(os.path.join(out, 'fine.h5'))/1e9:.2f} GB",
              flush=True)

elif mode == "run":
    from mpi4py import MPI
    import jax.numpy as jnp
    from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
    comm = MPI.COMM_WORLD
    T = float(sys.argv[3])
    path = os.path.join(out, "fine.h5")
    nsm = nsm_of()
    dt_write = float(sys.argv[4]) if len(sys.argv) > 4 else None
    s = SymbolicHyperbolicSolverJax(time_end=T, cfl=0.5, comm=comm, dt_write=dt_write,
                                    output=os.path.join(out, "fine_out.h5") if dt_write else None)
    t0 = clock.perf_counter()
    Q, A = s.setup(path, nsm)
    part = s.partition
    Q = jnp.stack([jnp.asarray(part.source.cell_data(k, part.gids))
                   for k in ("b", "h", "hu", "hv")])
    A = s.initial_aux(Q, A)
    comm.Barrier()
    t1 = clock.perf_counter()
    if comm.Get_rank() == 0:
        print(f"  setup {t1-t0:.0f} s", flush=True)
    m, rt = s.module, s.rt
    speed_k = rt.eigenvalues_cfl if rt.eigenvalues_cfl is not None else rt.eigenvalues

    def dt_cfl(Q, A):
        """The CFL step of a state: cfl 2 r / (d |lambda|), min over faces."""
        sp = m["face_wavespeed"](speed_k, Q, A, s.parameters, s.mesh)
        return float(m["reduce_min"](0.5 * 2 * m["face_inradius"](s.mesh) / (2 * sp)))
    s.time_end, s.dt_write = 1e-9, None                          # compile: one step
    if dt_write is None:
        s.run(Q, A)[1].block_until_ready()
    comm.Barrier()
    t2 = clock.perf_counter()
    if comm.Get_rank() == 0:
        print(f"  compile {t2-t1:.0f} s", flush=True)
    s.time_end, s.dt_write = T, dt_write
    tt, Q2, A2 = s.run(Q, A)
    Q2.block_until_ready()
    comm.Barrier()
    t3 = clock.perf_counter()
    n_own = part.n_own
    dt0, dt1 = min(dt_cfl(Q, A), 5.0), min(dt_cfl(Q2, A2), 5.0)
    h = np.asarray(Q2[1, :n_own])
    area = np.asarray(s.mesh.cell_volumes[:n_own])
    vol0 = comm.allreduce(float((np.asarray(Q[1, :n_own]) * area).sum()))
    vol = comm.allreduce(float((h * area).sum()))
    rain = R_O * comm.allreduce(float(area.sum())) * T
    hmax = comm.allreduce(float(h.max()), op=MPI.MAX)
    if comm.Get_rank() == 0:
        print(f"fine run: {part.n_global/1e6:.2f}M cells on {part.size} GPUs "
              f"({n_own} owned/rank 0): setup {t1-t0:.0f} s, compile {t2-t1:.0f} s, "
              f"{T:g} s simulated in {t3-t2:.1f} s; h_max {hmax:.2f} m; water change "
              f"{vol-vol0:.3f} m3 vs rain {rain:.3f} m3; "
              f"CFL dt {dt0:.4f} s (start) {dt1:.4f} s (end) -> ~{T/(0.5*(dt0+dt1)):.0f} steps, "
              f"~{1e3*(t3-t2)/(T/(0.5*(dt0+dt1))):.0f} ms/step; 1 h of catchment ~"
              f"{(t3-t2)/T*3600/3600:.1f} h wall", flush=True)

"""Distributed runs of the NumPy backend (mpi4py): the `halo` exchange and the global reductions of
the runtime's name table, the global sums and norms of the implicit solves, and the distributed
linear solves.  The partition -- owned cells, two halo layers, the exchange lists, the cells and
faces each rank counts -- is `zoomy_core.mesh.distributed.Partition`, the JAX backend's.

mpi4py and petsc4py are imported only when a run is distributed: the serial backend (and the
browser) runs without them.

Linear solves (`MpiLinearSolver`): PETSc through petsc4py -- KSP GMRES on the matrix-free
operator, FGMRES + BoomerAMG (or GAMG) on the colour-assembled rows, a parallel LU (MUMPS,
SuperLU_dist) of the implicit diffusion's Jacobian -- or, without petsc4py, a distributed
restarted FGMRES with block-Jacobi preconditioners from the rank's own block.

References:
    Dalcin, Fang 2021, https://doi.org/10.1109/MCSE.2021.3083216
    Dalcin, Paz, Kler, Cosimo 2011, https://doi.org/10.1016/j.advwatres.2011.04.013
    Balay et al. 2024, PETSc/TAO Users Manual, ANL-21/39 Rev. 3.21, https://petsc.org
    Amestoy, Duff, L'Excellent, Koster 2001, https://doi.org/10.1137/S0895479899358194
    Henson, Yang 2002, https://doi.org/10.1016/S0168-9274(01)00115-5
    Saad 1993, https://doi.org/10.1137/0914028
"""
from __future__ import annotations

import numpy as np

from zoomy_core.misc.logger_config import logger
from zoomy_core.misc.options import Option
from zoomy_core.symsolver.numpy_backend.assembly import cell_pattern


class MpiLinearSolver(Option):
    """The linear solves of a distributed NumPy run (pressure, implicit diffusion)."""
    #: petsc when petsc4py is installed, else krylov
    auto = "auto"
    #: PETSc (petsc4py): KSP GMRES matrix-free, FGMRES + BoomerAMG / GAMG on the assembled
    #: rows, a parallel LU (MUMPS / SuperLU_dist) of the implicit diffusion's Jacobian
    petsc = "petsc"
    #: a distributed restarted FGMRES (mpi4py only), block-Jacobi by the rank's own block
    krylov = "krylov"


_LOCAL = [False]


class no_halo:
    """Context manager inside which `halo` is the identity: an operator then gives the rank's own
    rows with its halo columns as probed (assembly by colouring, no collective inside).
    """

    def __enter__(self):
        """Switch the halo exchange off."""
        self._was, _LOCAL[0] = _LOCAL[0], True

    def __exit__(self, *exc):
        """Restore the previous halo state."""
        _LOCAL[0] = self._was


def _petsc():
    """petsc4py's `PETSc`, or None."""
    try:
        from petsc4py import PETSc
    except ImportError:
        return None
    return PETSc


class Distributed:
    """One rank's piece of a distributed NumPy run: halo exchange, global reductions and sums
    (each cell and face counted on one rank), the global numbering of the owned unknowns and the
    distributed linear solves.

    Unknowns of a `(k, cells)` block are numbered cell-major: local cell `c`, row `r` -> `c k + r`;
    the owned ones globally `gid(c) k + r`, a contiguous range per rank (`Partition.offsets`).
    """

    def __init__(self, part, mesh_np, method=MpiLinearSolver.auto):
        from mpi4py import MPI
        self.MPI, self.part = MPI, part
        self.comm = part.comm.Clone()
        self.rank, self.size = self.comm.Get_rank(), self.comm.Get_size()
        self.nc, self.nf = int(mesh_np.n_inner_cells), int(mesh_np.n_faces)
        if self.nc == self.nf:
            raise ValueError("local mesh with as many faces as cells: the reductions cannot "
                             "tell them apart")
        self.n_own = int(part.n_own)
        self.own_cell, self.own_face = part.owned(mesh_np)
        self.send = {int(q): np.asarray(i, dtype=np.int64) for q, i in part.send.items()}
        self.recv = {int(q): np.asarray(i, dtype=np.int64) for q, i in part.recv.items()}
        self.gids = np.asarray(part.gids, dtype=np.int64)
        self.offset, self.n_global = int(part.offsets[part.rank]), int(part.n_global)
        method = MpiLinearSolver(method)
        self.PETSc = None
        if method in (MpiLinearSolver.auto, MpiLinearSolver.petsc):
            self.PETSc = _petsc()
            if self.PETSc is None and method == MpiLinearSolver.petsc:
                raise ImportError("mpi_linear_solver=petsc needs petsc4py")
        self.method = MpiLinearSolver.petsc if self.PETSc is not None else MpiLinearSolver.krylov
        self.patterns, self.logged = {}, set()

    # ── halo and reductions ───────────────────────────────────────────────
    def halo(self, X, mesh=None):
        """`X` with its halo columns refreshed from their owners (point-to-point); the identity
        on a scalar and inside `no_halo`.
        """
        if _LOCAL[0] or self.size == 1 or np.ndim(X) == 0:
            return X
        X = np.array(X, copy=True)
        if X.shape[-1] != self.nc:
            raise ValueError(f"halo of shape {X.shape}: not {self.nc} cells")
        flat = X.reshape(-1, self.nc)
        reqs, got, keep = [], {}, []
        for q, idx in self.recv.items():
            got[q] = np.empty((flat.shape[0], len(idx)), dtype=X.dtype)
            reqs.append(self.comm.Irecv(got[q], source=q, tag=17))
        for q, idx in self.send.items():
            keep.append(np.ascontiguousarray(flat[:, idx]))
            reqs.append(self.comm.Isend(keep[-1], dest=q, tag=17))
        self.MPI.Request.Waitall(reqs)
        for q, idx in self.recv.items():
            flat[:, idx] = got[q]
        return X

    def owned(self, x, empty):
        """The entries of `x` (faces or cells on the last axis) this rank counts; the others
        `empty`.
        """
        x = np.asarray(x)
        if x.ndim == 0:
            return x
        if x.shape[-1] == self.nf:
            return np.where(self.own_face, x, empty)
        if x.shape[-1] == self.nc:
            return np.where(self.own_cell, x, empty)
        raise ValueError(f"reduce over {x.shape}: neither {self.nc} cells nor {self.nf} faces")

    def gather(self, x):
        """Every rank's `x`, in rank order (the same list on every rank)."""
        return self.comm.allgather(x)

    def reduce_min(self, x):
        """Global min over the counted entries; a NaN anywhere is the result (rank-local
        `np.min`, then the min of the gathered values: no MPI.MIN comparison drops it).
        """
        return np.min(np.asarray(self.gather(np.min(self.owned(x, np.inf)))))

    def reduce_max(self, x):
        """Global max over the counted entries; NaN-propagating as `reduce_min`."""
        return np.max(np.asarray(self.gather(np.max(self.owned(x, -np.inf)))))

    def reduce_any(self, x):
        """Global any over the counted entries."""
        return bool(self.comm.allreduce(bool(np.any(self.owned(x, False))), op=self.MPI.LOR))

    def total(self, x):
        """Sum over the ranks of `x` (scalar or array), in rank order: the same bits on every
        rank, so decisions taken on it agree.
        """
        return np.sum(np.asarray(self.gather(np.asarray(x, dtype=float))), axis=0)

    def norm(self, r):
        """Euclidean norm of `r` `(..., cells)` over the owned cells of all ranks."""
        r = np.asarray(r, dtype=float)
        return float(np.sqrt(self.total(np.sum(r[..., :self.n_own] ** 2))))

    def integral(self, X, volumes):
        """`sum_cells X vol` per row over the whole mesh (e.g. the mass), on every rank."""
        X = np.asarray(X, dtype=float)
        return self.total(X[..., :self.n_own] @ np.asarray(volumes)[:self.n_own])

    def blocks(self):
        """The name-table entries of a distributed run."""
        return dict(halo=self.halo, reduce_min=self.reduce_min, reduce_max=self.reduce_max,
                    reduce_any=self.reduce_any)

    # ── owned unknowns ───────────────────────────────────────────────────
    def own(self, X):
        """The owned unknowns of `X` `(k, cells)`, cell-major."""
        return np.asarray(X, dtype=float)[:, :self.n_own].T.reshape(-1)

    def embed(self, xo, k):
        """`(k, cells)` with the owned unknowns `xo` and zero halo columns."""
        X = np.zeros((k, self.nc))
        X[:, :self.n_own] = np.asarray(xo).reshape(self.n_own, k).T
        return X

    def assemble(self, op, k, nbrs, key):
        """The owned rows of the linear operator `op` `(k, cells) -> (k, cells)` on the cell
        stencil `nbrs`, by colouring with the halo off (no collective inside `op`, so the ranks
        may take different numbers of colours).

        Returns:
            `(rows, cols, vals)`: CSR-ordered owned rows, local cell-major columns.
        """
        if key not in self.patterns:
            self.patterns[key] = cell_pattern(k * self.nc, self.nc, nbrs, self.n_own)
        rows, cols, colors = self.patterns[key]
        AV = []
        with no_halo():
            for c in range(int(colors.max()) + 1):
                e = (colors == c).astype(float).reshape(self.nc, k).T
                AV.append(np.asarray(op(e), dtype=float).T.reshape(-1))
        return rows, cols, np.stack(AV)[colors[cols], rows]

    def global_csr(self, rows, cols, vals, k):
        """`(indptr, global columns, values)` of owned rows `rows`, columns sorted per row."""
        g = self.gids[cols // k] * k + cols % k
        o = np.lexsort((g, rows))
        rows, g, vals = rows[o], g[o], vals[o]
        indptr = np.searchsorted(rows, np.arange(k * self.n_own + 1)).astype(np.int32)
        return indptr, g.astype(np.int32), vals

    def log_once(self, key, msg):
        """Log `msg` on rank 0, once per `key`."""
        if key not in self.logged:
            self.logged.add(key)
            if self.rank == 0:
                logger.info(msg)

    # ── distributed linear solves ──────────────────────────────────────────
    def krylov(self, mv, b, tol, restart, maxiter, precond=None):
        """Restarted FGMRES of `mv(x) = b` on the owned unknowns (right preconditioner
        `precond`), global inner products (CGS2, two gathers per iteration); stops at
        `|r| <= tol |b|` or after `restart x maxiter` iterations.

        References:
            Saad 1993, https://doi.org/10.1137/0914028
        """
        b = np.asarray(b, dtype=float)
        M = precond or (lambda v: v)
        nb = float(np.sqrt(self.total(b @ b)))
        x = np.zeros_like(b)
        if nb == 0.0:
            return x
        m = int(restart)
        for _ in range(int(maxiter)):
            r = b - mv(x)
            beta = float(np.sqrt(self.total(r @ r)))
            if beta <= tol * nb:
                break
            V, Z = np.zeros((m + 1, b.size)), np.zeros((m, b.size))
            H = np.zeros((m + 1, m))
            V[0] = r / beta
            g = np.zeros(m + 1)
            g[0] = beta
            cs, sn = np.zeros(m), np.zeros(m)
            j = 0
            for j in range(m):
                Z[j] = M(V[j])
                w = mv(Z[j])
                h = self.total(V[:j + 1] @ w)
                w = w - h @ V[:j + 1]
                h2 = self.total(V[:j + 1] @ w)           # re-orthogonalize (CGS2)
                w = w - h2 @ V[:j + 1]
                H[:j + 1, j] = h + h2
                H[j + 1, j] = float(np.sqrt(self.total(w @ w)))
                if H[j + 1, j] > 0.0:
                    V[j + 1] = w / H[j + 1, j]
                for i in range(j):                      # Givens rotations
                    H[i, j], H[i + 1, j] = (cs[i] * H[i, j] + sn[i] * H[i + 1, j],
                                            -sn[i] * H[i, j] + cs[i] * H[i + 1, j])
                den = np.hypot(H[j, j], H[j + 1, j])
                cs[j], sn[j] = (1.0, 0.0) if den == 0.0 else (H[j, j] / den, H[j + 1, j] / den)
                H[j, j], H[j + 1, j] = den, 0.0
                g[j], g[j + 1] = cs[j] * g[j], -sn[j] * g[j]
                if abs(g[j + 1]) <= tol * nb:
                    break
            n = j + 1
            y = np.linalg.solve(np.triu(H[:n, :n]), g[:n]) if n else np.zeros(0)
            x = x + y @ Z[:n]
        return x

    def local_block(self, rows, cols, vals, k):
        """The rank's rows as a scipy CSR `(k n_own, k cells)` (local columns)."""
        import scipy.sparse as sps
        return sps.csr_matrix((vals, (rows, cols)), shape=(k * self.n_own, k * self.nc))

    def apply(self, J, xo, k):
        """`J` (local block) times the owned unknowns `xo`, their halo copies exchanged."""
        return J @ self.halo(self.embed(xo, k)).T.reshape(-1)

    def lu(self, rows, cols, vals, k):
        """A solver `solve(rhs_owned) -> x_owned` of the assembled rows: PETSc's parallel LU
        (MUMPS, else SuperLU_dist; else GMRES to 1e-14), or (krylov) FGMRES to 1e-14
        preconditioned by the LU of the rank's diagonal block.  `close()` frees it.
        """
        if self.method == MpiLinearSolver.petsc:
            return _PetscSystem(self, rows, cols, vals, k, kind="lu")
        from scipy.sparse.linalg import splu
        J = self.local_block(rows, cols, vals, k)
        n_o = k * self.n_own
        diag = splu(J[:, :n_o].tocsc())
        self.log_once("lu", "numpy MPI: implicit diffusion by FGMRES (1e-14) + block-Jacobi LU")

        class Solve:
            """Distributed FGMRES on `J` with the block-Jacobi LU."""
            close = staticmethod(lambda: None)

            @staticmethod
            def solve(rhs):
                """The owned solution of `J x = rhs`."""
                return self.krylov(lambda x: self.apply(J, x, k), rhs, 1e-14, 60, 20,
                                   diag.solve)
        return Solve()


class _PetscSystem:
    """An assembled distributed system in PETSc: the owned rows (global columns) as an MPIAIJ
    matrix and its KSP -- `kind="lu"`: a parallel direct solve; `kind="amg"`: FGMRES + BoomerAMG
    (hypre) or GAMG.
    """

    def __init__(self, dist, rows, cols, vals, k, kind, ls=None):
        PETSc = dist.PETSc
        self.dist, self.k = dist, k
        self.A = self._matrix(rows, cols, vals)
        self.ksp = PETSc.KSP().create(comm=dist.comm)
        self.ksp.setOperators(self.A)
        pc = self.ksp.getPC()
        if kind == "lu":
            pkg = next((p for p in ("mumps", "superlu_dist")
                        if PETSc.Sys.hasExternalPackage(p)), None)
            if pkg is not None:
                self.ksp.setType("preonly")
                pc.setType("lu")
                pc.setFactorSolverType(pkg)
            else:
                self.ksp.setType("gmres")
                pc.setType("bjacobi")
                self.ksp.setTolerances(rtol=1e-14, atol=0.0, max_it=2000)
            dist.log_once("lu", f"numpy MPI: implicit diffusion by PETSc "
                                f"{'LU (' + pkg + ')' if pkg else 'GMRES + block Jacobi'}")
        else:
            self.ksp.setType("fgmres")
            self.ksp.setGMRESRestart(int(ls["restart"]))
            self.ksp.setTolerances(rtol=float(ls["tol"]), atol=0.0, divtol=1e300,
                                   max_it=int(ls["restart"]) * int(ls["maxiter"]))
            amg = "hypre" if PETSc.Sys.hasExternalPackage("hypre") else "gamg"
            pc.setType(amg)
            if amg == "hypre":
                pc.setHYPREType("boomeramg")
            dist.log_once("amg", f"numpy MPI: pressure by PETSc FGMRES + "
                                 f"{'BoomerAMG' if amg == 'hypre' else 'GAMG'}")
        self.ksp.setFromOptions()
        self.x, self.b = self.A.createVecs()

    def _matrix(self, rows, cols, vals):
        """The MPIAIJ matrix of the owned rows (global columns)."""
        PETSc, d, k = self.dist.PETSc, self.dist, self.k
        n_loc, N = k * d.n_own, k * d.n_global
        indptr, g, v = d.global_csr(rows, cols, vals, k)
        A = PETSc.Mat().createAIJ(size=((n_loc, N), (n_loc, N)),
                                  csr=(indptr.astype(PETSc.IntType), g.astype(PETSc.IntType),
                                       np.asarray(v, dtype=PETSc.ScalarType)), comm=d.comm)
        A.assemble()
        return A

    def update(self, rows, cols, vals, rebuild):
        """New matrix values; the preconditioner rebuilt, or kept (lagged)."""
        old, self.A = self.A, self._matrix(rows, cols, vals)
        self.ksp.setReusePreconditioner(not rebuild)
        self.ksp.setOperators(self.A)
        old.destroy()

    def solve(self, rhs):
        """The owned solution for the owned right-hand side `rhs`."""
        self.b.setArray(np.asarray(rhs, dtype=float))
        self.ksp.solve(self.b, self.x)
        return self.x.getArray().copy()

    def iterations(self):
        """Iterations of the last solve."""
        return int(self.ksp.getIterationNumber())

    def close(self):
        """Free the PETSc objects (collective)."""
        for o in (self.ksp, self.A, self.x, self.b):
            o.destroy()


def shell_gmres(dist, mv, b, tol, restart, maxiter):
    """GMRES of the matrix-free `mv` (owned -> owned) for the owned `b`: PETSc's KSP on a shell
    matrix, or the distributed FGMRES -- the serial backend's unpreconditioned GMRES, with global
    inner products.
    """
    if dist.method != MpiLinearSolver.petsc:
        dist.log_once("gmres", "numpy MPI: pressure by the distributed FGMRES")
        return dist.krylov(mv, b, tol, restart, maxiter)
    PETSc = dist.PETSc
    n = b.size
    N = int(dist.total(n))

    class Shell:
        """The operator of the shell matrix."""

        def mult(self, mat, x, y):
            """`y = mv(x)` on the owned unknowns."""
            y.setArray(mv(x.getArray(readonly=True).copy()))
    A = PETSc.Mat().createPython(((n, N), (n, N)), context=Shell(), comm=dist.comm)
    A.setUp()
    ksp = PETSc.KSP().create(comm=dist.comm)
    ksp.setOperators(A)
    ksp.setType("gmres")
    ksp.setGMRESRestart(int(restart))
    ksp.getPC().setType("none")
    ksp.setTolerances(rtol=float(tol), atol=0.0, divtol=1e300,
                      max_it=int(restart) * int(maxiter))
    ksp.setFromOptions()
    x, rhs = A.createVecs()
    rhs.setArray(np.asarray(b, dtype=float))
    ksp.solve(rhs, x)
    out = x.getArray().copy()
    dist.log_once("gmres", "numpy MPI: pressure by PETSc GMRES (matrix-free)")
    for o in (ksp, A, x, rhs):
        o.destroy()
    return out


def amg_gmres(dist, op, b, k, nbrs, ls, cache):
    """GMRES preconditioned by AMG on the colour-assembled owned rows, left-scaled by the inverse
    cell blocks (as `assembly.host_amg_solve`): PETSc FGMRES + BoomerAMG / GAMG, the hierarchy
    lagged (`amg_rebuild`, `amg_growth`); or (krylov) the distributed FGMRES with the rank's own
    smoothed-aggregation V-cycle (block Jacobi).

    Args:
        op: the linear operator `(k, cells) -> (k, cells)` (halo inside)
        b: owned right-hand side
        cache: kept between the solves of one operator
    """
    import scipy.sparse as sps
    rows, cols, vals = dist.assemble(op, k, nbrs, ("amg", k))
    n_o = k * dist.n_own
    # inverse cell blocks of the owned rows: the (c, c) block of every owned cell
    diag = (cols // k) == (rows // k)
    blocks = np.zeros((dist.n_own, k, k))
    blocks[rows[diag] // k, rows[diag] % k, cols[diag] % k] = vals[diag]
    Dinv = np.linalg.inv(blocks)
    # scaled rows: (Dinv A)[c k + r, :] = sum_s Dinv[c, r, s] A[c k + s, :]
    A_loc = sps.csr_matrix((vals, (rows, cols)), shape=(n_o, k * dist.nc))
    S = sps.block_diag(Dinv, format="csr")
    As = (S @ A_loc).tocsr()
    As.sort_indices()
    As = As.tocoo()
    bs = S @ np.asarray(b, dtype=float)
    rebuild, growth = int(ls.get("amg_rebuild", 1)), float(ls.get("amg_growth", 2.0))
    fresh = (not cache or cache["age"] >= rebuild
             or (cache["it0"] is not None and cache["it"] > growth * max(cache["it0"], 1)))
    if dist.method == MpiLinearSolver.petsc:
        if "sys" not in cache:
            cache.update(sys=_PetscSystem(dist, As.row, As.col, As.data, k, "amg", ls), age=0,
                         it0=None, it=0)
        elif fresh:
            cache["sys"].update(As.row, As.col, As.data, True)
            cache.update(age=0, it0=None)
        else:
            cache["sys"].update(As.row, As.col, As.data, False)
        x = cache["sys"].solve(bs)
        its = cache["sys"].iterations()
    else:
        import pyamg
        if fresh:
            ml = pyamg.smoothed_aggregation_solver(As.tocsr()[:, :n_o].tobsr(blocksize=(k, k)))
            cache.update(P=ml.aspreconditioner(cycle="V"), age=0, it0=None, it=0)
        count, Ac = [0], As.tocsr()

        def mv(x):
            """The scaled operator on the owned unknowns."""
            count[0] += 1
            return dist.apply(Ac, x, k)
        dist.log_once("amg", "numpy MPI: pressure by the distributed FGMRES + per-rank "
                             "smoothed-aggregation V-cycle (block Jacobi)")
        x = dist.krylov(mv, bs, ls["tol"], ls["restart"], ls["maxiter"], cache["P"])
        its = count[0]
    cache["age"], cache["it"] = cache["age"] + 1, its
    if cache["it0"] is None:
        cache["it0"] = its
    return x

"""hypre through ctypes: FlexGMRES with one BoomerAMG V-cycle on the cell-block-scaled owned rows of
an operator; the JAX Chorin pressure solve on CPUs (`pressure_linear_solver=LinearSolver.hypre`,
`zoomy_jax.symsolver.runtime`).  The library is the shared build of
`containers/zoomy_amrex/build_hypre_cuda.sh cpu-env` (or `ZOOMY_HYPRE_LIB`); it must link the
MPI that mpi4py uses.

References:
    Falgout, Yang 2002, https://doi.org/10.1007/3-540-47789-6_66
    Henson, Yang 2002, https://doi.org/10.1016/S0168-9274(01)00115-5
    Saad 1993, https://doi.org/10.1137/0914028
"""
from __future__ import annotations

import ctypes
import os
import re
import time
from pathlib import Path

import numpy as np

VERSION = "3.2.0"
PARCSR = 5555                          # HYPRE_PARCSR
#: totals over every solve of the process: solves, iterations, setups, seconds per phase
STATS = dict(solves=0, iterations=0, setups=0, t_assemble=0.0, t_setup=0.0, t_solve=0.0)
_LIB = []


def library_path():
    """The hypre shared library: `ZOOMY_HYPRE_LIB`, else the cpu-env build under
    `ZOOMY_HYPRE_CACHE` (default `~/.cache/zoomy`).
    """
    if os.environ.get("ZOOMY_HYPRE_LIB"):
        return Path(os.environ["ZOOMY_HYPRE_LIB"])
    cache = Path(os.environ.get("ZOOMY_HYPRE_CACHE", Path.home() / ".cache" / "zoomy"))
    return cache / f"hypre-cpu-env-{VERSION}" / "lib" / "libHYPRE.so"


def available():
    """Whether the hypre shared library exists."""
    return library_path().is_file()


class _Lib:
    """The loaded library, its integer types and the `MPI_Comm` handle type."""

    def __init__(self, path):
        """Load `path`, declare the signatures used here and initialize hypre (MPI is up)."""
        from mpi4py import MPI                     # MPI initialized before hypre
        cfg = path.resolve().parent.parent / "include" / "HYPRE_config.h"
        text = cfg.read_text() if cfg.is_file() else ""
        on = lambda name: re.search(rf"^#define {name}\b", text, re.M) is not None  # noqa: E731
        if any(on(f"HYPRE_{p}") for p in ("SINGLE", "LONG_DOUBLE", "COMPLEX")):
            raise ImportError(f"{path}: hypre built without double-precision reals")
        self.Int = np.int64 if on("HYPRE_BIGINT") else np.int32
        self.Big = np.int64 if on("HYPRE_BIGINT") or on("HYPRE_MIXEDINT") else np.int32
        cint = {np.int32: ctypes.c_int32, np.int64: ctypes.c_int64}
        Int, Big = cint[self.Int], cint[self.Big]
        self.comm_t = (ctypes.c_int if MPI._sizeof(MPI.Comm) == ctypes.sizeof(ctypes.c_int)
                       else ctypes.c_void_p)
        lib = ctypes.CDLL(str(path))
        P, PP, D, C = ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_double, self.comm_t
        sig = {
            "HYPRE_Initialize": [], "HYPRE_ClearAllErrors": [],
            "HYPRE_IJMatrixCreate": [C, Big, Big, Big, Big, PP],
            "HYPRE_IJMatrixSetObjectType": [P, Int],
            "HYPRE_IJMatrixSetDiagOffdSizes": [P, P, P],
            "HYPRE_IJMatrixInitialize": [P],
            "HYPRE_IJMatrixSetValues": [P, Int, P, P, P, P],
            "HYPRE_IJMatrixAddToValues": [P, Int, P, P, P, P],
            "HYPRE_IJMatrixAssemble": [P], "HYPRE_IJMatrixGetObject": [P, PP],
            "HYPRE_IJMatrixDestroy": [P],
            "HYPRE_IJVectorCreate": [C, Big, Big, PP],
            "HYPRE_IJVectorSetObjectType": [P, Int], "HYPRE_IJVectorInitialize": [P],
            "HYPRE_IJVectorSetValues": [P, Int, P, P], "HYPRE_IJVectorGetValues": [P, Int, P, P],
            "HYPRE_IJVectorAssemble": [P], "HYPRE_IJVectorGetObject": [P, PP],
            "HYPRE_IJVectorDestroy": [P],
            "HYPRE_ParCSRMatrixMatvec": [D, P, P, D, P],
            "HYPRE_ParCSRFlexGMRESCreate": [C, PP], "HYPRE_ParCSRFlexGMRESDestroy": [P],
            "HYPRE_ParCSRFlexGMRESSetKDim": [P, Int], "HYPRE_ParCSRFlexGMRESSetMaxIter": [P, Int],
            "HYPRE_ParCSRFlexGMRESSetTol": [P, D], "HYPRE_ParCSRFlexGMRESSetAbsoluteTol": [P, D],
            "HYPRE_ParCSRFlexGMRESSetLogging": [P, Int],
            "HYPRE_ParCSRFlexGMRESSetPrintLevel": [P, Int],
            "HYPRE_ParCSRFlexGMRESSetPrecond": [P, P, P, P],
            "HYPRE_ParCSRFlexGMRESSetup": [P, P, P, P], "HYPRE_ParCSRFlexGMRESSolve": [P, P, P, P],
            "HYPRE_ParCSRFlexGMRESGetNumIterations": [P, P],
            "HYPRE_BoomerAMGCreate": [PP], "HYPRE_BoomerAMGDestroy": [P],
            "HYPRE_BoomerAMGSetTol": [P, D], "HYPRE_BoomerAMGSetMaxIter": [P, Int],
            "HYPRE_BoomerAMGSetPrintLevel": [P, Int], "HYPRE_BoomerAMGSetCoarsenType": [P, Int],
            "HYPRE_BoomerAMGSetInterpType": [P, Int], "HYPRE_BoomerAMGSetPMaxElmts": [P, Int],
            "HYPRE_BoomerAMGSetStrongThreshold": [P, D], "HYPRE_BoomerAMGSetRelaxType": [P, Int],
            "HYPRE_BoomerAMGSetNumSweeps": [P, Int], "HYPRE_BoomerAMGSetRelaxOrder": [P, Int],
        }
        for name, args in sig.items():
            f = getattr(lib, name)
            f.argtypes, f.restype = args, Int
        self.amg_solve = ctypes.cast(lib.HYPRE_BoomerAMGSolve, P)
        self.amg_setup = ctypes.cast(lib.HYPRE_BoomerAMGSetup, P)
        self.lib = lib
        self("Initialize")

    def __call__(self, name, *args, ok=()):
        """`HYPRE_<name>(*args)`; a nonzero error flag not in `ok` raises (all are cleared)."""
        err = getattr(self.lib, "HYPRE_" + name)(*args)
        if err:
            self.lib.HYPRE_ClearAllErrors()
            if err not in ok:
                raise RuntimeError(f"hypre: HYPRE_{name} returned error {err}")
        return err

    def comm(self, comm):
        """The `MPI_Comm` handle of the mpi4py communicator `comm`."""
        from mpi4py import MPI
        return self.comm_t(MPI._handleof(comm))


def load():
    """The hypre library, loaded and initialized once; ImportError when it is missing."""
    if not _LIB:
        path = library_path()
        if not path.is_file():
            raise ImportError(
                f"linear solver 'hypre' needs the hypre shared library ({path}): build it with "
                "containers/zoomy_amrex/build_hypre_cuda.sh cpu-env, or set ZOOMY_HYPRE_LIB.  "
                "Or use 'pyamg' / 'gmres'.")
        _LIB.append(_Lib(path))
    return _LIB[0]


def _ptr(a):
    """Address of the contiguous array `a`."""
    return ctypes.c_void_p(a.ctypes.data)


class BlockRows:
    """The owned rows of an operator on `k` unknowns per cell (row `c k + p` of owned cell
    `c`), its columns by global unknown number, solved by hypre with matrix values given per call.
    Every `k x k` block of a (cell, stencil cell) pair must be present.  Used by
    `zoomy_jax.symsolver.runtime` (`pressure_linear_solver=LinearSolver.hypre`).

    Args:
        comm: mpi4py communicator of the ranks (used as is; collective calls in `solve`)
        k: unknowns per cell
        n_own: owned cells; `lo`: the global number of the first
        rows, gcols: local row (`< k n_own`) and global column of every entry
        tol: target of the true relative residual `|b - A x| / |b|`
        restart, maxiter: FlexGMRES Krylov space `min(restart, 50)`, `restart * maxiter`
            iterations at most
        rebuild: rebuild the BoomerAMG hierarchy at least every this many solves (1: every
            solve; in between the lagged hierarchy preconditions the new matrix)
        growth: rebuild early once the iterations exceed this factor times those right after the
            last rebuild
    """

    def __init__(self, comm, k, n_own, lo, rows, gcols, tol, restart, maxiter, rebuild=1,
                 growth=2.0):
        """Index the `k x k` blocks and the cell blocks; no hypre object yet."""
        self.h = load()
        self.mpi, self.comm = comm, self.h.comm(comm)
        self.k, self.n = int(k), int(n_own)
        self.ilo = int(lo) * self.k
        self.ihi = self.ilo + self.k * self.n - 1
        rows, gcols = np.asarray(rows, np.int64), np.asarray(gcols, np.int64)
        if np.any(np.diff(rows) < 0):
            raise ValueError("BlockRows: entries must come row by row")
        c, p, d, q = rows // k, rows % k, gcols // k, gcols % k
        o = np.lexsort((q, p, d, c))              # (cell, stencil cell, p, q): the k x k blocks
        if len(o) % (k * k):
            raise ValueError("BlockRows: incomplete k x k blocks")
        self.blk = o.reshape(-1, k, k)
        self.blk_cell = c[self.blk[:, 0, 0]]
        diag = np.flatnonzero(d[self.blk[:, 0, 0]] == lo + self.blk_cell)
        self.diag = np.empty(self.n, np.int64)
        self.diag[self.blk_cell[diag]] = diag
        if len(diag) != self.n or len(np.unique(self.blk_cell[diag])) != self.n:
            raise ValueError("BlockRows: every owned cell needs its diagonal block")
        B = self.h.Big
        self.rows = rows
        self.grow, self.gcol = (self.ilo + rows).astype(B), gcols.astype(B)
        self.on_diag = self.grow == self.gcol
        self.local = (gcols >= self.ilo) & (gcols <= self.ihi)
        self.ids = np.arange(self.ilo, self.ihi + 1, dtype=B)
        self.tol, self.kdim = float(tol), int(min(restart, 50))
        self.maxits = int(restart) * int(maxiter)
        self.rebuild, self.growth = int(rebuild), float(growth)
        self.lag = {}                     # transpose -> lagged FlexGMRES + BoomerAMG and its age

    def _matrix(self, S, keep, transpose):
        """The IJ matrix of the scaled values `S` (entries `keep`), or of its transpose."""
        h, Int = self.h, self.h.Int
        M = ctypes.c_void_p()
        h("IJMatrixCreate", self.comm, self.ilo, self.ihi, self.ilo, self.ihi, ctypes.byref(M))
        h("IJMatrixSetObjectType", M, PARCSR)
        nr = self.k * self.n
        if not transpose:
            ncols = np.bincount(self.rows[keep], minlength=nr).astype(Int)
            nd = np.bincount(self.rows[keep & self.local], minlength=nr).astype(Int)
            no = (ncols - nd).astype(Int)
            h("IJMatrixSetDiagOffdSizes", M, _ptr(nd), _ptr(no))
            h("IJMatrixInitialize", M)
            cols, vals = np.ascontiguousarray(self.gcol[keep]), np.ascontiguousarray(S[keep])
            h("IJMatrixSetValues", M, nr, _ptr(ncols), _ptr(self.ids), _ptr(cols), _ptr(vals))
        else:                             # rows of other ranks: added, sent at Assemble
            h("IJMatrixInitialize", M)
            rows = np.ascontiguousarray(self.gcol[keep])
            cols = np.ascontiguousarray(self.grow[keep])
            vals, ones = np.ascontiguousarray(S[keep]), np.ones(int(keep.sum()), Int)
            h("IJMatrixAddToValues", M, len(vals), _ptr(ones), _ptr(rows), _ptr(cols), _ptr(vals))
        h("IJMatrixAssemble", M)
        PA = ctypes.c_void_p()
        h("IJMatrixGetObject", M, ctypes.byref(PA))
        return M, PA

    def _vector(self, values):
        """An IJ vector of the owned rows holding `values`; returns `(ij, par)`."""
        h = self.h
        V, PV = ctypes.c_void_p(), ctypes.c_void_p()
        h("IJVectorCreate", self.comm, self.ilo, self.ihi, ctypes.byref(V))
        h("IJVectorSetObjectType", V, PARCSR)
        h("IJVectorInitialize", V)
        vals = np.ascontiguousarray(values, dtype=np.float64)
        h("IJVectorSetValues", V, len(vals), _ptr(self.ids), _ptr(vals))
        h("IJVectorAssemble", V)
        h("IJVectorGetObject", V, ctypes.byref(PV))
        return V, PV

    def _get(self, V):
        """The owned values of IJ vector `V`."""
        out = np.empty(self.k * self.n)
        self.h("IJVectorGetValues", V, len(out), _ptr(self.ids), _ptr(out))
        return out

    def _solver(self):
        """FlexGMRES preconditioned by one BoomerAMG V-cycle, the settings of foam's and AMReX's
        CPU `hypre_solve` (HMIS, ext+i, hybrid symmetric Gauss-Seidel); returns `(gmres, amg)`.
        """
        h = self.h
        g, a = ctypes.c_void_p(), ctypes.c_void_p()
        h("ParCSRFlexGMRESCreate", self.comm, ctypes.byref(g))
        h("ParCSRFlexGMRESSetKDim", g, self.kdim)
        h("ParCSRFlexGMRESSetMaxIter", g, self.maxits)
        h("ParCSRFlexGMRESSetAbsoluteTol", g, 0.0)
        h("ParCSRFlexGMRESSetLogging", g, 0)
        h("ParCSRFlexGMRESSetPrintLevel", g, 0)
        h("BoomerAMGCreate", ctypes.byref(a))
        for name, v in (("Tol", 0.0), ("MaxIter", 1), ("PrintLevel", 0), ("CoarsenType", 10),
                        ("InterpType", 6), ("PMaxElmts", 4), ("StrongThreshold", 0.25),
                        ("RelaxType", 6), ("NumSweeps", 1), ("RelaxOrder", 0)):
            h("BoomerAMGSet" + name, a, v)
        h("ParCSRFlexGMRESSetPrecond", g, h.amg_solve, h.amg_setup, a)
        return g, a

    def _release(self, transpose):
        """Destroy the lagged solver of `transpose` and the matrix it was set up with."""
        L = self.lag.pop(transpose, None)
        if L is not None:
            self.h("BoomerAMGDestroy", L["amg"])
            self.h("ParCSRFlexGMRESDestroy", L["gmres"])
            self.h("IJMatrixDestroy", L["M"])

    def solve(self, vals, b, transpose=False, nb=None):
        """Solve `A x = b` (`transpose`: `A^T x = b`) for the matrix values `vals` and the
        owned right-hand side `b` (global norm `nb`, if known); collective on `comm`.
        FlexGMRES on `S = D^-1 A` (`D` the `k x k` cell blocks; transposed: `S^T z = b`,
        `x = D^-T z`) to 0.1 `tol`, re-entered with a tighter tolerance until the true relative
        residual is below `tol`.
        """
        h, k, t0 = self.h, self.k, time.perf_counter()
        vals, b = np.asarray(vals, np.float64), np.asarray(b, np.float64)
        A = vals[self.blk]                                     # (blocks, k, k)
        Dm = A[self.diag]
        Dinv = np.linalg.inv(Dm)
        S = np.empty_like(vals)
        S[self.blk] = np.einsum("bpr,brq->bpq", Dinv[self.blk_cell], A)
        keep = (S != 0.0) | self.on_diag
        bs = (b if transpose else np.einsum("cpr,cr->cp", Dinv, b.reshape(-1, k)).reshape(-1))
        M, PA = self._matrix(S, keep, transpose)
        Vb, Pb = self._vector(bs)
        Vx, Px = self._vector(np.zeros_like(bs))
        Vr, Pr = self._vector(np.zeros_like(bs))
        t1 = time.perf_counter()
        L = self.lag.get(transpose)
        if L is not None and (L["age"] >= self.rebuild
                              or L["it"] > self.growth * max(L["it0"], 1)):
            self._release(transpose)
            L = None
        fresh = L is None
        if fresh:
            g, a = self._solver()
            h("ParCSRFlexGMRESSetup", g, PA, Pb, Px)
            L = self.lag[transpose] = dict(gmres=g, amg=a, M=M, age=0, it0=None, it=0)
            STATS["setups"] += 1
        t2 = time.perf_counter()
        if nb is None:
            nb = np.sqrt(self.mpi.allreduce(float(b @ b)))
        x, rel, last, its, tol = np.zeros_like(bs), 0.0, 0.0, 0, 0.1 * self.tol
        if nb > 0.0:
            for it_pass in range(4):
                h("ParCSRFlexGMRESSetTol", L["gmres"], tol)
                h("ParCSRFlexGMRESSolve", L["gmres"], PA, Pb, Px, ok=(256,))  # 256: not converged
                it = ctypes.c_int64(0) if h.Int is np.int64 else ctypes.c_int32(0)
                h("ParCSRFlexGMRESGetNumIterations", L["gmres"], ctypes.byref(it))
                its += int(it.value)
                h("ParCSRMatrixMatvec", 1.0, PA, Px, 0.0, Pr)
                z = self._get(Vx)
                rs = bs - self._get(Vr)                        # the scaled residual
                r = rs if transpose else np.einsum("cpr,cr->cp", Dm, rs.reshape(-1, k)).reshape(-1)
                rel = np.sqrt(self.mpi.allreduce(float(r @ r))) / nb
                if os.environ.get("ZOOMY_KRYLOV_LOG") and self.mpi.Get_rank() == 0:
                    print(f"hypre pass {it_pass}: {int(it.value)} iterations to {tol:.2e}, true "
                          f"residual {rel:.2e}" + (" (new hierarchy)" if fresh else ""), flush=True)
                if (not rel > self.tol or it.value >= self.maxits
                        or (it_pass > 0 and rel > 0.5 * last)):
                    break
                last, tol = rel, tol * 0.5 * self.tol / rel   # the scaled residual understates
            x = (np.einsum("cpr,cp->cr", Dinv, z.reshape(-1, k)).reshape(-1) if transpose
                 else z)
        L["age"], L["it"] = L["age"] + 1, its
        if L["it0"] is None:
            L["it0"] = its
        if L["M"] is not M:
            h("IJMatrixDestroy", M)
        for V in (Vb, Vx, Vr):
            h("IJVectorDestroy", V)
        if self.rebuild <= 1:
            self._release(transpose)
        t3 = time.perf_counter()
        STATS["solves"] += 1
        STATS["iterations"] += its
        STATS["t_assemble"] += t1 - t0
        STATS["t_setup"] += t2 - t1
        STATS["t_solve"] += t3 - t2
        self.last = dict(iterations=its, residual=rel)
        return x

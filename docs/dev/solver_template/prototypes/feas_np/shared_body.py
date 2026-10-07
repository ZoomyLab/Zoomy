"""F1: single-source (xp-style) port of the numpy order-2 HyperbolicSolver hot path
(LSQ-MUSCL + Venkatakrishnan + batched NonconservativeRusanov + SSP-RK2).

The BODY (`make_step`) is written once against ``xp`` + a tiny ``ops`` table
(scatter_add / scatter_max / scatter_min); the per-backend pieces are only the
LOWERED KERNELS (`NumpyKernels` / `JaxKernels`) and the scatter primitives.
Mirrors zoomy_core/fvm/solver_numpy.py:949-1176 + reconstruction.py:474-870
statement for statement (same accumulation order wherever cheap), with the
CORRECT Venkatakrishnan negative branch (numpy's formula).
Flags (default = numpy reference semantics):
  vk_jax_bug      : HEAD jax negative branch  clip(-num/min(den,-eps))  (always 1)
  bounds_lift     : jax bounds convention     2*bf - u_cell
  stage2_aux      : jax RK2 refreshes Qaux from the stage state
  scatter         : "add.at" (reference order) | "bincount" (numpy only)
"""
import numpy as np

# ----------------------------------------------------------------- geometry (setup, numpy)
class Geo:
    def __init__(self, solver):
        """Pull every precomputed array from a set-up numpy HyperbolicSolver
        (setup-time only -> guarantees the same geometry bits as the reference)."""
        m = solver._sim_mesh
        dim = solver.nsm.sm.dimension
        rec = solver._sim_flux_operator.__closure__  # not used; kept explicit below
        from zoomy_core.fvm.reconstruction import LSQMUSCLReconstruction
        r = LSQMUSCLReconstruction(m, dim, limiter=solver.nsm.reconstruction.limiter)
        nc = m.n_inner_cells
        self.dim, self.nc = dim, nc
        self.n_faces = m.n_faces
        self.int_f = r._interior_faces; self.bnd_f = r._boundary_faces
        self.iA = r._iA_int; self.iB = r._iB_int; self.iI = r._iInner_bnd
        self.rA = r._r_Af_int; self.rB = r._r_Bf_int; self.rI = r._r_Af_bnd
        self.bf_cells = np.asarray(m.boundary_face_cells)          # limiter-bounds cells (=r._bf_cells)
        self.ev2 = r._eps_v2
        self.A_T = np.ascontiguousarray(m.lsq_gradQ[:nc].transpose(0, 2, 1)) if False else m.lsq_gradQ[:nc].transpose(0, 2, 1)
        self.nbr = np.asarray(m.lsq_neighbors[:nc])
        self.scale = m.lsq_scale_factors
        self.gcols = r._lsq_grad_cols
        self.has_bdy = r._lsq_has_bdy
        if self.has_bdy:
            b = np.asarray(m.lsq_boundary_face_neighbors[:nc])
            self.bdy_mask = b >= 0
            self.bdy_idx = np.maximum(b, 0)
        nrm = m.face_normals[:dim, :]
        self.n_int = nrm[:, self.int_f]; self.n_bnd = nrm[:, self.bnd_f]
        self.fv_int = m.face_volumes[self.int_f]; self.fv_bnd = m.face_volumes[self.bnd_f]
        cv = m.cell_volumes
        self.cvA = cv[self.iA]; self.cvB = cv[self.iB]; self.cvI = cv[self.iI]
        # BC kernel inputs (solver_numpy.py:870-934)
        self.bc_idx = np.asarray(solver._bc_indices)
        self.bf_kcells = np.asarray(solver._bf_cells)           # cells fed to the BC kernel
        self.bf_fidx = np.asarray(solver._bf_fidx)
        self.d_face = np.asarray(solver._d_face)
        self.bf_pos = np.ascontiguousarray(m.face_centers[self.bf_fidx, :].T)   # (3, nbf)
        self.bf_n = nrm[:, self.bf_fidx]
        self.bnd_pos = np.ascontiguousarray(m.face_centers[self.bnd_f, :].T)
        assert np.array_equal(self.bf_fidx, self.bnd_f), "bf order != boundary-face order"
        self.cc = np.ascontiguousarray(m.cell_centers[:, :nc])

    def to(self, xp):
        g = Geo.__new__(Geo)
        for k, v in self.__dict__.items():
            g.__dict__[k] = xp.asarray(v) if isinstance(v, np.ndarray) else v
        return g

# ----------------------------------------------------------------- scatter primitives
class NumpyOps:
    xp = np
    def __init__(self, scatter="add.at"): self.scatter = scatter
    @staticmethod
    def scatter_add(dQ, idx, vals):             # dQ (nv,nc), vals (nv,nf): np.add.at(dQ.T, idx, vals.T)
        out = dQ.copy(); np.add.at(out.T, idx, vals.T); return out
    @staticmethod
    def scatter_add_bincount(dQ, idx, vals):
        nc = dQ.shape[1]
        return dQ + np.stack([np.bincount(idx, weights=vals[k], minlength=nc) for k in range(vals.shape[0])])
    @staticmethod
    def scatter_max(u, idx, vals):              # u (nv,nc)
        out = u.copy(); np.maximum.at(out.T, idx, vals.T); return out
    @staticmethod
    def scatter_min(u, idx, vals):
        out = u.copy(); np.minimum.at(out.T, idx, vals.T); return out
    @staticmethod
    def matmul(a, b): return np.matmul(a, b)

class JaxOps:
    def __init__(self, scatter="add.at"):
        import jax.numpy as jnp
        self.xp = jnp; self.scatter = scatter
    def scatter_add(self, dQ, idx, vals): return dQ.at[:, idx].add(vals)
    scatter_add_bincount = scatter_add
    def scatter_max(self, u, idx, vals): return u.at[:, idx].max(vals)
    def scatter_min(self, u, idx, vals): return u.at[:, idx].min(vals)
    def matmul(self, a, b): return self.xp.matmul(a, b)

# ----------------------------------------------------------------- the SINGLE-SOURCE body
def make_step(xp, ops, G, K, p, order=2, vk_jax_bug=False, bounds_lift=False,
              stage2_aux=False):
    nv = K.nv; dim = G.dim; nc = G.nc
    eps = 1e-30
    sadd = ops.scatter_add_bincount if ops.scatter == "bincount" else ops.scatter_add

    def lsq_grad(U, Ubf):                                    # (nv,nc),(nv,nbf) -> (nv,dim,nc)
        d_cells = U[:, G.nbr] - U[:, :, None]
        if G.has_bdy:
            ubf = Ubf[:, G.bdy_idx]
            d_bdy = xp.where(G.bdy_mask[None], ubf - U[:, :, None], 0.0)   # GHOST convention (bare)
            delta = xp.concatenate([d_cells, d_bdy], axis=2)
        else:
            delta = d_cells
        coeffs = ops.matmul(G.A_T[None], delta[..., None])[..., 0]           # (nv,nc,nmon)
        coeffs = G.scale * coeffs
        return xp.transpose(coeffs[:, :, G.gcols], (0, 2, 1))

    def face_delta(grad, cells, r):                          # (nv,nf)
        d = xp.zeros((nv, cells.shape[0]))
        for k in range(dim):
            d = d + grad[:, k, cells] * r[k][None, :]
        return d

    def vk(U, grad, umin, umax):
        phi = xp.ones((nv, nc))
        for cells, r in ((G.iA, G.rA), (G.iB, G.rB), (G.iI, G.rI)):
            dl = face_delta(grad, cells, r)
            uc = U[:, cells]; e2 = G.ev2[cells][None, :]
            dmp = umax[:, cells] - uc; dmn = umin[:, cells] - uc
            nump = dmp ** 2 + e2 + 2 * dl * dmp; denp = dmp ** 2 + 2 * dl ** 2 + dl * dmp + e2
            numn = dmn ** 2 + e2 + 2 * dl * dmn; denn = dmn ** 2 + 2 * dl ** 2 + dl * dmn + e2
            cp = xp.minimum(1.0, nump / xp.maximum(denp, eps))
            if vk_jax_bug:
                cn = xp.clip(-numn / xp.minimum(denn, -eps), 0.0, 1.0)
            else:
                cn = xp.minimum(1.0, numn / xp.maximum(denn, eps))
            cand = xp.where(dl > eps, cp, xp.where(dl < -eps, cn, 1.0))
            phi = ops.scatter_min(phi, cells, cand)
        return xp.clip(phi, 0.0, 1.0)

    def bounds(U, bf):
        umax = ops.scatter_max(U, G.iA, U[:, G.iB]); umax = ops.scatter_max(umax, G.iB, U[:, G.iA])
        umin = ops.scatter_min(U, G.iA, U[:, G.iB]); umin = ops.scatter_min(umin, G.iB, U[:, G.iA])
        bv = 2.0 * bf - U[:, G.bf_cells] if bounds_lift else bf
        return ops.scatter_min(umin, G.bf_cells, bv), ops.scatter_max(umax, G.bf_cells, bv)

    def reconstruct(Q, bf):
        if order < 2:
            return Q[:, G.iA], Q[:, G.iB], Q[:, G.iI], None
        grad = lsq_grad(Q, bf)
        umin, umax = bounds(Q, bf)
        phi = vk(Q, grad, umin, umax)
        QL = Q[:, G.iA]; QR = Q[:, G.iB]; QI = Q[:, G.iI]
        for k in range(dim):
            QL = QL + phi[:, G.iA] * grad[:, k, G.iA] * G.rA[k][None, :]
            QR = QR + phi[:, G.iB] * grad[:, k, G.iB] * G.rB[k][None, :]
        for k in range(dim):
            QI = QI + phi[:, G.iI] * grad[:, k, G.iI] * G.rI[k][None, :]
        return QL, QR, QI, phi[:, None, :] * grad

    def flux_rhs(t, Q, Qaux):
        bf = K.bc(G.bc_idx, t, G.bf_pos, G.d_face, Q[:, G.bf_kcells], Qaux[:, G.bf_kcells], p, G.bf_n)
        QL, QR, QI, lgrad = reconstruct(Q, bf)
        aI = Qaux[:, G.iI]
        QRb = K.bc(G.bc_idx, t, G.bf_pos, G.d_face, QI, aI, p, G.bf_n)
        dQ = xp.zeros((nv, nc))
        if lgrad is not None and K.has_ncm:
            B = K.ncm(Q, Qaux, p)                                   # (nv,nv,dim,nc)
            for k in range(dim):
                Bk = xp.transpose(B[:, :, k, :], (2, 0, 1))          # (nc,nv,nv)
                dQ = dQ - ops.matmul(Bk, xp.transpose(lgrad[:, k, :])[:, :, None])[:, :, 0].T
        aA = Qaux[:, G.iA]; aB = Qaux[:, G.iB]
        F = K.flux(QL, QR, aA, aB, p, G.n_int)
        Dp, Dm = K.fluct(QL, QR, aA, aB, p, G.n_int)
        dQ = sadd(dQ, G.iA, -(F + Dm) * G.fv_int / G.cvA)
        dQ = sadd(dQ, G.iB, -(-F + Dp) * G.fv_int / G.cvB)
        aR = K.ghost_aux(QRb, aI, p, t, G.bnd_pos)
        Fb = K.flux(QI, QRb, aI, aR, p, G.n_bnd)
        _, Dmb = K.fluct(QI, QRb, aI, aR, p, G.n_bnd)
        dQ = sadd(dQ, G.iI, -(Fb + Dmb) * G.fv_bnd / G.cvI)
        return dQ

    def rhs(t, Q, Qaux, dt):
        return flux_rhs(t, Q, Qaux) + K.source(Q, Qaux, p, t, dt, G.cc)

    def step(Q, Qaux, t, dt):
        if order >= 2:
            Q1 = Q + dt * rhs(t, Q, Qaux, dt)
            A1 = K.update_aux(Q1, Qaux, p, t, G.cc) if stage2_aux else Qaux
            Q2 = Q1 + dt * rhs(t + dt, Q1, A1, dt)
            Qn = 0.5 * (Q + Q2)
        else:
            Qn = Q + dt * rhs(t, Q, Qaux, dt)
        Qn = K.update_q(Qn, Qaux, p, dt)
        An = K.update_aux(Qn, Qaux, p, t, G.cc)
        return Qn, An

    return step, flux_rhs

# ----------------------------------------------------------------- lowered kernels per backend
def _barray(x, *a, **k):
    if isinstance(x, (list, tuple)):
        return np.array(np.broadcast_arrays(*[_barray(e) for e in x]), *a, **k)
    return np.asarray(x, *a, **k)

class NumpyKernels:
    def __init__(self, solver):
        from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
        sm = solver.nsm.sm
        self.nv = len(sm.state); self.na = sm.n_aux_variables
        mod = dict(NumpyRuntimeModel.module); mod["array"] = _barray      # F4 broadcasting fix
        rt_b = NumpyRuntimeModel.from_system_model(sm, module=mod)
        self._bc = rt_b.boundary_conditions
        rt = solver._sim_model
        self._rn = solver._build_numerics(solver._get_symbolic_model(rt)).to_runtime_numpy()
        self._ncm = NumpyRuntimeModel.from_system_model(sm).nonconservative_matrix
        self.has_ncm = True
        self._rt = rt
    def bc(self, idx, t, pos, d, Q, A, p, n):
        return np.asarray(self._bc(idx, t, pos, d, Q, A, p, n), float).reshape(self.nv, -1)
    def flux(self, a, b, aa, ab, p, n):
        return np.asarray(self._rn.numerical_flux(a, b, aa, ab, p, n), float).reshape(self.nv, -1)
    def fluct(self, a, b, aa, ab, p, n):
        f = np.asarray(self._rn.numerical_fluctuations(a, b, aa, ab, p, n), float).reshape(2 * self.nv, -1)
        return f[:self.nv], f[self.nv:]
    def ncm(self, Q, A, p):
        B = np.asarray(self._ncm(Q, A, p), float)   # numpy runtime: (nv, nv, nc, dim)
        return np.moveaxis(B, 3, 2)                 # -> jax layout (nv, nv, dim, nc)
    def ghost_aux(self, QR, AI, p, t, pos):
        return np.asarray(self._rt.update_aux_variables(QR, AI, p, t, pos), float).reshape(self.na, -1)
    def source(self, Q, A, p, t, dt, cc):
        return np.asarray(self._rt.source(Q, A, p, t, cc), float).reshape(self.nv, -1)
    def update_q(self, Q, A, p, dt):
        return np.asarray(self._rt.update_variables(Q, A, p, dt), float)[:self.nv]
    def update_aux(self, Q, A, p, t, cc):
        return np.asarray(self._rt.update_aux_variables(Q, A, p, t, cc), float).reshape(self.na, -1)

class JaxKernels:
    def __init__(self, nsm):
        import jax, jax.numpy as jnp
        from zoomy_jax.transformation.jax_runtime import JaxRuntime
        self.jax, self.jnp = jax, jnp
        rt = JaxRuntime.from_nsm(nsm); self.rt = rt
        self.nv = len(nsm.sm.state); self.na = nsm.sm.n_aux_variables
        self.has_ncm = getattr(rt, "nonconservative_matrix", None) is not None
    def bc(self, idx, t, pos, d, Q, A, p, n):
        f = lambda i, x, dd, q, a, nn: self.rt.boundary_conditions(i, t, x, dd, q, a, p, nn)
        return self.jax.vmap(f, in_axes=(0, 1, 0, 1, 1, 1))(idx, pos, d, Q, A, n).T
    def flux(self, a, b, aa, ab, p, n):
        return self.rt.numerical_flux(a, b, aa, ab, p, n)
    def fluct(self, a, b, aa, ab, p, n):
        f = self.rt.numerical_fluctuations(a, b, aa, ab, p, n)
        return f[0], f[1]
    def ncm(self, Q, A, p):
        return self.rt.nonconservative_matrix(Q, A, p)
    def ghost_aux(self, QR, AI, p, t, pos):
        return self.rt.update_aux_variables(QR, AI, p, time=t, position=pos)
    def source(self, Q, A, p, t, dt, cc):
        return self.rt.source(Q, A, p, time=t, dt=dt, position=cc)
    def update_q(self, Q, A, p, dt):
        return self.rt.update_variables(Q, A, p)
    def update_aux(self, Q, A, p, t, cc):
        return self.rt.update_aux_variables(Q, A, p, time=t, position=cc)

"""Symbolic Riemann solvers over a SystemModel (Rusanov, HLL, HLLC, positive and
path-conservative variants) whose face kernels feed the NumericalSystemModel and the
solver templates. Classes needing the depth `h`, bed `b` or `hinv` find them in
`Q` or `Qaux` through `FieldHandle`.
"""

import numpy as np
import param
import sympy as sp

from zoomy_core.misc.misc import ZArray, Zstruct
from zoomy_core.misc.options import typed
from zoomy_core.model.basefunction import SymbolicRegistrar
from zoomy_core.model.kernel_functions import (
    conditional,
)
from zoomy_core.systemmodel.system_model import (
    SystemModel, _perron_radius_bound, SpectralRadius)
from zoomy_core.transformation.to_numpy import (
    NumpyRuntimeModel, NumpyRuntimeSymbolic,
)


def _spectral_core(M):
    """Indices left after repeatedly dropping a state whose row or column of the square
    symbolic matrix `M` is structurally zero: the matrix is then block triangular with a
    zero block, so the kept block has the same spectral radius (`b`, VAM's pressure rows).
    """
    keep = list(range(len(M)))
    dropped = True
    while dropped:
        dropped = False
        for i in list(keep):
            if all(M[i][j] == 0 for j in keep) or all(M[j][i] == 0 for j in keep):
                keep.remove(i)
                dropped = True
    return keep or [0]


def _bed_column(model):
    """`{bed index}` of the model's state (the declared bed role, else the state `b`), or
    the empty set."""
    roles = getattr(model, "roles", None)
    bed = getattr(roles, "bed", None) if roles is not None else None
    if bed is None:
        names = [str(s) for s in model.state]
        bed = names.index("b") if "b" in names else None
    return frozenset() if bed is None else frozenset((int(bed),))


def dropped_nonconservative_entries(riemann, model):
    """The live entries `(row, column, direction)` of `model.nonconservative_matrix` whose
    product `B grad Q` the face terms of the Riemann class `riemann` do not discretize
    (outside `treated_nonconservative_columns`): a scheme that runs with one drops a term
    of the PDE without any error.
    """
    B = getattr(model, "nonconservative_matrix", None)
    columns = getattr(riemann, "treated_nonconservative_columns", None)
    if B is None or columns is None:
        return []
    treated = columns(model)
    B = sp.Array(B)
    if B.rank() != 3:
        return []
    n_eq, n_st, dim = B.shape
    return [(i, j, k) for i in range(n_eq) for j in range(n_st) if j not in treated
            for k in range(dim) if sp.sympify(B[i, j, k]).is_zero is not True]


class FieldHandle:
    """A named field resolved to a slot of `Q` or `Qaux`, exposing its face symbols
    `minus` / `plus` and cell symbol `state`. Built by
    `Numerics.find_field`.
    """

    __slots__ = ("name", "container", "index", "minus", "plus", "state")

    def __init__(self, name, container, index, minus, plus, state):
        """Store the field's name, container (`"q"` or `"qaux"`), index and symbols.
        """
        if container not in {"q", "qaux"}:
            raise ValueError(
                f"FieldHandle({name}): container must be 'q' or 'qaux', "
                f"got {container!r}"
            )
        self.name = name
        self.container = container
        self.index = index
        self.minus = minus
        self.plus = plus
        self.state = state

    def access(self, q_array, qaux_array):
        """`q_array[index]` or `qaux_array[index]`, whichever container holds the
        field.
        """
        return (q_array[self.index] if self.container == "q"
                else qaux_array[self.index])

    def assign(self, q_array, qaux_array, value):
        """Write `value` in place into the field's container."""
        if self.container == "q":
            q_array[self.index] = value
        else:
            qaux_array[self.index] = value

    def __repr__(self):
        """`FieldHandle('h': q[0])`."""
        return (f"FieldHandle({self.name!r}: "
                f"{self.container}[{self.index}])")


class Numerics(param.Parameterized, SymbolicRegistrar):
    """Symbolic numerics over a `SystemModel`: the base class holds the field
    handles, registers the face kernels and on its own is the zero scheme. The
    Riemann solvers below subclass it.
    """

    name = param.String(default="NumericsV2")
    model = param.Parameter(
        default=None,
        doc="The SystemModel the numerics operates on (a Model is "
            "normalised via SystemModel.from_model in __init__).",
    )

    scaled_q_indices = param.List(default=None, allow_None=True)

    def __init__(self, model, **params):
        """Build the numerics for `model` and register its kernels (flux,
        fluctuations, face sides, star state, ...).

        Args:
            model: a SystemModel, or a Model normalised once via
                `SystemModel.from_model`.
        """
        sm = (model if isinstance(model, SystemModel)
              else SystemModel.from_model(model))
        super().__init__(model=sm, **params)
        self.functions, self.call = Zstruct(), Zstruct()

        self.n_variables = self.model.n_equations
        self.n_aux_variables = len(self.model.aux_state)

        self.variables = ZArray(list(self.model.state))
        self.aux_variables = ZArray(list(self.model.aux_state))
        # Parameter / normal symbols (used in symbolic derivation).
        self.parameters = ZArray(list(self.model.parameters.values()))
        self.normal = ZArray(list(self.model.normal.values()))

        self.variables_minus = self._create_v("Q_minus", self.n_variables)
        self.variables_plus = self._create_v("Q_plus", self.n_variables)
        self.aux_variables_minus = self._create_v(
            "Qaux_minus", self.n_aux_variables)
        self.aux_variables_plus = self._create_v(
            "Qaux_plus", self.n_aux_variables)

        self.flux_minus = self._create_v("flux_minus", self.n_variables)
        self.flux_plus = self._create_v("flux_plus", self.n_variables)
        self.source_term = self._create_v("source_term", self.n_variables)

        # The free-surface handles ``h`` (depth) and ``b`` (bed) come from the
        # model's DECLARED ROLES (``SystemModel.roles``) — the handle keys keep
        # their historical names so the free-surface-only solvers below
        # (Audusse HR, HLLC, WB-NC-Rusanov) read them unchanged.  Only a system
        # that never declared a depth role falls back to the conventional
        # names (legacy hand-built systems).  ``hinv`` is the aux the core
        # depth operation itself registers under that name.
        self._field_handles = {}
        roles = getattr(self.model, "roles", None)
        if roles is not None and roles.depth_declared:
            for key, row in (("h", roles.depth), ("b", roles.bed)):
                if row is not None:
                    self._field_handles[key] = FieldHandle(
                        key, "q", row, minus=self.variables_minus[row],
                        plus=self.variables_plus[row],
                        state=self.variables[row])
            legacy = ("hinv",)
        else:
            legacy = ("h", "b", "hinv")
        for name in legacy:
            h = self.find_field(name, required=False)
            if h is not None:
                self._field_handles[name] = h
        # Rows no dissipation may act on: the declared ``stationary`` role, or
        # (undeclared) the STRUCTURALLY stationary rows — never "the row named
        # b" (a transported tracer named ``b`` keeps its dissipation).
        self._stationary_rows = (
            tuple(r for r in roles.stationary if r < self.n_variables)
            if roles is not None else ())
        self._scaled_q_indices = self._resolve_scaled_q_indices(
            self.scaled_q_indices)

        self._initialize_functions()

    # ── Field-search API ──────────────────────────────────────────

    def find_field(self, name, *, required=True):
        """The `FieldHandle` of `name`, searching `model.state` then
        `model.aux_state` (cached).

        Args:
            required: raise `KeyError` if the field is absent; else return `None`.
        """
        cache = getattr(self, "_field_handles", None) or {}
        if name in cache:
            return cache[name]
        state_list = list(self.model.state)
        state_names = [str(s) for s in state_list]
        if name in state_names:
            i = state_names.index(name)
            h = FieldHandle(name, "q", i,
                            minus=self.variables_minus[i],
                            plus=self.variables_plus[i],
                            state=self.variables[i])
            cache[name] = h
            return h
        aux_list = list(self.model.aux_state)
        aux_names = [str(s) for s in aux_list]
        if name in aux_names:
            i = aux_names.index(name)
            h = FieldHandle(name, "qaux", i,
                            minus=self.aux_variables_minus[i],
                            plus=self.aux_variables_plus[i],
                            state=self.aux_variables[i])
            cache[name] = h
            return h
        if required:
            raise KeyError(
                f"Field {name!r} not found in model.state "
                f"({state_names}) nor model.aux_state ({aux_names})"
            )
        return None

    def has_field(self, name):
        """Whether the model has a field `name`."""
        return self.find_field(name, required=False) is not None

    def _declared_depth_handle(self):
        """The depth handle, only when the model declares a depth role."""
        roles = getattr(self.model, "roles", None)
        if roles is None or roles.depth is None:
            return None
        return self._field_handles.get("h")

    def _create_v(self, name, size):
        """Symbol vector `name_i` of the given size."""
        v = ZArray([sp.Symbol(f"{name}_{i}", real=True) for i in range(size)])
        v._symbolic_name = name
        return v

    def _resolve_scaled_q_indices(self, scaled_q_indices):
        """Depth-scaled state rows: the explicit argument, else the model's own
        `depth_scaled_state_indices`.
        """
        if scaled_q_indices is not None:
            cleaned = [int(i) for i in scaled_q_indices]
            for i in cleaned:
                if i < 0 or i >= self.n_variables:
                    raise IndexError(
                        f"scaled_q_indices contains out-of-bounds index {i}."
                    )
            return cleaned
        return list(self.model.depth_scaled_state_indices)

    def _eps_symbol(self):
        """The wet/dry threshold `wet_dry_eps` if the model declares it, else the
        floor `1e-12`.
        """
        if self.model.parameters.contains("wet_dry_eps"):
            return self.model.parameters.wet_dry_eps
        return sp.Float(1e-12)

    def _initialize_functions(self):
        """Register the face kernels (flux, fluctuations, face sides, cell-edge source,
        wave speed, star state).
        """
        sig = Zstruct(
            q_minus=self.variables_minus,
            q_plus=self.variables_plus,
            aux_minus=self.aux_variables_minus,
            aux_plus=self.aux_variables_plus,
            p=self.parameters,
            normal=self.normal,
        )

        # Build each face definition ONCE.  The registered names and the
        # fused ``numerical_face`` are registered from the *same* expression
        # objects, so the fused kernel is a repack of the legacy ones — never
        # a re-derivation.  Registering a cached definition is byte-identical
        # to registering the bound method (``register_symbolic_function``
        # calls it exactly once either way).
        flux = self.numerical_flux()
        fluctuations = self.numerical_fluctuations()

        # The REGISTERED ``numerical_flux`` is the flux and NOTHING else —
        # shape ``(n_variables, 1)``.  It briefly carried an EXTRA trailing
        # ``face_max_abs_eigenvalue`` row (REQ-212) so a driver could take its
        # local dt straight off the face kernel; the approved v6 solver design
        # removes it again.  Eigenvalues are now produced at the STEP HEAD by
        # a dedicated dt_pass that stores two scalars per face, so no consumer
        # has to unpack a wave speed out of a flux.
        #
        # That extra row was never actually consumable: every hand-written
        # backend header indexes the result FLAT as ``F[i]`` over n rows
        # (zoomy_foam/numerics.H, numerics_o2.H), UFL dots it against an
        # n-component test function (zoomy_firedrake), and jax subtracts it
        # from an (n, nfaces) residual — only zoomy_amrex was adapted, and it
        # was told the ``n_dof_q + 1`` accommodation was temporary.
        #
        # ``numerical_fluctuations`` is unchanged.
        flux_rows = list(sp.flatten(flux))
        if len(flux_rows) != self.n_variables:
            raise ValueError(
                f"numerical_flux: expected {self.n_variables} flux rows, got "
                f"{len(flux_rows)} (shape {getattr(flux, 'shape', None)})."
            )
        flux_only = ZArray(flux_rows).reshape(self.n_variables, 1)

        self.register_symbolic_function(
            "numerical_flux", lambda: flux_only, sig
        )
        self.register_symbolic_function(
            "numerical_fluctuations", lambda: fluctuations, sig
        )
        self.register_symbolic_function(
            "numerical_face", lambda: self.pack_face(flux, fluctuations), sig
        )
        self.register_symbolic_function(
            "numerical_face_sides", lambda: self.face_sides(flux, fluctuations),
            sig
        )
        self.register_symbolic_function(
            "numerical_face_sides_speed",
            lambda: self.face_sides_speed(flux, fluctuations), sig
        )
        # Same SHAPE of signature as the face kernels (two states + aux +
        # parameters + normal), so no printer or contract change is needed —
        # but the pair is reinterpreted as (cell, edge) rather than the two
        # sides of a face, and the normal points OUT of that cell.
        self.register_symbolic_function(
            "numerical_cell_edge_source",
            self.numerical_cell_edge_source, sig
        )

        eig_sig = Zstruct(
            Q=self.variables, Qaux=self.aux_variables, p=self.parameters, n=self.normal
        )
        self.register_symbolic_function(
            "local_max_abs_eigenvalue", self.local_max_eigenvalue_definition, eig_sig
        )

        # ``star_state`` carries the SAME signature as the face kernels
        # (two states + aux + parameters + normal), so every printer that
        # walks ``numerics.functions`` emits it with no printer change:
        # ``FoamNumericsPrinter._FOAM_NUMERICS_ARG`` and
        # ``GenericCppBase.ARG_MAPPING`` already spell all six keys.
        #
        # Registered LAST on purpose.  The C-family printers emit one kernel
        # per entry in INSERTION order, so appending keeps every existing
        # block at its existing position and the emitted header gains a
        # trailing function rather than a reshuffle.
        self.register_symbolic_function("star_state", self.star_state, sig)

    def face_sides(self, flux, fluctuations):
        """The face's contribution to each adjacent cell, shape `(2, n)`: row 0
        `F + D_minus` into the owner and row 1 `-F + D_plus` into the neighbour,
        each scaled by `|f|/|V|` and subtracted. It recombines the flux and
        fluctuation expressions exactly, so one printed silo shares the Riemann
        problem.

        References:
            Pares 2006, https://doi.org/10.1137/050628052
        """
        F = list(sp.flatten(flux))
        D = list(sp.flatten(fluctuations))
        n = self.n_variables
        if len(F) != n or len(D) != 2 * n:
            raise ValueError(
                f"face_sides: expected {n} flux and {2 * n} fluctuation rows, "
                f"got {len(F)} and {len(D)}")
        return ZArray([[F[i] + D[n + i] for i in range(n)],
                       [-F[i] + D[i] for i in range(n)]])

    def face_wave_speed(self):
        """The fastest signal speed `max(|s_L|, |s_R|)` of the scheme's own Riemann
        solver, at the states its dissipation is built on (the reconstructed ones
        where it has them). It uses the same expressions as the flux, so a printed
        silo computes them once.
        """
        vm, vp = self.variables_minus, self.variables_plus
        am, ap = self.aux_variables_minus, self.aux_variables_plus
        if hasattr(self, "get_path_integral_states"):
            vm, vp, am, ap = self.get_path_integral_states()
        elif hasattr(self, "hydrostatic_reconstruction"):
            vm, vp, am, ap = self.hydrostatic_reconstruction(vm, vp, am, ap)
        sL, sR = self.wave_speed_bounds(vm, vp, am, ap, self.parameters,
                                        self.normal)
        if sL + sR == 0:                           # the Rusanov pair (-a, a)
            return sR
        # max(|s_L|, |s_R|) from the clamped pair the HLL flux itself builds
        # (min(s_L, 0), max(s_R, 0)), so the printed silo shares it; the Abs makes a dry
        # face's speed +0, never -0 (a CFL step of -inf)
        return sp.Abs(sp.Max(-sp.Min(sL, sp.Integer(0)), sp.Max(sR, sp.Integer(0)),
                             evaluate=False), evaluate=False)

    def face_sides_speed(self, flux, fluctuations):
        """`face_sides` plus a third row carrying `face_wave_speed`, shape
        `(3, n)`, so a solver can take its CFL step from the Riemann problem it
        already solves.
        """
        sides = self.face_sides(flux, fluctuations)
        s = self.face_wave_speed()
        n = self.n_variables
        return ZArray([[sides[0, i] for i in range(n)],
                       [sides[1, i] for i in range(n)],
                       [s] * n])

    def pack_face(self, flux, fluctuations):
        """Pack one face's Riemann answer as
        `[flux(n) | D_plus(n) | D_minus(n) | lambda_max(1)]`, shape
        `(3n + 1, 1)`. The flux row alone is not a scheme for the nonconservative
        variants, whose dissipation lives in `D_plus` / `D_minus`.

        Args:
            flux: the built `numerical_flux`, shape `(n,)`.
            fluctuations: the built `numerical_fluctuations`, shape `(2, n)`.
        """
        lambda_max = self.face_max_abs_eigenvalue(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
            self.parameters,
            self.normal,
        )
        rows = list(sp.flatten(flux))
        rows += list(sp.flatten(fluctuations))
        rows.append(lambda_max)
        expected = 3 * self.n_variables + 1
        if len(rows) != expected:
            raise ValueError(
                f"pack_face: expected {expected} rows "
                f"(3·{self.n_variables} + 1), got {len(rows)} — "
                f"numerical_flux has shape {getattr(flux, 'shape', None)} and "
                f"numerical_fluctuations {getattr(fluctuations, 'shape', None)}; "
                "the packed layout requires (n,) and (2, n)."
            )
        return ZArray(rows).reshape(expected, 1)

    def face_max_abs_eigenvalue(self, qL, qR, auxL, auxR, p, n):
        """Face wave speed `max(rho(A_n)|_L, rho(A_n)|_R)`, the Rusanov (local
        Lax-Friedrichs) dissipation speed; the outer `Max` is unevaluated, a
        run-time reduction.
        """
        return sp.Max(
            self.local_max_abs_eigenvalue(qL, auxL, p, n),
            self.local_max_abs_eigenvalue(qR, auxR, p, n),
            evaluate=False,
        )

    def local_max_eigenvalue_definition(self):
        """Spectral radius of the normal-projected system at the numerics' own
        `Q/Qaux/p/n` symbols, the kernel body the printers emit.
        """
        return self.local_max_abs_eigenvalue(
            self.variables, self.aux_variables,
            self.parameters, self.normal,
        )

    def local_max_abs_eigenvalue(self, Q=None, Qaux=None, p=None, n=None):
        """Spectral radius at `(Q, Qaux, p, n)`: `Max(|lambda_i|)` over the model's
        closed-form eigenvalues, else per `spectral_radius_method` an `eig`, the Gelfand
        upper bound (the symbolic solvers' default) or the Perron bound. It sets the
        Rusanov / HLL dissipation and the CFL step.

        References:
            Horn, Johnson 2012, https://doi.org/10.1017/CBO9781139020411
        """
        if Q is None:
            return self.local_max_eigenvalue_definition()
        if self.model.eigenvalues is None:
            # No closed form.  DEFAULT is the exact numerical spectrum, because
            # this quantity sets the Rusanov / LF / HLL DISSIPATION as well as
            # the CFL, and an upper bound is extra numerical diffusion there
            # (see SystemModel.use_spectral_radius_bound for the measured cost
            # on the Ritter dry dam break).  The cheap bound is opt-in.  The
            # symbolic solvers set `gelfand` (zoomy_core.symsolver.WAVE_SPEED).
            method = self._spectral_method()
            if method == "bound":
                return self._spectral_radius_bound(Q, self._wave_speed_aux(Q, Qaux), p, n)
            if method == "gelfand":
                return self._gelfand(self._speed_matrix(Q, Qaux, p, n))
            src = list(sp.flatten(self._model_eval(
                "numerical_eigenvalues", Q, self._wave_speed_aux(Q, Qaux), p, n)))
            return sp.Max(*[sp.Abs(lam) for lam in src], evaluate=False)
        src = list(sp.flatten(self._model_eval("eigenvalues", Q, Qaux, p, n)))
        # ``evaluate=False``: ``Max`` over a wave spectrum is a runtime
        # reduction, never a symbolic simplification target.  Letting sympy
        # evaluate it triggers the O(n²) ``_find_localzeros`` →
        # ``factor_terms(λ_i − λ_j)`` pairwise ordering attempt — which cannot
        # succeed (non-orderable wave speeds) yet expands each huge
        # Jacobian-eigenvalue tree, exploding SETUP.  The unevaluated node
        # lowers to ``np.amax``/the backend max UNCHANGED (max is exact +
        # associative → bit-identical numerics).
        return sp.Max(*[sp.Abs(lam) for lam in src], evaluate=False)

    def _spectral_method(self):
        """The `SpectralRadius` of a model without a closed-form spectrum."""
        if getattr(self.model, "use_spectral_radius_bound", SystemModel.use_spectral_radius_bound):
            return SpectralRadius.bound
        return SpectralRadius(typed(
            SpectralRadius, getattr(self.model, "spectral_radius_method",
                                    SystemModel.spectral_radius_method),
            "spectral_radius_method"))

    def _speed_matrix(self, Q, Qaux, p, n, normal_frame=False):
        """`A_n = sum_d A[:, :, d] n_d` at `(Q, Qaux)` (the wave-speed aux), restricted to the
        rows and columns that carry its spectrum (`_spectral_core`), as a list of rows; with
        `normal_frame` in the face's (normal, tangential) frame (`_normal_frame`).
        """
        A = self._model_eval("quasilinear_matrix", Q, self._wave_speed_aux(Q, Qaux), p)
        nrm = list(n)
        M = [[sum(A[i, j, d] * nrm[d] for d in range(len(nrm)))
              for j in range(self.n_variables)] for i in range(self.n_variables)]
        if normal_frame:
            M = self._normal_frame(M, nrm)
        keep = _spectral_core(M)
        return [[M[i][j] for j in keep] for i in keep]

    def _normal_frame(self, M, nrm):
        """`T M T^T`, `T` rotating each horizontal vector `(q_x, q_y)` of the state to `(q.n, q.t)`,
        `t = (-n_y, n_x)`: a similarity (same spectrum), in which a wall's reflected state has a
        signed permutation of the inner state's matrix, so the bounds of the two are mirror images
        and a wall's HLL mass flux vanishes.  2 horizontal directions; else `M`.
        """
        from zoomy_core.model.boundary_conditions import derive_momentum_field_indices
        try:
            groups = derive_momentum_field_indices(self.model)
        except RuntimeError:
            return M
        if len(nrm) != 2 or any(len(g) != 2 for g in groups):
            return M
        T = sp.eye(len(M))
        for i, j in groups:
            T[i, i], T[i, j], T[j, i], T[j, j] = nrm[0], nrm[1], -nrm[1], nrm[0]
        R = T * sp.Matrix(M) * T.T
        return [[R[i, j] for j in range(len(M))] for i in range(len(M))]

    def _gelfand(self, M):
        """`gelfand_bound` of the square matrix `M` (list of rows): an upper bound of rho(M)."""
        from zoomy_core.model.kernel_functions import gelfand_bound
        return gelfand_bound(
            sp.Integer(getattr(self.model, "spectral_squarings",
                               SystemModel.spectral_squarings)),
            sp.Float(getattr(self.model, "spectral_cw_offset",
                             SystemModel.spectral_cw_offset)),
            *[x for row in M for x in row])

    def _gelfand_extremes(self, qL, qR, auxL, auxR, p, n):
        """`(s_L, s_R)` with `s_L <= min Re lambda` and `s_R >= max Re lambda` over the
        spectra of `A_n` at both face states, from Gelfand bounds of shifted matrices: for any
        real `c`, `rho(A + c I) >= max_i |lambda_i + c| >= max_i Re lambda_i + c`, so
        `s_R = bound(A + a I) - a`, `s_L = a - bound(a I - A)`; `a` the face's bound of rho (the
        Rusanov speed), both clamped to `[-a, a]`; the shifted matrices in the face's normal
        frame, so a wall's pair is symmetric (no mass through it).  Branch-free, every backend.

        References:
            Horn, Johnson 2012 (sections 5.6, 8.1), https://doi.org/10.1017/CBO9781139020411
            Davis 1988, https://doi.org/10.1137/0909030
        """
        a = self.face_max_abs_eigenvalue(qL, qR, auxL, auxR, p, n)
        shifted = []
        for sign in (1, -1):                     # bound(A + a I), bound(a I - A), both states
            per_state = []
            for q, aux in ((qL, auxL), (qR, auxR)):
                M = self._speed_matrix(q, aux, p, n, normal_frame=True)
                per_state.append(self._gelfand(
                    [[sign * x + (a if i == j else 0) for j, x in enumerate(row)]
                     for i, row in enumerate(M)]))
            shifted.append(sp.Max(*per_state, evaluate=False))
        sR = sp.Min(shifted[0] - a, a, evaluate=False)
        sL = sp.Max(a - shifted[1], -a, evaluate=False)
        return sL, sR

    def _wave_speed_aux(self, Q, Qaux):
        """`Qaux` with the KP `hinv` replaced by the `1/max(1e-14, h)` of the closed-form
        eigenvalues (`wave_speed_inverse_depth`, REQ-82): the KP value underestimates the speed
        at a thin reconstructed face and drains cells negative (ANALYSIS 48).
        """
        hinv = self.find_field("hinv", required=False)
        h = self.find_field("h", required=False)
        if hinv is None or h is None or hinv.container != "qaux":
            return Qaux
        from zoomy_core.systemmodel.operations import wave_speed_inverse_depth
        out = list(Qaux)
        out[hinv.index] = wave_speed_inverse_depth(h.access(Q, Qaux))
        return out

    def _spectral_radius_bound(self, Q, Qaux, p, n):
        """The model's spectral radius bound at the numerics' state symbols, built from
        the substituted quasilinear matrix so its unevaluated reductions survive.
        """
        A = self._model_eval("quasilinear_matrix", Q, Qaux, p)
        nrm = list(n)
        dim = len(nrm)
        M = [[sp.Abs(sum(A[i, j, d] * nrm[d] for d in range(dim)))
              for j in range(self.n_variables)]
             for i in range(self.n_variables)]
        return _perron_radius_bound(
            M, getattr(self.model, "spectral_bound_iterations",
                       SystemModel.spectral_bound_iterations))

    def _model_eval(self, operator_name, Q, Qaux, p, n=None):
        """Evaluate the SystemModel operator `operator_name` (`flux`,
        `eigenvalues`, ...), or a definition over the model's symbols, at the given
        state by substituting its symbols.
        """
        definition = (getattr(self.model, operator_name) if isinstance(operator_name, str)
                      else operator_name)
        sub_map = {}
        for sym, val in zip(self.model.state, list(Q)):
            sub_map[sym] = val
        for sym, val in zip(self.model.aux_state, list(Qaux)):
            sub_map[sym] = val
        for sym, val in zip(self.model.parameters.values(), list(p)):
            sub_map[sym] = val
        if n is not None:
            for sym, val in zip(self.model.normal.values(), list(n)):
                sub_map[sym] = val

        if isinstance(definition, sp.NDimArray):
            # ``NDimArray`` exposes ``subs`` only on the *immutable*
            # variant; ``SystemModel`` stores the nonconservative /
            # quasilinear tensors as mutable arrays, so normalise to
            # immutable before substituting (otherwise the state
            # symbols leak through unsubstituted).
            out = sp.ImmutableDenseNDimArray(definition).subs(sub_map)
        elif hasattr(definition, "subs"):
            out = definition.subs(sub_map)
        else:
            out = definition

        if isinstance(out, ZArray):
            return out
        if isinstance(out, (sp.NDimArray, sp.MatrixBase, list, tuple)):
            return ZArray(out)
        return out

    def to_runtime(self, backend="numpy"):
        """Lambdify the registered functions for `backend` (`"numpy"` or `"ufl"`).
        """
        backend = backend.lower()
        if backend == "numpy":
            return NumpyRuntimeSymbolic(self)
        if backend == "ufl":
            from zoomy_core.transformation.to_ufl import UFLRuntimeSymbolic
            return UFLRuntimeSymbolic(self)
        raise ValueError(f"Unsupported runtime backend '{backend}'.")

    def to_runtime_numpy(self):
        """Lambdify for numpy."""
        return self.to_runtime("numpy")

    def to_runtime_ufl(self):
        """Lambdify through the UFL module dict, for Firedrake backends."""
        return self.to_runtime("ufl")

    def numerical_flux(self):
        """The conservative numerical flux, shape `(n,)`; zero on the base class."""
        zeros = [sp.Integer(0)] * self.n_variables
        return ZArray(zeros)

    def numerical_fluctuations(self):
        """The fluctuations `(D_plus, D_minus)`, shape `(2, n)`; zero on the base
        class.
        """
        zeros = ZArray.zeros(self.n_variables)
        return ZArray([zeros, zeros])

    # ── Interface (star) state ───────────────────────────────────────────
    #
    # ``numerical_flux`` answers "how much crosses the face"; ``star_state``
    # answers "what STANDS on the face" — the intermediate state inside the
    # Riemann fan.  A coupling boundary needs the second: to impose a stage
    # (star depth h*, free surface h* + b*) rather than only a flux.
    #
    # It exists as a registered operator of the framework for the same reason
    # ``numerical_cell_edge_source`` does (see its docstring): a Riemann
    # quantity that lives in one hand-written backend is a scheme decision
    # nobody can supervise.  The hand-rolled version this replaces tried a
    # Rusanov intermediate state, fell back to a two-rarefaction estimate when
    # that returned h* <= SMALL, and fell back AGAIN to the receiving column's
    # own depth when that failed — a silent zero-gradient, measured in 78.6% of
    # the draining windows of one coupling run.  A flux imposed with no stage
    # let the column drain away from its peer, and the Riemann problem then
    # correctly demanded ever more flux across a widening jump.
    #
    # The fallback chain is removable ONLY if the operator cannot return a bad
    # depth in the first place — hence the admissibility projection in
    # ``_clamp_star_depth`` and the guarantee stated on ``star_state``.

    def _star_face_states(self):
        """The `(qL, qR, auxL, auxR)` the star construction runs on: the raw face
        states, or the reconstructed pair in the `Positive*` classes so
        `star_state` describes the problem the flux solves.
        """
        return (self.variables_minus, self.variables_plus,
                self.aux_variables_minus, self.aux_variables_plus)

    def _momentum_indices(self):
        """The momentum block, one discharge row per direction, read from the model's
        `discharge_state_indices`.
        """
        idx = list(self.model.discharge_state_indices)
        if any(k < 0 for k in idx):
            raise ValueError(
                f"{type(self).__name__} needs one discharge row per direction, "
                f"but the model publishes {idx} (a -1 means its mass flux is "
                "not a single state row — e.g. a multilayer system). Choose a "
                "Riemann solver that does not split off a contact wave.")
        return idx

    def _normal_velocity(self, q, aux, n, mom_idx):
        """Depth-averaged velocity `u.n` on the face normal, desingularized by the
        wet/dry `eps`.
        """
        h = self.find_field("h").access(q, aux)
        eps = self._eps_symbol()
        return sum(q[k] * n[d] for d, k in enumerate(mom_idx)) / (h + eps)

    def _physical_flux_n(self, q, aux, p, n):
        """Normal-projected physical flux `(F + P) n` of one state."""
        F = self._model_eval("flux", q, aux, p)
        P = self._model_eval("hydrostatic_pressure", q, aux, p)
        return (F + P) @ n

    # ── the lake at rest: what the hydrostatic reconstruction must restore ──
    def rest_flux_definition(self):
        """The physical flux `F + P` at the lake at rest of the model's own state, shape
        `(n, d)`: the depth and bed rows kept, every other state row zero (no velocity, no
        non-hydrostatic pressure).  It is the hydrostatic pressure wherever the model keeps
        it: `hydrostatic_pressure` (the derived models: their `F` vanishes at rest, so this
        is `P` itself) or the conservative `flux` (`SWE`: `g h^2 / 2` in `F`, `P = 0`).

        References:
            Audusse et al. 2004, https://doi.org/10.1137/S1064827503431090
        """
        keep = {f.index for f in (self.find_field("h", required=False),
                                  self.find_field("b", required=False))
                if f is not None and f.container == "q"}
        rest = {s: sp.S.Zero for i, s in enumerate(self.model.state) if i not in keep}
        F = sp.ImmutableDenseNDimArray(self.model.flux)
        P = sp.ImmutableDenseNDimArray(self.model.hydrostatic_pressure)
        return P + F.applyfunc(lambda e: sp.sympify(e).xreplace(rest))

    def consistency_flux_definition(self):
        """The rest flux the scheme's conservative numerical flux evaluates at the
        hydrostatically reconstructed states, which the consistency source `S~` restores to
        the cell value: `rest_flux_definition` (`Quasilinear*`: zero, no conservative flux).
        """
        return self.rest_flux_definition()

    def consistency_flux_is_zero(self):
        """Whether `consistency_flux_definition` is structurally zero (no `S~`)."""
        return all(sp.sympify(e) == 0 for e in sp.flatten(self.consistency_flux_definition()))

    def hydrostatic_consistency(self, qL, qR, auxL, auxR, qLs, qRs, auxLs, auxRs):
        """Audusse's consistency source per side, `(D_plus, D_minus)` increments: the
        consistency flux of each raw face state minus that of its reconstructed state,
        `Dm = (Pi(q_L) - Pi(q_L*)) n`, `Dp = (Pi(q_R*) - Pi(q_R)) n`.  At a lake at rest the
        face terms then sum to `Pi(q_i) n` over the faces of a cell: exactly zero.

        References:
            Audusse et al. 2004, https://doi.org/10.1137/S1064827503431090
        """
        Pi, p, n = self.consistency_flux_definition(), self.parameters, self.normal
        PL, PR = self._model_eval(Pi, qL, auxL, p), self._model_eval(Pi, qR, auxR, p)
        PLs, PRs = self._model_eval(Pi, qLs, auxLs, p), self._model_eval(Pi, qRs, auxRs, p)
        return (PRs - PR) @ n, (PL - PLs) @ n

    @classmethod
    def treated_nonconservative_columns(cls, model):
        """The columns (state indices) of the model's `nonconservative_matrix` the face terms
        of this class discretize: none on the conservative base (Rusanov, HLL, HLLC); a
        live entry elsewhere is dropped.
        """
        return frozenset()

    def wave_speed_bounds(self, qL, qR, auxL, auxR, p, n):
        """`(s_L, s_R)`, the slowest and fastest signal speeds at the face: here the
        local Lax-Friedrichs pair `(-a, +a)` with `a` the Rusanov dissipation
        speed. `HLL` overrides it with the Davis estimates.

        References:
            Rusanov 1962, https://doi.org/10.1016/0041-5553(62)90062-9
        """
        a = self.face_max_abs_eigenvalue(qL, qR, auxL, auxR, p, n)
        return -a, a

    def _clamp_star_depth(self, q_star, hL, hR):
        """Project the star depth onto `[0, max(h_L, 0) + max(h_R, 0)]`, inactive
        under the classical HLL wave-speed ordering but needed in floating point.
        """
        h = self._declared_depth_handle()
        if h is None or h.container != "q":
            return ZArray(q_star)
        zero = sp.Float(0.0)
        out = ZArray(q_star)
        # ``evaluate=False`` throughout, for the reason given on
        # ``local_max_abs_eigenvalue``: Min/Max over opaque wave-speed-scale
        # expressions is a runtime reduction, and letting sympy attempt the
        # pairwise ordering runs factor_terms over the whole star tree
        # (the REQ-189 setup blow-up).  The unevaluated node lowers to the
        # backend min/max unchanged.
        h_cap = (sp.Max(hL, zero, evaluate=False)
                 + sp.Max(hR, zero, evaluate=False))
        floored = sp.Max(out[h.index], zero, evaluate=False)
        out[h.index] = sp.Min(floored, h_cap, evaluate=False)
        return out

    def star_state(self):
        """The interface (star) state `q*`, shape `(n_variables, 1)`: the HLL
        two-wave average `(s_R q_R - s_L q_L - (F_R - F_L)) / (s_R - s_L)` from the
        solver's own wave speeds, its depth projected onto `[0, h_L + h_R]`. It has
        no supersonic upwind branch, and a model with `h` in `Qaux` gets no depth
        row and no positivity claim.

        References:
            Harten, Lax, van Leer 1983, https://doi.org/10.1137/1025002
        """
        qL, qR, auxL, auxR = self._star_face_states()
        p, n = self.parameters, self.normal
        sL, sR = self.wave_speed_bounds(qL, qR, auxL, auxR, p, n)
        FLn = self._physical_flux_n(qL, auxL, p, n)
        FRn = self._physical_flux_n(qR, auxR, p, n)
        # Floating-point floor on the fan width, not the model's wet/dry eps —
        # same reasoning as ``HLL._compute_flux``: a wet/dry-scale eps here
        # would rescale the (otherwise exact) star at every WET face.  The
        # floor cannot inflate the result: the numerator is bounded by
        # ``(s_R - s_L)·(h_L + h_R)``, so raising the denominator only shrinks
        # h* and the upper bound survives.
        den = sp.Max(sR - sL, sp.Float(1e-14), evaluate=False)
        q_star = (sR * ZArray(qR) - sL * ZArray(qL) - (FRn - FLn)) * (1 / den)
        h = self._declared_depth_handle()
        if h is not None:
            q_star = self._clamp_star_depth(
                q_star, h.access(qL, auxL), h.access(qR, auxR))
        return ZArray(list(sp.flatten(q_star))).reshape(self.n_variables, 1)

    def numerical_cell_edge_source(self):
        """The in-cell nonconservative term per (cell, edge) pair, the part of
        `int A(W) grad W dx` that a reconstruction puts inside the cells (zero on
        piecewise-constant data and on this conservative base class). It is needed
        for every live entry of the nonconservative matrix and is a consistency term,
        not a well-balancing device.

        References:
            Pares 2006, https://doi.org/10.1137/050628052
            Audusse, Bristeau 2005, https://doi.org/10.1016/j.jcp.2004.12.016
        """
        return ZArray.zeros(self.n_variables)

    def _bed_edge_source(self):
        """The in-cell bed-slope product of a (cell, edge) pair: the bed column of the
        nonconservative matrix, integrated along the segment cell -> edge by the
        Gauss-Legendre rule of the path-conservative classes (exact for `g h`), times the
        bed jump `b_e - b_c`: Audusse et al.'s centred source `g (h_c + h_e) / 2 (b_e - b_c)`
        of their second-order scheme.

        References:
            Audusse et al. 2004, https://doi.org/10.1137/S1064827503431090
        """
        bed = _bed_column(self.model)
        if not bed:
            return ZArray.zeros(self.n_variables)
        bed = next(iter(bed))
        qc, qe = self.variables_minus, self.variables_plus
        auxc, auxe = self.aux_variables_minus, self.aux_variables_plus
        xi_np, wi_np = np.polynomial.legendre.leggauss(3)
        column = ZArray.zeros(self.n_variables)
        for xi, wi in zip(0.5 * (xi_np + 1), 0.5 * wi_np):
            B = self._model_eval("nonconservative_matrix", qc + xi * (qe - qc),
                                 auxc + xi * (auxe - auxc), self.parameters)
            column += ZArray([wi * sum(B[i, bed, d] * self.normal[d]
                                       for d in range(len(self.normal)))
                              for i in range(self.n_variables)])
        return column * (qe[bed] - qc[bed])


class Rusanov(Numerics):
    """Rusanov (local Lax-Friedrichs) flux: the central average minus
    `s_max/2 (Q_R - Q_L)`, `s_max` the larger spectral radius of the two states.

    References:
        Rusanov 1962, https://doi.org/10.1016/0041-5553(62)90062-9
        Toro 2009, https://doi.org/10.1007/b79761
    """
    name = param.String(default="RusanovV2")

    def get_viscosity_identity_flux(self):
        """Identity on the dissipated rows (stationary rows zeroed), applied to
        `Q_R - Q_L` in the flux.
        """
        n = self.n_variables
        Id = ZArray.zeros(n, n)
        for i in range(n):
            Id[i, i] = 1
        # Exclude STATIONARY rows (declared role, e.g. the bed; else the
        # structurally zero rows) from the Rusanov dissipation.
        for r in self._stationary_rows:
            Id[r, r] = 0
        return Id

    def get_viscosity_identity_fluctuations(self):
        # Conservative Rusanov variants do not use fluctuation viscosity.
        """Zero: the conservative Rusanov variants dissipate in the flux."""
        n = self.n_variables
        return ZArray.zeros(n, n)

    def numerical_flux(self):
        """The Rusanov flux at the face states."""
        return self._compute_flux(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
            self.parameters,
            self.normal,
        )

    def _compute_flux(self, qL, qR, auxL, auxR, p, n):
        """Rusanov flux of the states `(qL, qR)`."""
        FL = self._model_eval("flux", qL, auxL, p)
        FR = self._model_eval("flux", qR, auxR, p)
        PL = self._model_eval("hydrostatic_pressure", qL, auxL, p)
        PR = self._model_eval("hydrostatic_pressure", qR, auxR, p)
        s_max = self.face_max_abs_eigenvalue(qL, qR, auxL, auxR, p, n)
        Id = self.get_viscosity_identity_flux()
        return 0.5 * ((FL + PL) @ n + (FR + PR) @ n) - 0.5 * s_max * Id @ (qR - qL)


class HLL(Numerics):
    """HLL approximate Riemann solver, model-agnostic: it needs the flux, hydrostatic
    pressure and eigenvalues; without a closed-form spectrum its wave speeds are Gelfand
    bounds of the signed extremes. The flux is one branch-free expression (wave speeds clamped by
    `Min(s_L, 0)` / `Max(s_R, 0)`), so it codegens to every backend.

    References:
        Harten, Lax, van Leer 1983, https://doi.org/10.1137/1025002
        Toro 2009, https://doi.org/10.1007/b79761
    """

    name = param.String(default="HLLV2")

    def numerical_flux(self):
        """The HLL flux at the face states."""
        return self._compute_flux(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
            self.parameters,
            self.normal,
        )

    def wave_speed_bounds(self, qL, qR, auxL, auxR, p, n):
        """Davis estimates, the min / max over the spectrum of both face states; without a
        closed-form spectrum bounds of them from the Gelfand bound (`_gelfand_extremes`), and
        the Lax-Friedrichs pair for the `bound` method.

        References:
            Toro 2009, https://doi.org/10.1007/b79761
            Davis 1988, https://doi.org/10.1137/0909030
        """
        if self.model.eigenvalues is None:
            if self._spectral_method() == "gelfand":
                return self._gelfand_extremes(qL, qR, auxL, auxR, p, n)
            return super().wave_speed_bounds(qL, qR, auxL, auxR, p, n)
        eig = list(sp.flatten(self._model_eval("eigenvalues", qL, auxL, p, n)))
        eig += list(sp.flatten(self._model_eval("eigenvalues", qR, auxR, p, n)))
        # ``evaluate=False`` — Davis wave-speed bounds are a runtime min/max over
        # the spectrum, never a symbolic ordering target (see
        # ``local_max_abs_eigenvalue``); bit-identical when lowered.
        return sp.Min(*eig, evaluate=False), sp.Max(*eig, evaluate=False)

    def _state_jump(self, qL, qR):
        """`qR - qL` with the stationary rows (a bed in the state) zeroed."""
        dq = ZArray(qR) - ZArray(qL)
        for r in self._stationary_rows:
            dq[r] = sp.Integer(0)
        return dq

    def _compute_flux(self, qL, qR, auxL, auxR, p, n):
        """HLL flux of the states `(qL, qR)`, the fan width floored in floating point
        (not at the wet/dry `eps`) so lake-at-rest stays exact.
        """
        FLn = self._physical_flux_n(qL, auxL, p, n)
        FRn = self._physical_flux_n(qR, auxR, p, n)
        sL, sR = self.wave_speed_bounds(qL, qR, auxL, auxR, p, n)
        sLm = sp.Min(sL, sp.Integer(0))
        sRp = sp.Max(sR, sp.Integer(0))
        # Guard the 0/0 at a fully-dry face (sRp = sLm = 0) with a
        # FLOATING-POINT floor, NOT the model's wet/dry threshold ``eps``
        # (~1e-2).  At a WET lake-at-rest face the reconstructed states are
        # equal, so sRp - sLm = 2c and FLn = FRn = P*·n, dq = 0; a wet/dry-
        # scale eps then rescales the (otherwise exact) pressure flux by
        # 1 - eps/(2c) while the Audusse S̃ consistency source stays
        # unscaled — the two no longer cancel and lake-at-rest drifts by
        # O(eps) (the PositiveHLL / PositiveNonconservativeHLL WB defect;
        # max|u| = 1.087·eps measured on the parabolic-bowl WB test).  Same
        # reasoning as the ``hr_eps = 1e-14`` used in
        # ``PositiveRusanov.hydrostatic_reconstruction``.  ``PositiveRusanov``
        # has no denominator, which is why it was already exactly WB.
        eps = sp.Float(1e-14)
        inv = 1 / (sRp - sLm + eps)
        dq = self._state_jump(qL, qR)
        return (sRp * FLn - sLm * FRn + sLm * sRp * dq) * inv


class HLLC(HLL):
    """HLLC solver for the free-surface family: it restores the contact / shear wave
    that HLL smears, needs a depth field `h`, and advects tangential momentum and
    higher moments with the contact. Region selection uses the opaque
    `conditional`, so it codegens to every backend.

    References:
        Toro 2009, https://doi.org/10.1007/b79761
        Toro, Spruce, Speares 1994, https://doi.org/10.1007/bf01414629
    """

    name = param.String(default="HLLCV2")

    @property
    def h_field(self):
        """Depth `FieldHandle`; raises `KeyError` if the model has no `h`
        (use `HLL`).
        """
        return self.find_field("h")

    def _star_state(self, q, aux, h, un, s_side, s_star, n, mom_idx, den=None):
        """HLLC star state on one side of the contact wave; `den` overrides the
        contact denominator (the flux path's `+eps` is sign-blind, `star_state`
        passes a sign-correct floor).
        """
        eps = self._eps_symbol()
        if den is None:
            den = s_side - s_star + eps
        h_star = h * (s_side - un) / den
        out = ZArray(q)
        if self.h_field.container == "q":
            out[self.h_field.index] = h_star
        # Momentum block: tangential part advected, normal part -> h* s*.
        for d, k in enumerate(mom_idx):
            u_k = q[k] / (h + eps)
            u_tan_k = u_k - un * n[d]
            out[k] = h_star * (u_tan_k + s_star * n[d])
        # Remaining depth-scaled rows (higher moments): advected, scale with depth.
        for k in self._scaled_q_indices:
            if k in mom_idx:
                continue
            out[k] = q[k] * h_star / (h + eps)
        return out

    def _star_pair(self, qL, qR, auxL, auxR, p, n, *, sign_safe=False):
        """Build the three-wave data once, `(QLs, QRs, sL, sR, s_star, hL, hR)`,
        shared by the flux and `star_state`; `sign_safe` selects the sign-correct
        denominator floor.
        """
        sL, sR = self.wave_speed_bounds(qL, qR, auxL, auxR, p, n)

        mom_idx = self._momentum_indices()
        eps = self._eps_symbol()

        hL = self.h_field.access(qL, auxL)
        hR = self.h_field.access(qR, auxR)
        unL = self._normal_velocity(qL, auxL, n, mom_idx)
        unR = self._normal_velocity(qR, auxR, n, mom_idx)

        # Contact-wave speed (depth-weighted HLL middle state).
        num = sL * hR * (unR - sR) - sR * hL * (unL - sL)
        den = hR * (unR - sR) - hL * (unL - sL)
        s_star = num / (den + eps)

        denL = denR = None
        if sign_safe:
            fp = sp.Float(1e-14)
            denL = sp.Min(sL - s_star, -fp, evaluate=False)
            denR = sp.Max(sR - s_star, fp, evaluate=False)

        QLs = self._star_state(qL, auxL, hL, unL, sL, s_star, n, mom_idx,
                               den=denL)
        QRs = self._star_state(qR, auxR, hR, unR, sR, s_star, n, mom_idx,
                               den=denR)
        return QLs, QRs, sL, sR, s_star, hL, hR

    def _compute_flux(self, qL, qR, auxL, auxR, p, n):
        """HLLC flux of the states `(qL, qR)`, selecting `F_L | F_L* | F_R* | F_R`
        by the wave speeds.
        """
        FLn = self._physical_flux_n(qL, auxL, p, n)
        FRn = self._physical_flux_n(qR, auxR, p, n)
        QLs, QRs, sL, sR, s_star, _hL, _hR = self._star_pair(
            qL, qR, auxL, auxR, p, n)

        FLs = FLn + sL * (QLs - ZArray(qL))
        FRs = FRn + sR * (QRs - ZArray(qR))

        flux = conditional(
            sL >= 0,
            FLn,
            conditional(
                s_star >= 0,
                FLs,
                conditional(sR > 0, FRs, FRn),
            ),
        )
        return ZArray(flux)

    def star_state(self):
        """The contact-resolved interface state: the side of the contact selected by
        `s*`, built from the star the flux uses and with the same depth projection
        as `Numerics.star_state`. Its depth equals HLL's; the contact shows
        only in the momentum and higher-moment rows.
        """
        qL, qR, auxL, auxR = self._star_face_states()
        p, n = self.parameters, self.normal
        QLs, QRs, _sL, _sR, s_star, hL, hR = self._star_pair(
            qL, qR, auxL, auxR, p, n, sign_safe=True)
        sel = ZArray(conditional(s_star >= 0, QLs, QRs))
        out = self._clamp_star_depth(sel, hL, hR)
        return ZArray(list(sp.flatten(out))).reshape(self.n_variables, 1)


class PositiveRusanov(Rusanov):
    """Rusanov with the Audusse hydrostatic reconstruction `h* = max(0, h + b - b*)`,
    `b* = max(b_L, b_R)`; `h`, `b` and `hinv` are found through
    `Numerics.find_field`, in `Q` or `Qaux`.

    References:
        Audusse et al. 2004, https://doi.org/10.1137/S1064827503431090
    """

    name = param.String(default="PositiveRusanovV2")

    # Field handles are exposed as lazy properties (not set in
    # ``__init__``): ``Numerics.__init__`` eagerly builds the symbolic
    # ``numerical_flux`` — which calls ``hydrostatic_reconstruction`` —
    # *before* a subclass ``__init__`` body runs past
    # ``super().__init__()``.  Resolving on first access hits the
    # ``_field_handles`` cache that ``Numerics.__init__`` already
    # populates.  (Same pattern as :attr:`HLLC.h_field`.)

    @property
    def h_field(self):
        """Depth `FieldHandle`."""
        return self.find_field("h")

    @property
    def b_field(self):
        """Bathymetry `FieldHandle`."""
        return self.find_field("b")

    @property
    def hinv_field(self):
        """Optional `1/h` `FieldHandle` (`None` if absent)."""
        return self.find_field("hinv", required=False)

    def hydrostatic_reconstruction(self, qL, qR, auxL, auxR):
        """Reconstructed face states with `h* = max(0, h + b - b*)`, the depth-scaled
        rows rescaled by `h*/h` and `hinv` rebuilt in the model's KP form;
        returns `(qL, qR, auxL, auxR)`.

        References:
            Audusse et al. 2004, https://doi.org/10.1137/S1064827503431090
        """
        qLs = ZArray(qL)
        qRs = ZArray(qR)
        qauxL = ZArray(auxL)
        qauxR = ZArray(auxR)

        bL = self.b_field.access(qL, auxL)
        bR = self.b_field.access(qR, auxR)
        hL = self.h_field.access(qL, auxL)
        hR = self.h_field.access(qR, auxR)

        b_star = sp.Max(bL, bR)
        hL_star = sp.Max(0.0, hL + bL - b_star)
        hR_star = sp.Max(0.0, hR + bR - b_star)

        # Audusse momentum rescaling regularization.  Must be at
        # **floating-point scale** (just enough to avoid the ``0 / 0``
        # at truly dry faces), *not* the model's wet/dry threshold
        # ``eps`` (typically ``1e-2`` for SWE).  Using the wet/dry
        # threshold here biases the rescaling — for any face value
        # ``hL ∈ (0, eps)`` it scales the momentum down by
        # ``hL / eps`` instead of leaving it unchanged, which breaks
        # the cell-mean positivity decomposition of Xing & Zhang 2013
        # (J. Sci. Comput. 57(1):19-41, doi:10.1007/s10915-013-9695-y).
        # NB: the ``hinv`` slot does NOT use this FP-scale floor — it must
        # be reconstructed with the model's own desingularization (KP form,
        # ``kp_hinv``) so that ``hinv* → 0`` as ``h* → 0`` at an Audusse
        # near-dry face.  The naive ``1/(h*+hr_eps)`` gives ``hinv* ≈ 1e14``
        # there, which detonates the ``hinv²·q²`` quasilinear entries in the
        # NCP path integral (predictor blow-up).  See below.
        hr_eps = sp.Float(1e-14)
        hL_eff = sp.Max(hL, hr_eps)
        hR_eff = sp.Max(hR, hr_eps)

        self.b_field.assign(qLs, qauxL, b_star)
        self.b_field.assign(qRs, qauxR, b_star)
        self.h_field.assign(qLs, qauxL, hL_star)
        self.h_field.assign(qRs, qauxR, hR_star)

        for idx in self._scaled_q_indices:
            qLs[idx] = qL[idx] * hL_star / hL_eff
            qRs[idx] = qR[idx] * hR_star / hR_eff

        if self.hinv_field is not None:
            # Reconstruct the desingularized inverse depth with the model's
            # KP form at the canonical wet/dry threshold — NOT the naive
            # reciprocal (which blows up to ~1/hr_eps at a clamped dry face).
            from zoomy_core.systemmodel.operations import kp_hinv
            eps = self._eps_symbol()
            self.hinv_field.assign(qLs, qauxL, kp_hinv(hL_star, eps))
            self.hinv_field.assign(qRs, qauxR, kp_hinv(hR_star, eps))

        return qLs, qRs, qauxL, qauxR

    def _star_face_states(self):
        """The Audusse-reconstructed pair, the Riemann problem this scheme's own flux
        solves; the star depth is measured from `b* = max(b_L, b_R)`.
        """
        return self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )

    def numerical_flux(self):
        """Rusanov flux on the hydrostatically reconstructed states."""
        qLs, qRs, qauxL, qauxR = self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )
        return self._compute_flux(qLs, qRs, qauxL, qauxR, self.parameters, self.normal)

    def numerical_fluctuations(self):
        """The base fluctuations plus Audusse's consistency source `S~ = Pi_raw - Pi*`
        (`hydrostatic_consistency`), which restores well-balancing.
        """
        qLs, qRs, qauxL, qauxR = self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )
        Dp_jump, Dm_jump = self.hydrostatic_consistency(
            self.variables_minus, self.variables_plus, self.aux_variables_minus,
            self.aux_variables_plus, qLs, qRs, qauxL, qauxR)

        base_fluct = super().numerical_fluctuations()
        Dp_base = ZArray(base_fluct[0, :])
        Dm_base = ZArray(base_fluct[1, :])

        Dp = Dp_base + Dp_jump
        Dm = Dm_base + Dm_jump

        return ZArray([Dp, Dm])

    def numerical_cell_edge_source(self):
        """The next class's in-cell term, plus the bed's (`_bed_edge_source`) where that
        class does not integrate the bed column: a reconstructed bed varies inside the cell,
        and without it a second-order run drops that part of the bed slope (and is not
        well-balanced).
        """
        base = super().numerical_cell_edge_source()
        if _bed_column(self.model) <= super().treated_nonconservative_columns(self.model):
            return base
        return base + self._bed_edge_source()

    @classmethod
    def treated_nonconservative_columns(cls, model):
        """The next class's columns plus the bed's: the hydrostatic reconstruction and `S~`
        discretize the bed-slope product `B[:, b] grad b`.
        """
        return super().treated_nonconservative_columns(model) | _bed_column(model)


class NonconservativeRusanov(Rusanov):
    """Path-conservative Rusanov: a purely central flux, and fluctuations carrying the
    segment-path integral of the nonconservative matrix plus the Rusanov dissipation.

    References:
        Pares 2006, https://doi.org/10.1137/050628052
        Rusanov 1962, https://doi.org/10.1016/0041-5553(62)90062-9
    """
    name = param.String(default="NonconservativeRusanovV2")
    integration_order = param.Integer(default=3)
    # Kernels verified bit-identical between per-face and face-batched
    # evaluation — the numpy solver may evaluate them with a trailing
    # face axis (see solver_numpy.flux_operator).
    supports_batched_faces = True

    def get_path_integral_states(self):
        """The path-integral endpoints `(qL, qR, auxL, auxR)`: the raw face states.
        """
        return (
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )

    def get_viscosity_identity_flux(self):
        # Nonconservative/quasilinear variants place dissipation in
        # numerical_fluctuations to avoid double counting in flux + fluctuation
        # updates.
        """Zero: the dissipation lives in the fluctuations, so it is not counted twice.
        """
        n = self.n_variables
        return ZArray.zeros(n, n)

    def get_viscosity_identity_fluctuations(self):
        """Identity on the dissipated rows (stationary rows zeroed)."""
        n = self.n_variables
        Id = ZArray.zeros(n, n)
        for i in range(n):
            Id[i, i] = 1
        # Preserve STATIONARY rows (declared role, e.g. the bed; else the
        # structurally zero rows).
        for r in self._stationary_rows:
            Id[r, r] = 0
        return Id

    def numerical_fluctuations(self):
        """Base fluctuations plus the path-integral advection and dissipation terms."""
        qLs, qRs, qauxL, qauxR = self.get_path_integral_states()
        nc_fluct = self._compute_fluctuations(
            qLs, qRs, qauxL, qauxR, self.parameters, self.normal
        )
        out = super().numerical_fluctuations()
        return out + nc_fluct

    @classmethod
    def treated_nonconservative_columns(cls, model):
        """Every column: the path integral discretizes the whole nonconservative product."""
        return frozenset(range(len(model.state)))

    def _call_model_matrix(self):
        """The matrix integrated along the path: the model's nonconservative matrix."""
        return lambda Q, Qaux, p: self._model_eval("nonconservative_matrix", Q, Qaux, p)

    def _path_integral_matrix(self, qL, qR, auxL, auxR, p, n):
        """`int_0^1 A(Psi(s)) n ds` along `Psi(s) = qL + s (qR - qL)` by
        Gauss-Legendre of `integration_order`; shared by the fluctuations and the
        in-cell source, so their well-balancing cancellation stays exact.
        """
        xi_np, wi_np = np.polynomial.legendre.leggauss(self.integration_order)
        xi_np = 0.5 * (xi_np + 1)
        wi_np = 0.5 * wi_np

        dQ = qR - qL
        dAux = auxR - auxL
        n_vars = self.n_variables
        dim = len(n)

        A_int = ZArray.zeros(n_vars, n_vars)

        for xi, wi in zip(xi_np, wi_np):
            q_path = qL + xi * dQ
            aux_path = auxL + xi * dAux
            A_tensor = self._call_model_matrix()(q_path, aux_path, p)

            A_n = ZArray.zeros(n_vars, n_vars)
            for i in range(n_vars):
                for j in range(n_vars):
                    val = 0
                    for d in range(dim):
                        val += A_tensor[i, j, d] * n[d]
                    A_n[i, j] = val
            A_int += wi * A_n
        return A_int

    def numerical_cell_edge_source(self):
        """The in-cell nonconservative term: the segment-path integral between cell and
        edge state times their jump, with no dissipation; it vanishes at first order.

        References:
            Audusse, Bristeau 2005, https://doi.org/10.1016/j.jcp.2004.12.016
        """
        qc, qe = self.variables_minus, self.variables_plus
        auxc, auxe = self.aux_variables_minus, self.aux_variables_plus
        A_int = self._path_integral_matrix(
            qc, qe, auxc, auxe, self.parameters, self.normal)
        return A_int @ (qe - qc)

    def _compute_fluctuations(self, qL, qR, auxL, auxR, p, n):
        """`(D_plus, D_minus)` = half of the path-integral advection plus / minus the
        Rusanov dissipation.
        """
        dQ = qR - qL
        A_int = self._path_integral_matrix(qL, qR, auxL, auxR, p, n)

        s_max = self.face_max_abs_eigenvalue(qL, qR, auxL, auxR, p, n)

        term_advection = A_int @ dQ
        Id = self.get_viscosity_identity_fluctuations()
        term_dissipation = s_max * (Id @ dQ)

        Dp_matrix = 0.5 * (term_advection + term_dissipation)
        Dm_matrix = 0.5 * (term_advection - term_dissipation)

        return ZArray([Dp_matrix, Dm_matrix])


class WellBalancedNonconservativeRusanov(NonconservativeRusanov):
    """Path-conservative Rusanov whose dissipation acts on the free-surface jump
    `dh + db`: the depth row couples to the bed column, which makes lake-at-rest
    exact. Applied only when `h` and `b` are both conservative-state fields.
    """

    name = param.String(default="WellBalancedNonconservativeRusanovV2")

    def get_viscosity_identity_fluctuations(self):
        """Identity with the depth row coupled to the bed column."""
        Id = super().get_viscosity_identity_fluctuations()
        h = self.find_field("h", required=False)
        b = self.find_field("b", required=False)
        if (h is not None and b is not None
                and h.container == "q" and b.container == "q"):
            Id[h.index, b.index] = 1
        return Id


class PositiveHLL(HLL):
    """HLL on the Audusse hydrostatically reconstructed states plus the consistency
    source `S~`: positivity-preserving and well-balanced, recommended for dam-break
    and wet/dry runs.

    References:
        Harten, Lax, van Leer 1983, https://doi.org/10.1137/1025002
        Audusse et al. 2004, https://doi.org/10.1137/S1064827503431090
    """

    name = param.String(default="PositiveHLLV2")

    @property
    def h_field(self):
        """Depth `FieldHandle`."""
        return self.find_field("h")

    @property
    def b_field(self):
        """Bathymetry `FieldHandle`."""
        return self.find_field("b")

    @property
    def hinv_field(self):
        """Optional `1/h` `FieldHandle` (`None` if absent)."""
        return self.find_field("hinv", required=False)

    # Reuse the hydrostatic reconstruction from PositiveRusanov by
    # composition (don't multiply-inherit through Rusanov's flux).
    hydrostatic_reconstruction = PositiveRusanov.hydrostatic_reconstruction
    # Same composition for the star-state face pair: the star is built on the
    # reconstructed states, exactly as ``numerical_flux`` below is.
    _star_face_states = PositiveRusanov._star_face_states

    def numerical_flux(self):
        """HLL flux on the hydrostatically reconstructed states."""
        qLs, qRs, qauxL, qauxR = self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )
        return self._compute_flux(
            qLs, qRs, qauxL, qauxR, self.parameters, self.normal,
        )

    def numerical_fluctuations(self):
        """The next fluctuations on the MRO plus Audusse's consistency source
        `S~ = Pi_raw - Pi*` (`hydrostatic_consistency`), independent of the Riemann solver.
        """
        qLs, qRs, qauxL, qauxR = self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )
        Dp_jump, Dm_jump = self.hydrostatic_consistency(
            self.variables_minus, self.variables_plus, self.aux_variables_minus,
            self.aux_variables_plus, qLs, qRs, qauxL, qauxR)

        base_fluct = super().numerical_fluctuations()
        Dp_base = ZArray(base_fluct[0, :])
        Dm_base = ZArray(base_fluct[1, :])

        return ZArray([Dp_base + Dp_jump, Dm_base + Dm_jump])

    def numerical_cell_edge_source(self):
        """As `PositiveRusanov.numerical_cell_edge_source`: the bed's in-cell term where
        the next class does not integrate the bed column."""
        base = super().numerical_cell_edge_source()
        if _bed_column(self.model) <= super().treated_nonconservative_columns(self.model):
            return base
        return base + self._bed_edge_source()

    @classmethod
    def treated_nonconservative_columns(cls, model):
        """The next class's columns plus the bed's (as `PositiveRusanov`)."""
        return super().treated_nonconservative_columns(model) | _bed_column(model)


class PositiveNonconservativeRusanov(PositiveRusanov, NonconservativeRusanov):
    """Hydrostatic reconstruction with the path-conservative fluctuations of
    `NonconservativeRusanov`.
    """
    name = param.String(default="PositiveNonconservativeRusanovV2")
    # HR kernels not yet batch-validated — keep the per-face loop.
    supports_batched_faces = False

    def get_path_integral_states(self):
        """The reconstructed states when the scheme has a rest flux for `S~` to
        restore (`consistency_flux_definition`: the hydrostatic pressure, in `P` or in
        `F`), else the raw states (reconstructed endpoints would then change the mass
        flux).
        """
        if self.consistency_flux_is_zero():
            return (self.variables_minus, self.variables_plus,
                    self.aux_variables_minus, self.aux_variables_plus)
        return self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )


class NonconservativeRoe(NonconservativeRusanov):
    """Path-conservative Roe: the path-integral advection of
    `NonconservativeRusanov` with the scalar dissipation replaced by the Roe
    matrix `|A(Q*)| dQ = R|Lambda|L dQ` at the midpoint state, taken from the
    model's `eigensystem`. It is well-balanced without reconstruction, since the
    bed mode has `lambda = 0`.

    References:
        Pares 2006, https://doi.org/10.1137/050628052
        Roe 1981, https://doi.org/10.1016/0021-9991(81)90128-5
    """

    name = param.String(default="NonconservativeRoe")

    def _abs_quasilinear(self, q_state, aux_state, p, n):
        """`|A_n| = R|Lambda|L` of the normal-projected quasilinear matrix at
        `q_state`, from the model's `eigensystem` operator.
        """
        n_vars = self.n_variables
        es = self._model_eval("eigensystem", q_state, aux_state, p, n)
        R = sp.Matrix(n_vars, n_vars, lambda i, j: es[n_vars + i * n_vars + j])
        L = sp.Matrix(n_vars, n_vars,
                      lambda i, j: es[n_vars + n_vars * n_vars + i * n_vars + j])
        abs_lam = sp.diag(*[sp.Abs(es[k]) for k in range(n_vars)])
        return R @ abs_lam @ L

    def _compute_fluctuations(self, qL, qR, auxL, auxR, p, n):
        """`(D_plus, D_minus)` = half of the path-integral advection plus / minus the
        Roe dissipation.
        """
        xi_np, wi_np = np.polynomial.legendre.leggauss(self.integration_order)
        xi_np = 0.5 * (xi_np + 1)
        wi_np = 0.5 * wi_np

        dQ = qR - qL
        dAux = auxR - auxL
        n_vars = self.n_variables
        dim = len(n)

        # NCP-consistent advection term: path-integral of A along Q_L -> Q_R.
        A_int = ZArray.zeros(n_vars, n_vars)
        for xi, wi in zip(xi_np, wi_np):
            q_path = qL + xi * dQ
            aux_path = auxL + xi * dAux
            A_tensor = self._call_model_matrix()(q_path, aux_path, p)
            A_n = ZArray.zeros(n_vars, n_vars)
            for i in range(n_vars):
                for j in range(n_vars):
                    A_n[i, j] = sum(A_tensor[i, j, d] * n[d] for d in range(dim))
            A_int += wi * A_n
        term_advection = A_int @ dQ

        # Roe matrix dissipation |A(Q*)|*dQ at the midpoint state (closed form).
        absA = self._abs_quasilinear(qL + 0.5 * dQ, auxL + 0.5 * dAux, p, n)
        diss = absA @ sp.Matrix([dQ[i] for i in range(n_vars)])
        term_dissipation = ZArray([diss[i] for i in range(n_vars)])

        Dp_matrix = 0.5 * (term_advection + term_dissipation)
        Dm_matrix = 0.5 * (term_advection - term_dissipation)
        return ZArray([Dp_matrix, Dm_matrix])


class PositiveNonconservativeHLL(PositiveHLL, NonconservativeRusanov):
    """HLL flux, Audusse reconstruction and path-integral fluctuations, dissipating once
    in the flux: HLL upwinds structurally, so the fluctuation viscosity identity is
    zero.

    References:
        Harten, Lax, van Leer 1983, https://doi.org/10.1137/1025002
        Audusse et al. 2004, https://doi.org/10.1137/S1064827503431090
        Pares 2006, https://doi.org/10.1137/050628052
    """

    name = param.String(default="PositiveNonconservativeHLLV2")

    def get_viscosity_identity_fluctuations(self):
        """Zero matrix: the HLL flux already carries the dissipation."""
        n = self.n_variables
        return ZArray.zeros(n, n)

    def get_path_integral_states(self):
        """The hydrostatically reconstructed face states."""
        return self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )


class QuasilinearRusanov(NonconservativeRusanov):
    """Path-conservative Rusanov on the full quasilinear matrix `A = dF/dQ + B`: zero
    flux, everything in the path integral.
    """
    name = param.String(default="QuasilinearRusanovV2")

    def numerical_flux(self):
        """Zero: the conservative part is inside the path integral."""
        zeros = [sp.Integer(0)] * self.n_variables
        return ZArray(zeros)

    def _call_model_matrix(self):
        """The full quasilinear matrix."""
        return lambda Q, Qaux, p: self._model_eval("quasilinear_matrix", Q, Qaux, p)


class PositiveQuasilinearRusanov(PositiveRusanov, QuasilinearRusanov):
    """`QuasilinearRusanov` on the hydrostatically reconstructed states.  The path
    integral of the full quasilinear matrix between the reconstructed states carries the
    pressure and the bed together (`g h* (eta_R - eta_L)`, zero at a lake at rest), so
    there is no conservative flux and no `S~` to restore.
    """
    name = param.String(default="PositiveQuasilinearRusanovV2")

    def numerical_flux(self):
        """Zero: the conservative part is inside the path integral."""
        zeros = [sp.Integer(0)] * self.n_variables
        return ZArray(zeros)

    def consistency_flux_definition(self):
        """Zero: no conservative flux is evaluated at the reconstructed states (adding
        `S~` would count the pressure jump twice: not well-balanced, and inconsistent)."""
        n, d = self.n_variables, len(self.normal)
        return sp.ImmutableDenseNDimArray.zeros(n, d)

    def get_path_integral_states(self):
        """The hydrostatically reconstructed face states."""
        return self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )


class QuasilinearRoe(NonconservativeRoe):
    """Path-conservative Roe on the full quasilinear matrix, so a characteristic
    coupling interface and the flux transmit the same `A`: `D^- + D^+ = A_int dQ`
    with Roe dissipation `R|Lambda|L`. It is unguarded by design, with no
    reconstruction and no entropy fix.

    References:
        Pares 2006, https://doi.org/10.1137/050628052
        Roe 1981, https://doi.org/10.1016/0021-9991(81)90128-5
    """

    name = param.String(default="QuasilinearRoe")

    def numerical_flux(self):
        """Zero: the conservative part rides inside the path integral."""
        return ZArray([sp.Integer(0)] * self.n_variables)

    def _call_model_matrix(self):
        """The full quasilinear matrix."""
        return lambda Q, Qaux, p: self._model_eval("quasilinear_matrix", Q, Qaux, p)

"""Symbolic Riemann solvers: Rusanov, positive, nonconservative variants.

Every index-dependent class (``PositiveRusanov``, ``NonconservativeRusanov``,
``PositiveNonconservativeRusanov``, …) auto-locates the named fields
``h`` and ``b`` (and ``hinv`` when present) in ``model.state`` /
``model.aux_state`` via :class:`FieldHandle` so the same Riemann
code path works whether bathymetry is part of the conservative state
(legacy SWE) or lives in ``Qaux`` (chain-DAE convention).
"""

import numpy as np
import param
import sympy as sp

from zoomy_core.misc.misc import ZArray, Zstruct
from zoomy_core.model.basefunction import SymbolicRegistrar
from zoomy_core.model.kernel_functions import (
    conditional,
)
from zoomy_core.systemmodel.system_model import (
    SystemModel, _perron_radius_bound)
from zoomy_core.transformation.to_numpy import (
    NumpyRuntimeModel, NumpyRuntimeSymbolic,
)


class FieldHandle:
    """A named field resolved to either ``Q`` (state) or ``Qaux``.

    Looks up the location at construction time via :meth:`Numerics.find_field`,
    then exposes the symbolic Symbol on the **minus / plus / state** side of
    a face so callers can write reconstruction code without knowing where
    the field lives.

    Attributes
    ----------
    name : str
        Field name (e.g. ``"h"``, ``"b"``, ``"hinv"``).
    container : {"q", "qaux"}
        Which array carries this field.
    index : int
        Index into that array.
    minus, plus, state : sympy.Symbol
        Direct references — ``minus`` and ``plus`` are the per-face
        symbolic state on the L / R side of a Rusanov face; ``state``
        is the cell-centre reference.  Using these in symbolic code
        produces the same lambdified output as ``Q[index]`` /
        ``Qaux[index]`` — a true placeholder.
    """

    __slots__ = ("name", "container", "index", "minus", "plus", "state")

    def __init__(self, name, container, index, minus, plus, state):
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
        """Return ``q_array[index]`` or ``qaux_array[index]``
        depending on container.  Works for symbolic ``ZArray`` and
        numeric numpy arrays alike."""
        return (q_array[self.index] if self.container == "q"
                else qaux_array[self.index])

    def assign(self, q_array, qaux_array, value):
        """In-place write into the appropriate container."""
        if self.container == "q":
            q_array[self.index] = value
        else:
            qaux_array[self.index] = value

    def __repr__(self):
        return (f"FieldHandle({self.name!r}: "
                f"{self.container}[{self.index}])")


class Numerics(param.Parameterized, SymbolicRegistrar):
    """Symbolic numerics over a :class:`SystemModel`.

    The numerics consumes a :class:`SystemModel` — the frozen
    operator-form snapshot of a derivation.  A :class:`Model` passed to
    the constructor is normalised once via
    :meth:`SystemModel.from_model`; everything internal reads the
    SystemModel's stored operators (``flux``, ``hydrostatic_pressure``,
    ``eigenvalues``, ``nonconservative_matrix``, ``quasilinear_matrix``)
    and its ``state`` / ``aux_state`` / ``parameters`` / ``normal``.

    Subclasses that depend on the location of named fields (``h``,
    ``b``, ``hinv``, …) call :meth:`find_field` to obtain a
    :class:`FieldHandle` — the search walks state then aux
    automatically, so the same numerics works whether bathymetry is in
    the state or in ``aux_state``."""

    name = param.String(default="NumericsV2")
    model = param.Parameter(
        default=None,
        doc="The SystemModel the numerics operates on (a Model is "
            "normalised via SystemModel.from_model in __init__).",
    )

    scaled_q_indices = param.List(default=None, allow_None=True)

    def __init__(self, model, **params):
        """Initialize the instance.

        ``model`` may be a :class:`SystemModel` (used directly) or a
        :class:`Model` (normalised once via
        :meth:`SystemModel.from_model`).
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
        """Return a :class:`FieldHandle` for the named field.

        Searches ``self.model.state`` (Q) first, then
        ``self.model.aux_state`` (Qaux).  Caches the result in
        ``self._field_handles[name]`` so repeat lookups are free.

        Parameters
        ----------
        name : str
        required : bool
            If True (default), raise ``KeyError`` when the field is
            in neither container.  Otherwise return ``None``.
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
        return self.find_field(name, required=False) is not None

    def _declared_depth_handle(self):
        """The depth handle ONLY when the model declares a depth role — the
        generic (all-model) kernels never fall back to a field name."""
        roles = getattr(self.model, "roles", None)
        if roles is None or roles.depth is None:
            return None
        return self._field_handles.get("h")

    def _create_v(self, name, size):
        """Internal helper `_create_v`."""
        v = ZArray([sp.Symbol(f"{name}_{i}", real=True) for i in range(size)])
        v._symbolic_name = name
        return v

    def _resolve_scaled_q_indices(self, scaled_q_indices):
        """Internal helper `_resolve_scaled_q_indices`.  An explicit
        ``scaled_q_indices=`` wins; otherwise READ the model's own row map
        (:attr:`SystemModel.depth_scaled_state_indices`) — the depth-weighted
        rows are a property of the system, not of the flux, and there is
        exactly one answer-holder for them."""
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
        """The canonical wet/dry threshold ``wet_dry_eps`` (REQ-48).

        Returns the symbolic parameter when the model declares it (e.g.
        MalpassetSWE → the ``wet_dry_eps`` symbol, so the threshold is
        run-time tunable through ``parameter_values``); models that do not
        model wetting/drying carry no such parameter and fall back to the
        desingularisation-only ``1e-12`` literal floor.
        """
        if self.model.parameters.contains("wet_dry_eps"):
            return self.model.parameters.wet_dry_eps
        return sp.Float(1e-12)

    def _initialize_functions(self):
        """Internal helper `_initialize_functions`."""
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
        """The face's contribution to each adjacent cell, shape ``(2, n)``:

            row 0 = ``F + D_minus``  (into the owner,     with +|f|/|V|)
            row 1 = ``-F + D_plus``  (into the neighbour, with +|f|/|V|)

        so a finite-volume update is one face loop scattering two rows,
        ``dQ_owner -= row0 |f|/|V_o|``, ``dQ_neighbour -= row1 |f|/|V_n|``.  A
        conservative scheme is the special case ``D_plus = D_minus = 0``.  Built
        from the same expression objects as :meth:`numerical_flux` /
        :meth:`numerical_fluctuations` (row 0 of ``fluctuations`` is
        ``D_plus``, row 1 ``D_minus``), so it is an exact recombination; one
        printed silo shares the Riemann problem (wave speeds, eigenvalues)."""
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
        """The fastest signal speed of this scheme's own Riemann solver:
        ``max(|s_L|, |s_R|)`` of its :meth:`wave_speed_bounds` (Davis for HLL,
        ``+-a`` for the Rusanov family) at the states its dissipation is built
        on -- the path-integral / hydrostatically reconstructed states where
        the scheme has them, else the face states.  The same expressions the
        flux builds, so a printed silo computes them once."""
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
        # (min(s_L, 0), max(s_R, 0)), so the printed silo shares it
        return sp.Max(-sp.Min(sL, sp.Integer(0)), sp.Max(sR, sp.Integer(0)),
                      evaluate=False)

    def face_sides_speed(self, flux, fluctuations):
        """:meth:`face_sides` plus a third row carrying
        :meth:`face_wave_speed` (in every column), shape ``(3, n)``: a solver
        can take its CFL step from the Riemann problem it already solves."""
        sides = self.face_sides(flux, fluctuations)
        s = self.face_wave_speed()
        n = self.n_variables
        return ZArray([[sides[0, i] for i in range(n)],
                       [sides[1, i] for i in range(n)],
                       [s] * n])

    def pack_face(self, flux, fluctuations):
        """Pack one face's COMPLETE Riemann answer into the flat layout

            ``[ flux(n) | D_plus(n) | D_minus(n) | lambda_max(1) ]``

        of shape ``(3·n_variables + 1, 1)``, on the same two-state
        ``(q_minus, q_plus, aux_minus, aux_plus, p, normal)`` signature the
        legacy face kernels carry.

        ``flux`` / ``fluctuations`` are the ALREADY-BUILT definitions of
        :meth:`numerical_flux` / :meth:`numerical_fluctuations` (shape
        ``(n,)`` and ``(2, n)``, row 0 = ``D_plus``, row 1 = ``D_minus`` —
        the convention every ``numerical_fluctuations`` override follows).
        Packing the same expression objects is what makes the fused kernel a
        strict repack: unpacking it reproduces the three legacy kernels
        exactly, not just to tolerance.

        The payoff is the CSE silo.  The printers run one ``sp.cse`` per
        registered function, so ``1/h``, the hydrostatic reconstruction and
        the eigenvalue candidates are each recomputed once PER legacy
        kernel.  Fused, they are shared subexpressions in a single silo and
        collapse to one temp apiece.

        ``lambda_max`` is the face wave speed on the RAW (un-reconstructed)
        face states — the Rusanov/CFL speed a driver needs for ``dt``, and
        by construction the same block the schemes already build for their
        own dissipation (see :meth:`face_max_abs_eigenvalue`), so it costs
        nothing extra in the fused silo for the non-reconstructing schemes.

        NB the flux row alone is NOT a usable scheme: for the
        nonconservative variants ``get_viscosity_identity_flux`` is zero and
        ALL dissipation lives in ``D_plus``/``D_minus``.  A consumer must
        use the fluctuation rows too.
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
        """Face wave speed ``max(ρ(A_n)|_L, ρ(A_n)|_R)`` — the Rusanov /
        local-Lax-Friedrichs dissipation speed at a face.

        ``evaluate=False`` for the same reason as
        :meth:`local_max_abs_eigenvalue`: the outer ``Max`` is a runtime
        reduction over two opaque wave speeds, never a symbolic ordering
        target.
        """
        return sp.Max(
            self.local_max_abs_eigenvalue(qL, auxL, p, n),
            self.local_max_abs_eigenvalue(qR, auxR, p, n),
            evaluate=False,
        )

    def local_max_eigenvalue_definition(self):
        """Spectral radius of the normal-projected system at the numerics'
        own ``Q/Qaux/p/n`` symbols — the kernel body the code printers emit.

        Computed DIRECTLY from the model spectrum:
        ``Max(|λ_i|)`` over the SystemModel's normal-projected
        ``eigenvalues``.  When the model carries no closed-form spectrum
        (``model.eigenvalues is None`` — SME by default, hand-built models
        such as VAM) it falls back to
        :attr:`~zoomy_core.systemmodel.system_model.SystemModel.
        spectral_radius_bound`, a strict closed-form upper bound on
        ``ρ(A_n)`` built from the normal-projected quasilinear matrix — see
        :meth:`local_max_abs_eigenvalue`.
        """
        return self.local_max_abs_eigenvalue(
            self.variables, self.aux_variables,
            self.parameters, self.normal,
        )

    def local_max_abs_eigenvalue(self, Q=None, Qaux=None, p=None, n=None):
        """Spectral radius at the given state ``(Q, Qaux, p, n)`` — the real
        expression substituted into Rusanov dissipation / CFL.

        Two paths, and the model's own spectrum always wins:

        * closed-form ``eigenvalues`` (SWE, SME with ``symbolic_spectrum``)
          → ``Max(|λ_i|)`` over EXACTLY that spectrum, unchanged;
        * no closed form (``eigenvalues is None`` — SME by default, VAM, every
          hand-built model) → the model's :attr:`~zoomy_core.systemmodel.
          system_model.SystemModel.spectral_radius_bound`, a closed-form
          UPPER bound on ``ρ(A_n)`` with no eigen-decomposition in it.

        The second path used to be the opaque ``numerical_eigenvalues``
        kernel, i.e. a full numerical eigen-decomposition of the 12×12 (SME
        level 4, 2-D) normal-projected quasilinear matrix PER FACE PER STEP,
        on both sides of every face — which is what put the emitted C / foam
        SME cases at ~40× their old cost the moment a model stopped carrying a
        symbolic spectrum.  This kernel wants a RADIUS, and a radius is far
        cheaper than a spectrum.  MEASURED on the EMITTED foam kernel of the
        confluence ``SME(level=4, dimension=3)``, compiled ``-O2``, LAPACK
        ``dgeev`` standing in for ``Eigen::EigenSolver``: 23.13 µs/call before,
        0.241 µs/call after — **96×**, and 49× on ``SME(level=1)`` at n = 4.
        The bound is STRICT at any iteration count, so the CFL cannot go
        unsafe (see that property for the derivation and the tightness).

        NOT a replacement for a spectrum: the bound is one number, so Roe —
        which needs ``R |Λ| L`` — still goes through ``eigensystem``.  Only
        the wave-speed consumers (Rusanov / LF dissipation, CFL dt) read this.
        """
        if Q is None:
            return self.local_max_eigenvalue_definition()
        if self.model.eigenvalues is None:
            # No closed form.  DEFAULT is the exact numerical spectrum, because
            # this quantity sets the Rusanov / LF / HLL DISSIPATION as well as
            # the CFL, and an upper bound is extra numerical diffusion there
            # (see SystemModel.use_spectral_radius_bound for the measured cost
            # on the Ritter dry dam break).  The cheap bound is opt-in.  The
            # symbolic solvers set "power" (zoomy_core.symsolver.WAVE_SPEED).
            method = getattr(self.model, "spectral_radius_method",
                             SystemModel.spectral_radius_method)
            if method == "bound" or getattr(self.model, "use_spectral_radius_bound",
                                             SystemModel.use_spectral_radius_bound):
                return self._spectral_radius_bound(Q, Qaux, p, n)
            if method == "power":
                from zoomy_core.model.kernel_functions import spectral_radius
                A = self._model_eval("quasilinear_matrix", Q, Qaux, p)
                nrm = list(n)
                flat = [sum(A[i, j, d] * nrm[d] for d in range(len(nrm)))
                        for i in range(self.n_variables)
                        for j in range(self.n_variables)]
                m = self.model
                return spectral_radius(
                    sp.Integer(getattr(m, "spectral_power_steps",
                                       SystemModel.spectral_power_steps)),
                    sp.Float(getattr(m, "spectral_power_safety",
                                     SystemModel.spectral_power_safety)), *flat)
            if method != "eig":
                raise ValueError(f"spectral_radius_method {method!r}: "
                                 "'eig', 'power' or 'bound'")
            src = list(sp.flatten(
                self._model_eval("numerical_eigenvalues", Q, Qaux, p, n)))
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

    def _spectral_radius_bound(self, Q, Qaux, p, n):
        """The model's :attr:`~zoomy_core.systemmodel.system_model.SystemModel.
        spectral_radius_bound` at the numerics' own state symbols.

        Built HERE from the ALREADY-SUBSTITUTED quasilinear matrix rather than
        read off the SystemModel property and ``subs``-ed: ``subs`` rebuilds
        the tree WITH evaluation, which would collapse the two
        ``evaluate=False`` reductions the bound is made of and hand sympy the
        pairwise ``Max``/``Min`` ordering of a dozen wave-speed trees.  Both
        routes call the same :func:`_perron_radius_bound`, so the emitted
        expression is structurally identical either way.
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
        """Evaluate a SystemModel operator at a given state.

        ``operator_name`` is the SystemModel attribute name — ``flux``,
        ``hydrostatic_pressure``, ``eigenvalues``,
        ``nonconservative_matrix`` or ``quasilinear_matrix``.  Returns
        the operator with ``state`` / ``aux_state`` / ``parameters`` /
        ``normal`` symbols substituted by the supplied values.
        """
        definition = getattr(self.model, operator_name)
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
        """To runtime."""
        backend = backend.lower()
        if backend == "numpy":
            return NumpyRuntimeSymbolic(self)
        if backend == "ufl":
            from zoomy_core.transformation.to_ufl import UFLRuntimeSymbolic
            return UFLRuntimeSymbolic(self)
        raise ValueError(f"Unsupported runtime backend '{backend}'.")

    def to_runtime_numpy(self):
        """To runtime numpy."""
        return self.to_runtime("numpy")

    def to_runtime_ufl(self):
        """Lambdify all registered symbolic functions through the UFL
        module dict — for use by Firedrake-based backends."""
        return self.to_runtime("ufl")

    def numerical_flux(self):
        """Numerical flux."""
        zeros = [sp.Integer(0)] * self.n_variables
        return ZArray(zeros)

    def numerical_fluctuations(self):
        """Numerical fluctuations."""
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
        """The ``(qL, qR, auxL, auxR)`` the star construction runs on.

        The RAW face states here; the ``Positive*`` subclasses override with
        their Audusse-Bristeau-Klein reconstructed pair, so ``star_state``
        always describes the SAME Riemann problem that scheme's own
        ``numerical_flux`` solves.  For those subclasses the star depth is
        therefore measured from ``b* = max(b_L, b_R)``, not from the local
        bed: a consumer wanting a free-surface stage forms
        ``eta* = h* + max(b_L, b_R)``.  (At a coupling interface the two
        sides sample the same bed, so ``b_L = b_R`` and the distinction is
        empty; it only bites across a bed step.)
        """
        return (self.variables_minus, self.variables_plus,
                self.aux_variables_minus, self.aux_variables_plus)

    def _momentum_indices(self):
        """The momentum block: the direction-``d`` discharge row, READ from the
        model's own row map (:attr:`SystemModel.discharge_state_indices`).

        This used to be ``self._scaled_q_indices[:n_dim]`` — "the first n_dim
        depth-scaled rows", which is the state-ORDER guess that is true only
        for the four-row SWE state.  On ``SME(level>=1, dimension=3)`` it
        returned ``[q_x_0, q_x_1]`` where the momentum block is
        ``[q_x_0, q_y_0]``, so the HLLC contact wave was built on a
        y-velocity that was really the first x-moment."""
        idx = list(self.model.discharge_state_indices)
        if any(k < 0 for k in idx):
            raise ValueError(
                f"{type(self).__name__} needs one discharge row per direction, "
                f"but the model publishes {idx} (a -1 means its mass flux is "
                "not a single state row — e.g. a multilayer system). Choose a "
                "Riemann solver that does not split off a contact wave.")
        return idx

    def _normal_velocity(self, q, aux, n, mom_idx):
        """Depth-averaged velocity projected on the face normal, ``u·n``,
        desingularized by the model's wet/dry ``eps``."""
        h = self.find_field("h").access(q, aux)
        eps = self._eps_symbol()
        return sum(q[k] * n[d] for d, k in enumerate(mom_idx)) / (h + eps)

    def _physical_flux_n(self, q, aux, p, n):
        """Normal-projected physical flux ``(F + P) @ n`` for a single state."""
        F = self._model_eval("flux", q, aux, p)
        P = self._model_eval("hydrostatic_pressure", q, aux, p)
        return (F + P) @ n

    def wave_speed_bounds(self, qL, qR, auxL, auxR, p, n):
        """Return ``(s_L, s_R)`` — slowest / fastest signal speeds at the face.

        On this base class the LOCAL LAX-FRIEDRICHS pair ``(-a, +a)`` with
        ``a = max(rho(A_n)|_L, rho(A_n)|_R)`` — precisely the dissipation
        speed :class:`Rusanov` already builds, so the Rusanov family's star
        state is built from ITS OWN wave speeds rather than a borrowed pair.
        :class:`HLL` overrides with the sharper Davis estimates.
        """
        a = self.face_max_abs_eigenvalue(qL, qR, auxL, auxR, p, n)
        return -a, a

    def _clamp_star_depth(self, q_star, hL, hR):
        """Project the star depth onto the admissible interval
        ``[0, max(h_L, 0) + max(h_R, 0)]``.

        THE invariant of :meth:`star_state`.  Under the classical HLL
        wave-speed ordering the clamp is provably INACTIVE (see the
        positivity argument on :meth:`star_state`), so this is a projection
        onto a set the construction is already in — not a correction that
        changes the answer.  It is written out anyway because the guarantee
        has to hold in EMITTED FLOATING-POINT CODE, where the star numerator
        is a difference of large terms and can round to a small negative, and
        because a model whose spectrum does not bracket the advective speed
        would otherwise break the ordering silently.

        No-op for a model with no ``h`` field, or one carrying ``h`` in
        ``Qaux`` (the returned vector is the Q rows only, so there is no
        depth row to project — such models get a star state with no
        positivity claim attached, which is stated on ``star_state``).
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
        """THE INTERFACE (STAR) STATE of this solver's Riemann fan.

        Returns the intermediate state vector ``q*`` of shape
        ``(n_variables, 1)`` — the Q rows only — on the same
        ``(q_minus, q_plus, aux_minus, aux_plus, p, normal)`` signature the
        other face kernels carry.  Registered in ``self.functions``, so it is
        lambdified for numpy / jax / UFL and code-printed by every backend
        alongside ``numerical_flux``.

        CONSTRUCTION (this base class, and every subclass that does not
        override).  The HLL two-wave average

            q* = ( s_R q_R - s_L q_L - (F_R - F_L) ) / (s_R - s_L)

        with ``F`` the normal-projected physical flux ``(F + P)·n`` and
        ``(s_L, s_R)`` from this solver's own :meth:`wave_speed_bounds` —
        the Davis estimates for the HLL family, ``(-a, +a)`` with
        ``a = max rho(A_n)`` for the Rusanov family.  With those speeds the
        expression above IS the Rusanov intermediate state
        ``½(q_L + q_R) - (F_R - F_L)/(2a)``, written in the HLL form so the
        positivity argument below applies verbatim.  That is the deliberate
        answer to "what do the Rusanov variants return": the HLL-form star
        built from their own wave speeds, PLUS the admissibility projection —
        never the raw average, which is exactly the estimate that could go
        non-positive and hand the caller a fallback problem.

        This is the PURE star state: no supersonic upwind branch.  ``q*_HLL``
        is the fan average and stays a well-defined, branch-free, continuous
        consistent state even when the fan does not straddle the interface,
        which is what a boundary condition wants; switching to ``q_L`` or
        ``q_R`` there would reintroduce exactly the "the interface is just my
        own column" behaviour this operator exists to remove.

        POSITIVITY — THE GUARANTEE.  ``h* >= 0`` ALWAYS, and
        ``h* <= max(h_L,0) + max(h_R,0)`` always; both hold unconditionally
        because :meth:`_clamp_star_depth` projects onto that interval.  The
        projection is inactive — i.e. the value returned is the exact
        textbook star depth — under the classical HLL ordering

            s_L <= u_L·n   and   s_R >= u_R·n   and   s_R > s_L,

        the same hypothesis under which Harten, Lax and van Leer prove the HLL
        intermediate state admissible.  With ``F_h·n = h u·n`` (true of every
        depth-averaged model in the framework: the continuity flux IS the
        momentum) the depth row reads

            h* = [ h_R (s_R - u_R·n) + h_L (u_L·n - s_L) ] / (s_R - s_L),

        a sum of two NON-NEGATIVE terms over a positive denominator, so
        ``h* >= 0``; and since ``s_R - u_R·n <= s_R - s_L`` and
        ``u_L·n - s_L <= s_R - s_L`` under the same ordering, ``h* <= h_L +
        h_R``.  The ordering holds for the Davis bounds of any model whose
        normal spectrum contains ``u·n ± c`` with ``c >= 0`` — every
        free-surface model here — and for the Lax-Friedrichs pair whenever
        ``a >= |u·n|``, which the spectral radius satisfies for the same
        reason.  Strictly: ``h* > 0`` whenever EITHER face is wet, because
        then ``c > 0`` on that side makes its term strictly positive;
        ``h* = 0`` says the interface is genuinely dry, which is a stage, not
        a failure, and needs no fallback.

        DOMAIN — the other half of the guarantee, and the honest half.  The
        clamp bounds an ARITHMETIC result; it cannot manufacture one.  Feed a
        face state at which the model itself is undefined — a NEGATIVE depth
        (``sqrt(g h)``), or a dry face to a model with no wet/dry
        desingularisation — and the spectrum is NaN, so the star is NaN, and
        ``Max``/``Min`` propagate it rather than clip it (NaN is not ``> 0``).
        That is inherited, not introduced: ``numerical_flux`` and
        ``local_max_abs_eigenvalue`` are NaN at exactly the same states.  It
        is deliberately NOT papered over here.  A caller handed a finite stage
        next to a NaN flux is worse off than one handed both NaN, and a
        negative face depth is a positivity failure upstream that must
        surface.
        The framework's answer for that regime is the reconstruction, not a
        second clamp: the ``Positive*`` family evaluates the star on
        ``h* = max(0, h + b - b*) >= 0``, so the model is never asked about an
        inadmissible state.  MEASURED, 12 solvers x 14400 face pairs spanning
        ``h in {-1, -1e-9, 0, 1e-14, 1e-3, 0.5, 2, 50}``,
        ``u in {-30, -2, 0, 2, 30}``, bed steps to 3 m: zero violations of
        ``0 <= h* <= h_L + h_R`` for every solver on the states where the
        model has a finite spectrum, and zero violations for the ``Positive*``
        family over ALL states including the negative depths.  Prefer
        :class:`PositiveHLL` at a wet/dry or coupling boundary — which is
        already this framework's standing recommendation.

        CAVEATS, stated rather than hidden.  (1) The construction uses the
        CONSERVATIVE part only.  For the path-conservative / "Malaga"
        formulations, where ``hydrostatic_pressure`` is empty and ``g h dh``
        lives in the nonconservative matrix, the momentum rows of ``q*`` do
        not see that product; the DEPTH row, and hence the stage, is
        unaffected (continuity carries no nonconservative term).  (2) When
        ``b`` is part of the conservative state its row is the wave-speed
        weighted average ``(s_R b_R - s_L b_L)/(s_R - s_L)``, a convex
        combination whenever ``s_L <= 0 <= s_R``.  (3) Models carrying ``h``
        in ``Qaux`` get no depth row in the returned vector and therefore no
        positivity claim.
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
        """THE IN-CELL NONCONSERVATIVE TERM, per (cell, edge) pair.

        A path-conservative scheme discretises ``W_t + A(W)·W_x = 0`` as

            |C_i| dW_i/dt = -[ sum_j D^-(W_ij, W_ji)
                               + int_{C_i} A(W_i(x))·grad W_i(x) dx ]

        With PIECEWISE-CONSTANT data all variation of W sits in the interface
        jumps, so the fluctuations carry the whole nonconservative product and
        the integral is zero.  Once a RECONSTRUCTION exists the variation
        splits in two — jumps BETWEEN cells and smooth slopes INSIDE them —
        and the fluctuations see only the first.  This term is the second.

        Omitting it is not a loss of accuracy: the missing amount is
        ``int A·grad W ~ dx·A·grad W``, which divided by ``dx`` in the update
        is O(1).  It deletes a term of the PDE, so the error is a CONSISTENCY
        error and therefore dt-INDEPENDENT.  Measured on smooth, flat-bed,
        wet-everywhere SME(level=1) — where ``B[q_1][q_1] = -q_0/h`` is the
        only live nonconservative product — the row it feeds stalls at rate
        1.26 while h and q_0 reach 1.73/1.78; at order 1 all three sit
        together near 1.0.

        NOT a well-balancing device.  It is required for EVERY live entry of
        the nonconservative matrix; the bed column ``B[q_0][b] = g·h`` is
        merely the entry whose omission is most visible (it breaks
        lake-at-rest).  For SWE the general form below reduces exactly to
        Audusse & Bristeau (2005) eq. (6.13),
        ``-(g/2)(h_ij + h_i)(Z_ij - Z_i)·n_ij`` — their centred term is the
        closed-form path integral for a ``B`` that happens to be LINEAR in the
        state.  Ours are not (``hinv``, ``q^2/h``, moment couplings), which is
        why the quadrature has to be the real one rather than a trapezoid.

        Returns the contribution to the residual of the cell whose OUTWARD
        normal is ``normal``.  Called once per (cell, edge) pair — twice per
        interior face, with the two cells swapped and the normal negated.

        WHY THIS IS EMITTED RATHER THAN WRITTEN IN A BACKEND (mandate 6).
        This term already existed — as ``B(Q_c)·s_c``, hand-written inside
        ``zoomy_jax/fvm/solver_jax.py`` and nowhere else.  Because it was never
        exposed through the NumericalSystemModel it was invisible to
        ``describe()``, to every golden, and to every other backend: foam,
        amrex and firedrake silently had no interior nonconservative term at
        all, and the one implementation that existed could not be reviewed,
        diffed or held to a quadrature.  Diagnosing its presence took a full
        session of measurement that reading a printed operator would have
        answered in a minute.

        A scheme decision that lives in one backend is a scheme decision
        nobody can supervise.  Routing every LOCAL operator through the NSM is
        what makes them fetchable, comparable and consistent across backends —
        and only then is a convergence result attributable to the scheme
        rather than to whichever backend the case happened to run on.

        Zero on this base class: a purely conservative solver carries no
        nonconservative product.  Overridden where ``B`` is live."""
        return ZArray.zeros(self.n_variables)


class Rusanov(Numerics):
    """Rusanov. (class)."""
    name = param.String(default="RusanovV2")

    def get_viscosity_identity_flux(self):
        """Get viscosity identity flux."""
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
        """Get viscosity identity fluctuations."""
        n = self.n_variables
        return ZArray.zeros(n, n)

    def numerical_flux(self):
        """Numerical flux."""
        return self._compute_flux(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
            self.parameters,
            self.normal,
        )

    def _compute_flux(self, qL, qR, auxL, auxR, p, n):
        """Internal helper `_compute_flux`."""
        FL = self._model_eval("flux", qL, auxL, p)
        FR = self._model_eval("flux", qR, auxR, p)
        PL = self._model_eval("hydrostatic_pressure", qL, auxL, p)
        PR = self._model_eval("hydrostatic_pressure", qR, auxR, p)
        s_max = self.face_max_abs_eigenvalue(qL, qR, auxL, auxR, p, n)
        Id = self.get_viscosity_identity_flux()
        return 0.5 * ((FL + PL) @ n + (FR + PR) @ n) - 0.5 * s_max * Id @ (qR - qL)


class HLL(Numerics):
    """HLL (Harten-Lax-van-Leer) approximate Riemann solver.

    Model-agnostic: needs only the SystemModel's ``flux``,
    ``hydrostatic_pressure`` and ``eigenvalues`` operators.  When the
    SystemModel carries a symbolic spectrum the wave-speed bounds are
    the Davis estimates — min / max over the eigenvalues of *both* face
    states.  When ``eigenvalues`` is ``None`` (the model skipped the
    spectral derivation) it falls back to ``± local_max_abs_eigenvalue``,
    i.e. HLL collapses to local Lax-Friedrichs (a valid, more diffusive
    HLL).

    The numerical flux is a single closed-form (branch-free) SymPy
    expression — clamping the wave speeds with ``Min(s_L, 0)`` /
    ``Max(s_R, 0)`` recovers the upwind branches without ``Piecewise``,
    so it codegens cleanly to every backend.
    """

    name = param.String(default="HLLV2")

    def numerical_flux(self):
        """Numerical flux."""
        return self._compute_flux(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
            self.parameters,
            self.normal,
        )

    def wave_speed_bounds(self, qL, qR, auxL, auxR, p, n):
        """Davis estimates — min / max over the spectrum of BOTH face states.

        Falls back to the base class's Lax-Friedrichs pair ``(-a, +a)`` when
        the model carries no closed-form spectrum, i.e. HLL collapses to
        local Lax-Friedrichs (a valid, more diffusive HLL).
        """
        if self.model.eigenvalues is None:
            return super().wave_speed_bounds(qL, qR, auxL, auxR, p, n)
        eig = list(sp.flatten(self._model_eval("eigenvalues", qL, auxL, p, n)))
        eig += list(sp.flatten(self._model_eval("eigenvalues", qR, auxR, p, n)))
        # ``evaluate=False`` — Davis wave-speed bounds are a runtime min/max over
        # the spectrum, never a symbolic ordering target (see
        # ``local_max_abs_eigenvalue``); bit-identical when lowered.
        return sp.Min(*eig, evaluate=False), sp.Max(*eig, evaluate=False)

    def _state_jump(self, qL, qR):
        """``qR - qL`` with stationary fields (bed in conservative state) zeroed."""
        dq = ZArray(qR) - ZArray(qL)
        for r in self._stationary_rows:
            dq[r] = sp.Integer(0)
        return dq

    def _compute_flux(self, qL, qR, auxL, auxR, p, n):
        """Internal helper `_compute_flux`."""
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
    """HLLC approximate Riemann solver for the free-surface family.

    Restores the contact / shear wave that HLL smears.  Requires a depth
    field ``h`` (resolved via :meth:`Numerics.find_field`); the momentum
    block is the first ``model.dimension`` depth-scaled Q rows.  Any
    further depth-scaled rows (higher moments) are advected by the
    contact wave; non-scaled rows (e.g. bed in conservative state) pass
    through unchanged.  Models without an ``h`` field should use
    :class:`HLL` instead.

    Region selection (``F_L | F_L* | F_R* | F_R``) uses the opaque
    ``conditional`` primitive, so it codegens to ``np.where`` / ternary
    expressions on every backend.
    """

    name = param.String(default="HLLCV2")

    @property
    def h_field(self):
        """Depth :class:`FieldHandle` — raises ``KeyError`` if the model
        has no ``h`` field (such models should use :class:`HLL`)."""
        return self.find_field("h")

    def _star_state(self, q, aux, h, un, s_side, s_star, n, mom_idx, den=None):
        """HLLC star state on one side of the contact wave.

        ``den`` overrides the contact denominator.  ``None`` (the flux path,
        unchanged) uses ``s_side - s_star + eps``, whose ``+eps`` is
        sign-BLIND: on the left side ``s_L - s*`` is negative, so adding the
        model's wet/dry eps pushes it TOWARDS zero and can flip its sign.
        :meth:`star_state` therefore passes a sign-correct floor instead —
        that path needs the ratio's sign to be structural, the flux path
        never depended on it.
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
        """Build the HLLC three-wave data ONCE: ``(QLs, QRs, sL, sR, s_star,
        hL, hR)``.

        Shared by :meth:`_compute_flux` and :meth:`star_state` so the star
        state the coupling reads is the SAME object the flux is built from —
        the same reason ``_path_integral_matrix`` is shared between the
        fluctuations and the in-cell source.  ``sign_safe`` swaps the contact
        denominator for the sign-correct floor (see :meth:`_star_state`).
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
        """Internal helper `_compute_flux`."""
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
        """The CONTACT-RESOLVED interface state — HLLC's own star, promoted.

        Same contract and same guarantee as :meth:`Numerics.star_state`, but
        built from :meth:`_star_state` (the star this solver already uses for
        its flux) rather than the two-wave average, so the contact / shear
        wave HLL smears is resolved.  The side is selected by the contact
        position, ``conditional(s* >= 0, Q_L*, Q_R*)`` — the state standing on
        the interface.

        POSITIVITY — AND WHY IT IS THE SAME GUARANTEE, NOT A WEAKER ONE.  On
        the left side

            h*_L = h_L (s_L - u_L·n) / (s_L - s*),

        whose numerator is <= 0 and denominator < 0 under the ordering
        ``s_L <= u_L·n`` and ``s_L <= s*``; mirrored on the right.  So each
        side's star depth is non-negative for the same wave-speed ordering the
        base class needs, and the sign of that ratio is made STRUCTURAL here
        by the sign-correct denominator floor (``sign_safe=True``) rather than
        left to the flux path's sign-blind ``+eps``.

        It is in fact the IDENTICAL depth.  Write ``A = h_R (s_R - u_R·n)``
        and ``B = h_L (u_L·n - s_L)``, the two non-negative terms of the HLL
        star numerator.  The contact speed used above — the depth-weighted HLL
        middle state, the one ``_compute_flux`` already builds — is
        ``s* = (s_L A + s_R B)/(A + B)``, so ``s* - s_L = B (s_R - s_L)/(A +
        B)`` and therefore

            h*_L = B / (s* - s_L) = (A + B)/(s_R - s_L) = h*_HLL,

        and the same on the right.  HLLC's star DEPTH equals HLL's exactly;
        the contact resolution shows up only in the MOMENTUM, tangential and
        higher-moment rows.  Consequences worth stating plainly: the
        ``[0, h_L + h_R]`` bound carries over unchanged (this is not a weaker
        class), and — the practical point for a coupling boundary — the STAGE
        does not depend on which member of the family is chosen.  The solvers
        differ in the momentum they put under that stage, not in the stage.
        Measured on 1764 finite face pairs spanning ``h`` in ``[1e-6, 50]``
        and ``|u| <= 30``: max relative depth deviation HLL vs HLLC 9.6e-3
        (the residue of the two ``eps`` desingularisations — ``h + eps`` in
        :meth:`_normal_velocity` and ``den + eps`` in ``s*`` — which break the
        identity numerically but not structurally), against up to 600x
        relative difference in the momentum row.
        """
        qL, qR, auxL, auxR = self._star_face_states()
        p, n = self.parameters, self.normal
        QLs, QRs, _sL, _sR, s_star, hL, hR = self._star_pair(
            qL, qR, auxL, auxR, p, n, sign_safe=True)
        sel = ZArray(conditional(s_star >= 0, QLs, QRs))
        out = self._clamp_star_depth(sel, hL, hR)
        return ZArray(list(sp.flatten(out))).reshape(self.n_variables, 1)


class PositiveRusanov(Rusanov):
    """PositiveRusanov. (class).

    Hydrostatic reconstruction follows Audusse-Bristeau-Klein:
    ``h_L* = max(0, h_L + b_L − b*)``, ``b* = max(b_L, b_R)``.
    The ``h``, ``b`` and (optional) ``hinv`` fields are resolved via
    :meth:`Numerics.find_field` so the same logic works whether
    bathymetry is part of conservative state or lives in ``Qaux``.
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
        """Depth :class:`FieldHandle`."""
        return self.find_field("h")

    @property
    def b_field(self):
        """Bathymetry :class:`FieldHandle`."""
        return self.find_field("b")

    @property
    def hinv_field(self):
        """Optional ``1/h`` :class:`FieldHandle` (``None`` if absent)."""
        return self.find_field("hinv", required=False)

    def hydrostatic_reconstruction(self, qL, qR, auxL, auxR):
        """Hydrostatic reconstruction."""
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
        """The Audusse-reconstructed pair — the Riemann problem this scheme's
        own flux solves, so ``star_state`` describes the same one.

        ``h_L*, h_R* = max(0, h + b - b*) >= 0`` by construction, so the
        depth floor in :meth:`Numerics._clamp_star_depth` is inactive on its
        INPUTS as well as on its output.  The star depth is measured from
        ``b* = max(b_L, b_R)``; see :meth:`Numerics._star_face_states`.
        """
        return self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )

    def numerical_flux(self):
        """Numerical flux."""
        qLs, qRs, qauxL, qauxR = self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )
        return self._compute_flux(qLs, qRs, qauxL, qauxR, self.parameters, self.normal)

    def numerical_fluctuations(self):
        """Numerical fluctuations."""
        qLs, qRs, qauxL, qauxR = self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )

        P_mat_L_raw = self._model_eval(
            "hydrostatic_pressure",
            self.variables_minus,
            self.aux_variables_minus,
            self.parameters,
        )
        P_mat_R_raw = self._model_eval(
            "hydrostatic_pressure",
            self.variables_plus,
            self.aux_variables_plus,
            self.parameters,
        )
        P_mat_L_star = self._model_eval(
            "hydrostatic_pressure", qLs, qauxL, self.parameters
        )
        P_mat_R_star = self._model_eval(
            "hydrostatic_pressure", qRs, qauxR, self.parameters
        )

        Dm_jump = (P_mat_L_raw - P_mat_L_star) @ self.normal
        Dp_jump = (P_mat_R_star - P_mat_R_raw) @ self.normal

        base_fluct = super().numerical_fluctuations()
        Dp_base = ZArray(base_fluct[0, :])
        Dm_base = ZArray(base_fluct[1, :])

        Dp = Dp_base + Dp_jump
        Dm = Dm_base + Dm_jump

        return ZArray([Dp, Dm])


class NonconservativeRusanov(Rusanov):
    """NonconservativeRusanov. (class)."""
    name = param.String(default="NonconservativeRusanovV2")
    integration_order = param.Integer(default=3)
    # Kernels verified bit-identical between per-face and face-batched
    # evaluation — the numpy solver may evaluate them with a trailing
    # face axis (see solver_numpy.flux_operator).
    supports_batched_faces = True

    def get_path_integral_states(self):
        """Get path integral states."""
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
        """Get viscosity identity flux."""
        n = self.n_variables
        return ZArray.zeros(n, n)

    def get_viscosity_identity_fluctuations(self):
        """Get viscosity identity fluctuations."""
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
        """Numerical fluctuations."""
        qLs, qRs, qauxL, qauxR = self.get_path_integral_states()
        nc_fluct = self._compute_fluctuations(
            qLs, qRs, qauxL, qauxR, self.parameters, self.normal
        )
        out = super().numerical_fluctuations()
        return out + nc_fluct

    def _call_model_matrix(self):
        """Internal helper `_call_model_matrix`."""
        return lambda Q, Qaux, p: self._model_eval("nonconservative_matrix", Q, Qaux, p)

    def _path_integral_matrix(self, qL, qR, auxL, auxR, p, n):
        """``int_0^1 A(Psi(s))·n ds`` along the segment path ``Psi(s) = qL +
        s·(qR - qL)``, by Gauss-Legendre at ``integration_order``.

        SHARED by the face fluctuations and the in-cell source
        (:meth:`numerical_cell_edge_source`), and that sharing is REQUIRED, not
        just tidy: well-balancing at second order comes from cancellation
        between those two, and they are two halves of ONE nonconservative
        product.  Discretise them with different rules and the cancellation is
        inexact — well-balanced at order 1, silently broken at order 2."""
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
        """The in-cell nonconservative term — see the base-class docstring.

        ``int_{T_ij} B(W)·grad W dx = [int_0^1 B(Psi(s))·Psi'(s) ds]·(W_ij -
        W_i)``: with a linear reconstruction ``grad W`` is constant on the
        subcell, so it factors out and what remains is the SAME segment path
        integral the fluctuations use — only with endpoints (cell, edge)
        instead of (minus, plus).  Hence no dissipation term: this is an
        integral over a smooth reconstruction, not a Riemann problem.

        Vanishes identically when ``q_edge == q_cell``, i.e. at first order,
        with no branch on the reconstruction order."""
        qc, qe = self.variables_minus, self.variables_plus
        auxc, auxe = self.aux_variables_minus, self.aux_variables_plus
        A_int = self._path_integral_matrix(
            qc, qe, auxc, auxe, self.parameters, self.normal)
        return A_int @ (qe - qc)

    def _compute_fluctuations(self, qL, qR, auxL, auxR, p, n):
        """Internal helper `_compute_fluctuations`."""
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
    """Path-conservative Rusanov with equilibrium-variable fluctuation
    viscosity for free-surface (lake-at-rest) well-balancing.

    Both the bed-slope ``g·h·∂_x b`` and the hydrostatic pressure
    ``g·h·∂_x h`` live in the nonconservative product (the "Malaga"
    formulation; ``hydrostatic_pressure`` is empty, ``b`` is a trivial
    conserved state with ``∂_t b = 0``).  The base
    :meth:`NonconservativeRusanov.get_viscosity_identity_fluctuations`
    already zeros the stationary-bed row.  This subclass additionally
    couples the depth-continuity row to the bed column so the Rusanov
    dissipation acts on the free-surface jump ``Δη = Δh + Δb`` instead
    of ``Δh``.  At lake-at-rest ``Δη = 0`` while ``Δh = −Δb ≠ 0``, so
    only the coupled form vanishes — giving exact well-balancing for
    both Rusanov *and* HLL-flavoured path integrals.

    For ``Q = [b, h, hu]`` the fluctuation viscosity becomes::

        [[0, 0, 0],
         [1, 1, 0],     # continuity dissipates on Δη = Δh + Δb
         [0, 0, 1]]

    Model-derived — the coupling is added only when both ``h`` and
    ``b`` resolve to conservative-state fields via
    :meth:`Numerics.find_field`.  A plain SWE model with no bed gets the
    unmodified identity (no bed-indexed term at all), as required.
    """

    name = param.String(default="WellBalancedNonconservativeRusanovV2")

    def get_viscosity_identity_fluctuations(self):
        """Get viscosity identity fluctuations (equilibrium-coupled)."""
        Id = super().get_viscosity_identity_fluctuations()
        h = self.find_field("h", required=False)
        b = self.find_field("b", required=False)
        if (h is not None and b is not None
                and h.container == "q" and b.container == "q"):
            Id[h.index, b.index] = 1
        return Id


class PositiveHLL(HLL):
    """HLL with Audusse-Bristeau-Klein hydrostatic reconstruction.

    Mirrors :class:`PositiveRusanov` but uses the sharper HLL
    two-wave numerical flux underneath instead of LF/Rusanov
    dissipation.  Recommended for free-surface dam-break / wet-dry
    simulations: the hydrostatic reconstruction enforces
    ``h_face ≥ 0`` (positivity), and HLL captures the rarefaction /
    shock fronts more accurately than Rusanov on the same mesh.

    Reconstruction (same as PositiveRusanov, Audusse-Bristeau-Klein):
    ``b* = max(b_L, b_R)``,  ``h_L* = max(0, h_L + b_L − b*)``,
    momentum scaled by ``h_L* / max(h_L, eps)``.

    The depth field ``h``, bathymetry ``b`` and (optional) ``1/h``
    inverse are resolved through :meth:`Numerics.find_field` — the
    same flux code works whether bathymetry is part of the
    conservative state or carried in ``Qaux``.
    """

    name = param.String(default="PositiveHLLV2")

    @property
    def h_field(self):
        return self.find_field("h")

    @property
    def b_field(self):
        return self.find_field("b")

    @property
    def hinv_field(self):
        return self.find_field("hinv", required=False)

    # Reuse the hydrostatic reconstruction from PositiveRusanov by
    # composition (don't multiply-inherit through Rusanov's flux).
    hydrostatic_reconstruction = PositiveRusanov.hydrostatic_reconstruction
    # Same composition for the star-state face pair: the star is built on the
    # reconstructed states, exactly as ``numerical_flux`` below is.
    _star_face_states = PositiveRusanov._star_face_states

    def numerical_flux(self):
        """HLL flux evaluated on the hydrostatically-reconstructed
        face states — positivity-preserving and well-balanced under
        the lake-at-rest steady state.

        The bed row is exactly zero by construction: HLL's
        ``_state_jump`` zeros ``dq[b]`` when ``b`` lives in the
        conservative state, and a well-posed SWE has ``F[b, :] = 0``
        symbolically.
        """
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
        """Audusse 2004 well-balancing consistency source ``S̃``.

        Audusse-Bouchut-Bristeau-Klein-Perthame 2004, SIAM J. Sci.
        Comput. 25(6):2050-2065, eq. (2.17)-(2.18):

            F_{i+1/2,L} = F_num(U*_L, U*_R) - S̃_L
            F_{i+1/2,R} = F_num(U*_L, U*_R) + S̃_R

            S̃_L = (0, ½g h_L² - ½g h*_L²)^T  =  (0, P_raw_L - P*_L)^T
            S̃_R = (0, ½g h*_R² - ½g h_R²)^T  =  (0, P*_R - P_raw_R)^T

        The HR'd numerical flux alone is not well-balanced; the per-
        cell-per-face S̃ correction restores consistency with the
        original SWE (paper, Theorem 2.5).  S̃ depends only on
        ``hydrostatic_pressure`` and is independent of the underlying
        Riemann solver — same formula already in
        :meth:`PositiveRusanov.numerical_fluctuations`.

        ``super().numerical_fluctuations()`` chains the next entry on
        the MRO.  For plain :class:`PositiveHLL` it lands at
        :meth:`Numerics.numerical_fluctuations` (returns zero).  For
        :class:`PositiveNonconservativeHLL` it lands at
        :meth:`NonconservativeRusanov.numerical_fluctuations` (the
        DLM path-integral over the NCP) — so the same override gives
        well-balanced HLL whether the model carries an NCP or not.
        """
        qLs, qRs, qauxL, qauxR = self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )

        P_mat_L_raw = self._model_eval(
            "hydrostatic_pressure",
            self.variables_minus,
            self.aux_variables_minus,
            self.parameters,
        )
        P_mat_R_raw = self._model_eval(
            "hydrostatic_pressure",
            self.variables_plus,
            self.aux_variables_plus,
            self.parameters,
        )
        P_mat_L_star = self._model_eval(
            "hydrostatic_pressure", qLs, qauxL, self.parameters
        )
        P_mat_R_star = self._model_eval(
            "hydrostatic_pressure", qRs, qauxR, self.parameters
        )

        Dm_jump = (P_mat_L_raw - P_mat_L_star) @ self.normal
        Dp_jump = (P_mat_R_star - P_mat_R_raw) @ self.normal

        base_fluct = super().numerical_fluctuations()
        Dp_base = ZArray(base_fluct[0, :])
        Dm_base = ZArray(base_fluct[1, :])

        return ZArray([Dp_base + Dp_jump, Dm_base + Dm_jump])


class PositiveNonconservativeRusanov(PositiveRusanov, NonconservativeRusanov):
    """PositiveNonconservativeRusanov. (class)."""
    name = param.String(default="PositiveNonconservativeRusanovV2")
    # HR kernels not yet batch-validated — keep the per-face loop.
    supports_batched_faces = False

    def get_path_integral_states(self):
        """Endpoints of the DLM path integral.

        Hydrostatic-reconstructed states, BUT ONLY when the model carries a
        ``hydrostatic_pressure`` slot for Audusse's consistency term ``S~`` to
        compensate.  Otherwise the RAW states.

        Why the distinction is load-bearing.  Audusse reconstructs ``h*`` from
        the bed so the pressure flux and the bed source balance at rest, and
        ``S~ = P_raw - P*`` restores what the reconstruction removed.  ``S~`` is
        built from ``hydrostatic_pressure``.  A path-conservative model leaves
        that slot EMPTY and carries ``g h dh`` in the nonconservative matrix, so
        ``S~`` is structurally absent -- and then the path integral runs between
        reconstructed endpoints with nothing compensating them.

        Continuity carries no nonconservative term, so this was previously
        argued not to touch the depth row.  That argument is wrong: the
        ENDPOINTS are still reconstructed, so the mass flux taken between
        ``h*_L`` and ``h*_R`` differs from the one between ``h_L`` and ``h_R``
        wherever ``db != 0``.  Measured on the width-averaged curved channel,
        that lost ``W db q00`` per face -- a steady-state discharge gain of
        1.55 percent per metre, mesh-independent, obeying ``dQ/ds = S0 Q/h``
        (0.01429 measured against 0.01443 predicted) and scaling exactly with
        the bed slope (2x slope -> +99.6 percent against +100.0 predicted).
        With raw endpoints the same model conserves to -0.004 percent and
        settles on its design operating point.

        Models that DO populate ``hydrostatic_pressure`` are unaffected: they
        keep the reconstructed endpoints and their well-balancing.
        """
        if getattr(self.model, "hydrostatic_pressure_is_zero", False):
            return (self.variables_minus, self.variables_plus,
                    self.aux_variables_minus, self.aux_variables_plus)
        return self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )


class NonconservativeRoe(NonconservativeRusanov):
    """Roe scheme.

    The Gauss-Legendre path-integral NCP advection term (as
    :class:`NonconservativeRusanov`), with the scalar Rusanov viscosity
    ``s_max*Id*dQ`` replaced by the Roe matrix dissipation
    ``|A(Q*)|*dQ = R|Lambda|L*dQ`` at the midpoint state
    ``Q* = 1/2 (Q_L + Q_R)``.  Each characteristic field is upwinded by its own
    ``|lambda|`` (vs a single max wave speed), so shocks and contacts are markedly
    sharper than Rusanov; the stationary bed mode (lambda=0) contributes
    ``|0| = 0`` automatically -- no special bed handling.

    The eigendecomposition is sourced from the model's ``eigensystem`` operator:
    the SystemModel forms the normal-projected quasilinear matrix
    ``A_n = Sum_d n_d * quasilinear_matrix[:, :, d]`` and wraps it in the opaque
    :class:`zoomy_core.model.kernel_functions.eigensystem` kernel, which the
    backends realise numerically (``np.linalg.eig`` / Eigen via UserFunctions).
    The scheme then assembles ``|A| = R|Lambda|L`` symbolically on top -- no
    closed-form symbolic spectral derivation, so it is general over ANY
    SystemModel (SWE with an analytic spectrum, or SME/VAM without one).  The
    printer emits the ``eigensystem(...)`` calls as opaque function applications;
    each backend's UserFunctions supplies the implementation.

    Roe is well-balanced by construction (the path-integral fluctuations carry
    the bed coupling and the lambda=0 bed mode is exact), so it is built on the
    plain :class:`NonconservativeRusanov` -- NO hydrostatic reconstruction.
    """

    name = param.String(default="NonconservativeRoe")

    def _abs_quasilinear(self, q_state, aux_state, p, n):
        """``|A_n| = R|Lambda|L`` of the normal-projected quasilinear matrix at
        ``q_state``, assembled here (the 'real Roe' part) from the model's
        ``eigensystem`` operator — which evaluates ``quasilinear_matrix`` and
        applies the opaque ``eigensystem`` kernel (numerical eig in the backends:
        np.linalg.eig / Eigen).  No symbolic diagonalization, so it is general
        over any SystemModel (SWE, SME/VAM, HR-reconstructed)."""
        n_vars = self.n_variables
        es = self._model_eval("eigensystem", q_state, aux_state, p, n)
        R = sp.Matrix(n_vars, n_vars, lambda i, j: es[n_vars + i * n_vars + j])
        L = sp.Matrix(n_vars, n_vars,
                      lambda i, j: es[n_vars + n_vars * n_vars + i * n_vars + j])
        abs_lam = sp.diag(*[sp.Abs(es[k]) for k in range(n_vars)])
        return R @ abs_lam @ L

    def _compute_fluctuations(self, qL, qR, auxL, auxR, p, n):
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
    """HLL conservative flux + Audusse-Bristeau-Klein hydrostatic
    reconstruction + path-integral NCP fluctuations — dissipating ONCE,
    in the flux.

    Combines the sharper HLL two-wave numerical flux (vs. Rusanov / LF)
    with the same well-balanced reconstruction and path-integral NCP
    fluctuations as :class:`PositiveNonconservativeRusanov`.

    Dissipation placement — the two nonconservative families differ:

    * The Rusanov nonconservative family has a PURE CENTRAL flux
      (:meth:`NonconservativeRusanov.get_viscosity_identity_flux` is
      zero), so its single LF dissipation ``s_max·Id·Δq`` correctly
      lives in the fluctuations.
    * HLL's upwinding is STRUCTURAL — the ``sLm·sRp·Δq`` term inside
      :meth:`HLL._compute_flux` — it does not route through the
      viscosity-identity hooks.  The flux therefore already carries
      this scheme's dissipation, and this class zeros
      :meth:`get_viscosity_identity_fluctuations` so the fluctuation
      side carries ONLY the DLM path-integral NCP advection term plus
      the Audusse ``S̃`` consistency source.  (Inheriting the Rusanov
      fluctuation identity here dissipated TWICE — measured as an O(1)
      cell-update error vs :class:`PositiveHLL` on NCP-free
      configurations and a blow-up on river's R2 smooth-bump run at
      CFL 0.5 where :class:`PositiveHLL` ran clean: the doubled
      viscosity halves the scheme's stability bound.)

    The Python MRO ``(PositiveNonconservativeHLL → PositiveHLL → HLL →
    NonconservativeRusanov → Rusanov → Numerics)`` resolves
    ``numerical_flux`` to ``PositiveHLL.numerical_flux`` (hydrostatic
    reconstruction → HLL combine → bed-row mask) and
    ``numerical_fluctuations`` to ``PositiveHLL.
    numerical_fluctuations`` (the Audusse ``S̃`` source), which chains
    via ``super()`` into ``NonconservativeRusanov.
    numerical_fluctuations`` — the NCP path integral on the
    reconstructed states below, with its LF term disabled by the zero
    viscosity identity.
    """

    name = param.String(default="PositiveNonconservativeHLLV2")

    def get_viscosity_identity_fluctuations(self):
        """Zero matrix — single-dissipation split.

        The Malaga-style placement hook, used here to MOVE the
        dissipation, not remove it: the HLL flux upwinds structurally,
        so the LF ``s_max·Id·Δq`` the Rusanov nonconservative family
        adds in :meth:`NonconservativeRusanov._compute_fluctuations`
        would dissipate a second time on top.  The NCP path-integral
        advection term and the Audusse ``S̃`` source are unaffected.
        """
        n = self.n_variables
        return ZArray.zeros(n, n)

    def get_path_integral_states(self):
        """NCP path-integral evaluated on the hydrostatically-
        reconstructed face states."""
        return self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )


class QuasilinearRusanov(NonconservativeRusanov):
    """QuasilinearRusanov. (class)."""
    name = param.String(default="QuasilinearRusanovV2")

    def numerical_flux(self):
        """Numerical flux."""
        zeros = [sp.Integer(0)] * self.n_variables
        return ZArray(zeros)

    def _call_model_matrix(self):
        """Internal helper `_call_model_matrix`."""
        return lambda Q, Qaux, p: self._model_eval("quasilinear_matrix", Q, Qaux, p)


class PositiveQuasilinearRusanov(PositiveRusanov, QuasilinearRusanov):
    """PositiveQuasilinearRusanov. (class)."""
    name = param.String(default="PositiveQuasilinearRusanovV2")

    def numerical_flux(self):
        """Numerical flux."""
        zeros = [sp.Integer(0)] * self.n_variables
        return ZArray(zeros)

    def get_path_integral_states(self):
        """Get path integral states."""
        return self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )


class QuasilinearRoe(NonconservativeRoe):
    """Path-conservative Roe in FULLY QUASILINEAR form.

    :class:`NonconservativeRoe` inherits ``_call_model_matrix`` from
    :class:`NonconservativeRusanov`, so its Gauss-Legendre path integral runs
    over the ``nonconservative_matrix`` (``B`` alone) while its dissipation
    ``|A(Q*)|dQ`` is built on the FULL quasilinear matrix via the ``eigensystem``
    kernel.  That split is correct when the conservative part is carried by
    ``numerical_flux`` (flux-difference + NCP + Roe viscosity), but it is NOT
    what a characteristic INTERFACE needs: a coupling ghost built on
    ``A = dF/dQ + B`` must face a flux whose advection term is the SAME ``A``,
    otherwise ghost and flux disagree about what the interface transmits.

    This subclass applies the :class:`QuasilinearRusanov` pattern to Roe — the
    two changes belong together:

    * ``_call_model_matrix`` -> ``quasilinear_matrix``: the path integral
      becomes ``A_int = int_0^1 A(Q_L + s dQ) ds``, the Dal Maso-LeFloch-Murat /
      Pares-Castro path-conservative matrix for the segment path;
    * ``numerical_flux`` -> ``0``: ``dF/dQ`` is now INSIDE that integral, so
      keeping a conservative flux would count it twice (and would re-introduce
      the Rusanov ``s_max`` viscosity, which lives in
      ``get_viscosity_identity_flux``; the fluctuation-side identity is already
      zero, so with a zero flux the ONLY dissipation is Roe's ``|A|dQ``).

    Result: ``D^- + D^+ = A_int dQ`` — the generalized Rankine-Hugoniot
    condition for the segment path — with per-field upwinding ``R|Lambda|L``.
    Unguarded by design (no hydrostatic reconstruction, no entropy fix): the
    ``lambda = 0`` bed mode contributes ``|0| = 0`` exactly, and any failure of
    a one-sided linearization is meant to be OBSERVED, not masked.
    """

    name = param.String(default="QuasilinearRoe")

    def numerical_flux(self):
        """Zero: the conservative part rides inside the path integral."""
        return ZArray([sp.Integer(0)] * self.n_variables)

    def _call_model_matrix(self):
        """Internal helper `_call_model_matrix` — the FULL quasilinear matrix."""
        return lambda Q, Qaux, p: self._model_eval("quasilinear_matrix", Q, Qaux, p)

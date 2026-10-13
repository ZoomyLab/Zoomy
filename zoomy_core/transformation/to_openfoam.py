"""Foam printer for the SystemModel contract.

Trusts that the incoming `SystemModel` is complete and well-shaped
(emitted by Zoomy's own pipeline) — no defensive checks, no fallback
machinery.  Options that affect the *content* of the emitted C++ live as
printer flags, not as branching in the printer's plumbing.
"""

from __future__ import annotations

import itertools

import sympy as sp

from zoomy_core.misc.misc import ZArray
from zoomy_core.numerics.numerical_system_model import to_numerical_system_model
from zoomy_core.transformation.generic_c import GenericCppBase, GenericCppModel


def _promote_with_dry_gate(obj):
    """Coerce `obj` to an NSM, opting the depth eigenvalue gate back in.

    REQ-181 dropped `gate_eigenvalues_dry` from the depth
    `NumericalSystemModel.default_operations` ("we do not make this
    eigenvalues guard a default").  The foam pipeline hands the printer a raw
    `SystemModel` (`SystemModel.from_model(model)`), so this coercion
    is where foam used to receive the gate via the default.  Re-apply it here —
    under the SAME predicate the default used (transport system carrying a depth
    `h`) — so foam's generated eigenvalue code stays byte-identical.

    An already-built NSM is returned unchanged: its builder already chose its
    operations (and re-gating would double-wrap the `conditional`)."""
    from zoomy_core.numerics.numerical_system_model import NumericalSystemModel
    from zoomy_core.systemmodel.operations import gate_eigenvalues_dry
    if isinstance(obj, NumericalSystemModel):
        return obj
    nsm = to_numerical_system_model(obj)
    if nsm._is_transport_system() and any(str(s) == "h" for s in nsm.state):
        nsm.apply(gate_eigenvalues_dry())   # no-op when eigenvalues is None
    return nsm


# ── Legacy printer (unchanged) ───────────────────────────────────────────


# ── SystemModel printer ──────────────────────────────────────────────────
# (The legacy ``FoamModel`` Model-path printer was removed — foam codegen goes
#  through ``FoamSystemModelPrinter`` (SystemModel/NSM-native).)


_FOAM_ARG = {
    "Q": "const Foam::List<Foam::scalar>& Q",
    "Qaux": "const Foam::List<Foam::scalar>& Qaux",
    "W": "const Foam::List<Foam::scalar>& W",
    "p": "const Foam::List<Foam::scalar>& p",
    "dt": "const Foam::scalar dt",
    "n": "const Foam::vector& n",
    "X": "const Foam::vector& X",
    "time": "const Foam::scalar& time",
    "dX": "const Foam::scalar& dX",
    "bc_idx": "const int bc_idx",
    "z": "const Foam::scalar& z",
    "profile": "const Foam::List<Foam::scalar>& profile",
    "I": "const Foam::List<Foam::scalar>& I",
    "column": "const Foam::List<Foam::List<Foam::scalar>>& column",
}

_AXIS = ("x", "y", "z")


def _qualified_numerics_call(name):
    """Print an opaque UserFunctions leaf as `numerics::<name>(…)`.

    The leaves (`eigensystem` — the Roe `|A| = R|Λ|L` decomposition;
    `eigenvalues` — its λ-only companion; `solve` — the point-implicit
    source's per-cell `A⁻¹b`) are implemented in `UserFunctions.H` under
    `namespace numerics`, while the generated kernels live in ``namespace
    Model`` / `namespace Numerics`.  Unqualified lookup does not reach
    `numerics`, so the call MUST carry the namespace (REQ-187: it did not,
    and VAM / opaque-eigenvalue models failed to compile with "'eigenvalues'
    not declared in this scope" while analytic-eigenvalue SWE was unaffected).

    Shared by BOTH foam printers.  It used to sit on the Numerics printer
    alone, which left the SAME bug live in `Model.H` — and there
    `eigenvalues` is worse than unresolved, because the emitted kernel is
    itself named `eigenvalues` and the unqualified call binds to IT.
    """
    def _emit(printer, *args):
        return (f"numerics::{name}("
                + ", ".join(printer.doprint(a) for a in args) + ")")
    return _emit


_NUMERICS_QUALIFIED = {
    name: _qualified_numerics_call(name)
    for name in ("eigensystem", "eigenvalues", "gelfand_bound", "solve")
}

# Canonical 3D-field profile exchanged across a preCICE interface (Phase 7).
# ``interpolate_to_3d`` emits these in order; ``project_from_3d`` consumes
# them via fresh ``P3_<field>`` symbols mapped to ``profile[i]``.
_PROFILE_3D_FIELDS = ("b", "h", "u", "v", "w", "p")

# Column wrapper around the per-position interpolate kernel: makes
# interpolate_to_3d operate on one full interface COLUMN (the higher-dim solver
# dictates the column z; the lower-dim solver lifts to it).  ``project_from_3d``
# has NO foam-only wrapper — it is emitted in the generic-C ``I[j]``
# column-quadrature form (see ``_emit_projection_kernels``): the C++ driver
# supplies the reduced ``profile`` and the ζ-quadrature accumulators ``I``.
_INTERPOLATE_COLUMN_WRAPPER = """inline Foam::List<Foam::List<Foam::scalar>> interpolate_to_3d(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::List<Foam::scalar>& p,
    const Foam::List<Foam::scalar>& z)
{
    Foam::List<Foam::List<Foam::scalar>> out(z.size());
    forAll(z, k) out[k] = interpolate_to_3d_at(Q, Qaux, p, z[k]);
    return out;
}"""


class FoamSystemModelPrinter(GenericCppBase):
    """Foam printer for a frozen `SystemModel`.

    Emits `Model.H` with one kernel per operator matrix.  Per-direction
    kernels (`flux_x` / `_y` / `_z` etc.) match the calling
    convention of the existing hand-written `numerics.H`.

    Options
    -------
    analytical_eigenvalues : bool, default False
        If True, emit a REAL spectrum: the SystemModel's symbolic
        `eigenvalues` when it has them, else the numerical spectrum off
        `quasilinear_matrix`.  If False, emit a zero placeholder — the
        solver computes eigenvalues numerically itself.  Either way this
        slot is NOT the wave speed: that is
        `Numerics::local_max_abs_eigenvalue`, which takes the closed-form
        `spectral_radius_bound` and never an eigensolve.
    """

    _output_subdir = ".foam_interface"
    real_type = "Foam::scalar"
    math_namespace = "Foam::"
    # The BC kernel's ``idx`` group has no entry in the shared C-family
    # ``ARG_MAPPING``; the C family (foam included) spells it ``bc_idx``.
    # Pure spelling — the group itself is declared on the BC Function.
    ARG_MAPPING = {**GenericCppBase.ARG_MAPPING, "idx": "bc_idx"}
    # Opaque UserFunctions leaves resolve through ``numerics::`` HERE too, not
    # just in the Numerics printer — see :func:`_qualified_numerics_call`.
    c_functions = {**GenericCppBase.c_functions, **_NUMERICS_QUALIFIED}
    analytical_eigenvalues = False
    # Phase 7 coupling: the inverse 3D→2D map.  Read from the model-owned
    # ``sm.project_from_3d`` slot (filled by ``register_group("project", …)``)
    # — the ``project_from_3d=`` kwarg remains as an explicit override.
    # None (no registration, no kwarg) ⇒ not emitted.
    project_from_3d = None
    # C++ namespace the kernels are emitted into.  The default ``Model``
    # matches the single-system foam interface; the Chorin split printer
    # overrides it per sub-system (predictor / pressure / corrector) so the
    # three sub-systems coexist in one driver.
    namespace_name = "Model"
    # REQ-40 Chorin split: a free time-step symbol (e.g. the ``dt`` baked into
    # the pressure elliptic source and the corrector update by
    # ``split_for_pressure_structural``).  When set it is appended to the
    # parameter vector ``p`` as the last slot — the exact convention the JAX
    # ChorinSplit solver uses (``_params_with_dt`` sets ``p[-1] = dt``) — so a
    # bare ``dt`` symbol prints as ``p[n_parameters - 1]`` and the C++ driver
    # writes the current step size into that slot.  ``None`` ⇒ unchanged.
    dt_symbol = None

    def __init__(self, sm, **opts):
        super().__init__()
        # Normalise the entry: accept a Model, a SystemModel, or an NSM.
        self.sm = sm = _promote_with_dry_gate(sm)
        # Apply printer options first so ``dt_symbol`` is in effect before the
        # parameter symbol map (which may append it) is built.
        for k, v in opts.items():
            setattr(self, k, v)
        self.register_map("Q", list(sm.state))
        self.register_map("Qaux", list(sm.aux_state))
        self.register_map("n", list(sm.normal.values()))
        self.register_map("p", self._parameter_symbols())
        # The per-cell update kernels carry an explicit scalar ``dt`` argument.
        # Registered AFTER ``p`` so that a pressure sub-system which bakes dt
        # into the parameter vector (``dt_symbol``) still resolves dt → p[last]
        # in its source; for any other sub-system the bare ``dt`` arg wins.
        self.symbol_maps.append({sp.Symbol("dt", positive=True): "dt"})
        if self.project_from_3d is None:
            self.project_from_3d = sm.project_from_3d

    def _parameter_symbols(self):
        """Ordered parameter Symbols for the `p` interface — the
        SystemModel's parameters, plus `dt_symbol` as a trailing slot when
        the Chorin split baked a time-step into the operators."""
        p_syms = list(self.sm.parameters.values())
        if self.dt_symbol is not None:
            p_syms = p_syms + [self.dt_symbol]
        return p_syms

    def _parameter_keys_and_values(self):
        """(names, default_values) for the emitted `parameter_names` /
        `default_parameters` — mirrors `_parameter_symbols`."""
        keys = list(self.sm.parameters.keys())
        vals = list(self.sm.parameter_values.values())
        if self.dt_symbol is not None:
            keys = keys + ["dt"]
            vals = vals + [0.0]
        return keys, vals

    # ── Foam syntax hooks ────────────────────────────────────────────────

    def format_accessor(self, var, idx):
        if var in ("n", "X") and idx < 3:
            return f"{var}.{('x()', 'y()', 'z()')[idx]}"
        return f"{var}[{idx}]"

    def format_assignment(self, target, indices, value, shape):
        return f"{target}{''.join(f'[{i}]' for i in indices)} = {value};"

    def _print_Abs(self, expr):
        """Foam has no `Foam::fabs`; canonical abs is `Foam::mag`."""
        return f"Foam::mag({self._print(expr.args[0])})"

    def _foam_type(self, shape):
        if not shape:
            return self.real_type
        return f"Foam::List<{self._foam_type(shape[1:])}>"

    def _foam_init(self, shape):
        if len(shape) == 1:
            return f"Foam::List<{self.real_type}>({shape[0]}, 0.0)"
        return (
            f"Foam::List<{self._foam_type(shape[1:])}>"
            f"({shape[0]}, {self._foam_init(shape[1:])})"
        )

    def get_array_declaration(self, target, shape, init_zero=False):
        """Foam-flavoured `auto res = Foam::List<...>(...);` decl, used
        by the inherited `convert_expression_body` in place of the
        base's `SimpleArray<T, N>` declaration."""
        return f"auto {target} = {self._foam_init(shape)};"

    def wrap_function_signature(self, name, args_str, body_str, shape):
        """Signature wrapper — the body already declares `res` and
        returns it, so we only emit the surrounding function."""
        return (
            f"\ninline {self._foam_type(shape)} {name}(\n"
            f"    {args_str})\n"
            f"{{\n"
            f"{body_str}\n"
            f"}}\n"
        )

    # ── Emission ─────────────────────────────────────────────────────────

    def _kernel(self, name, expr, shape, args):
        body = self.convert_expression_body(expr, shape)
        sig = ",\n    ".join(_FOAM_ARG[a] for a in args)
        return self.wrap_function_signature(name, sig, body, shape)

    def _coord_symbol_map(self):
        """REQ-185: bind the time scalar `t` → `time` and the position
        VECTOR components `x/y/z` → `X.x()/X.y()/X.z()` for the `source`
        kernel (same convention the BC kernel uses).  Scoped-pushed only around
        that kernel so it never shadows the `interpolate_to_3d` z-mapping."""
        cmap = {self.sm.time: "time"}
        pos = getattr(self.sm, "position", None)
        if pos is not None:
            for i, s in enumerate(pos.values()):
                cmap[s] = self.format_accessor("X", i)
        return cmap

    def _slice(self, tensor, axis_idx, out_shape):
        """`tensor[..., axis_idx]` reshaped to `out_shape`.  If
        `out_shape` has a trailing `1` padding (the `flux_x` column
        convention), walk one fewer axis when collecting source values."""
        walk = (
            out_shape[:-1]
            if (len(out_shape) == len(tensor.shape) and out_shape[-1] == 1)
            else out_shape
        )
        flat = [
            tensor[(*idx, axis_idx)]
            for idx in itertools.product(*(range(s) for s in walk))
        ]
        return sp.Array(flat).reshape(*out_shape)

    def _per_direction(self, base, tensor, out_shape, args):
        return [
            self._kernel(
                f"{base}_{_AXIS[d]}",
                self._slice(tensor, d, out_shape),
                out_shape,
                args,
            )
            for d in range(self.sm.dimension)
        ]

    def create_code(self):
        from zoomy_core.model.boundary_conditions import Coupled
        sm = self.sm
        n_eq, n_state = sm.n_equations, len(sm.state)
        # ``_bc_source`` is the original BoundaryConditions list — pure tag /
        # preCICE metadata.  Chorin sub-systems (from the splitter, which is
        # out of this printer's scope) share the parent's indexed BC *kernel*
        # — what actually gets emitted — but not this metadata object; fall
        # back to empty tag/preCICE lists when it is absent.
        bc_source = getattr(sm, "_bc_source", None)
        bc_dict = (bc_source.boundary_conditions_list_dict
                   if bc_source is not None else {})
        bc_tags = sorted(bc_dict.keys())
        bc_str = ", ".join(f'"{t}"' for t in bc_tags)
        p_keys, p_default = self._parameter_keys_and_values()
        p_names = ", ".join(f'"{k}"' for k in p_keys)
        p_vals = ", ".join(str(v) for v in p_default)
        # preCICE-coupled patches (Phase 7): a patch↔mesh-name binding for
        # every Coupled BC.  Empty for models with no coupling.
        precice = [(t, bc_dict[t].mesh_name) for t in bc_tags
                   if isinstance(bc_dict[t], Coupled)]
        precice_patch_str = ", ".join(f'"{t}"' for t, _ in precice)
        precice_mesh_str = ", ".join(f'"{m}"' for _, m in precice)

        blocks = [
            "#pragma once",
            '#include "List.H"',
            '#include "vector.H"',
            '#include "scalar.H"',
            '#include "word.H"',
            '#include "volFields.H"',
            '#include "fvMesh.H"',
            "",
            f"namespace {self.namespace_name}",
            "{",
            f"constexpr int n_dof_q    = {n_eq};",
            f"constexpr int n_dof_qaux = {len(sm.aux_state)};",
            f"constexpr int n_parameters = {len(p_keys)};",
            f"constexpr int dimension  = {sm.dimension};",
            # REQ-190: standard numerical timestep cap (a wave-free/dry domain
            # steps at this value, not a magic floor) — the same dt_max the NSM
            # carries and every other backend reads.
            f"constexpr Foam::scalar dt_max = {float(sm.dt_max)};",
            f"const Foam::List<Foam::word> map_boundary_tag_to_function_index{{ {bc_str} }};",
            f"const Foam::List<Foam::word> parameter_names{{ {p_names} }};",
            f"inline Foam::List<Foam::scalar> default_parameters() {{ return {{ {p_vals} }}; }}",
            f"constexpr int n_precice_patches = {len(precice)};",
            f"const Foam::List<Foam::word> precice_patch_names{{ {precice_patch_str} }};",
            f"const Foam::List<Foam::word> precice_mesh_names{{ {precice_mesh_str} }};",
        ]

        # REQ-40: the row→state-slot map.  For a square ``from_model``
        # extraction this is the identity; for a rectangular Chorin sub-system
        # (predictor / pressure / corrector) it tells the driver which state
        # slot each emitted row writes — e.g. the pressure block's
        # ``equation_to_state_index`` is the pressure-mode indices ``[6, 7]``,
        # the corrector's is the velocity-mode indices ``[2, 3, 4, 5]``.
        e2s = sm.equation_to_state_index
        if e2s is not None:
            e2s_str = ", ".join(str(int(i)) for i in e2s)
            blocks.append(
                "const Foam::List<Foam::label> equation_to_state_index"
                f"{{ {e2s_str} }};"
            )

        # Every operator's argument list is read off the SystemModel's declared
        # ``Function.args`` (the single signature source) and spelled in Foam
        # syntax via ``_FOAM_ARG``.
        blocks += self._per_direction(
            "flux", sm.flux, (n_eq, 1), self._operator_arg_keys("flux")
        )
        blocks += self._per_direction(
            "nonconservative_matrix",
            sm.nonconservative_matrix,
            (n_eq, n_state),
            self._operator_arg_keys("nonconservative_matrix"),
        )
        blocks += self._per_direction(
            "quasilinear_matrix",
            sm.quasilinear_matrix,
            (n_eq, n_state),
            self._operator_arg_keys("quasilinear_matrix"),
        )

        # ``analytical_eigenvalues=True`` asks for a REAL spectrum here.  When
        # the model carries none it used to hand ``None`` straight to
        # ``_kernel``, which dies inside the CSE pass with a bare
        # ``AttributeError: 'NoneType' object has no attribute 'replace'`` —
        # so every foam emission of an SME model stopped working the moment
        # ``symbolic_spectrum`` defaulted off, with no hint of why.  Fall back
        # to the NUMERICAL spectrum, which is what "a real spectrum, no closed
        # form" means everywhere else in the stack.  It is not the wave-speed
        # path (that is ``Numerics::local_max_abs_eigenvalue``, which takes
        # ``spectral_radius_bound`` and never an eigensolve) and no foam solver
        # reads ``Model::eigenvalues`` today, so this puts no decomposition on
        # any hot path — it just stops the printer lying by omission.
        if self.analytical_eigenvalues:
            eig_expr = (sm.eigenvalues if sm.eigenvalues is not None
                        else ZArray(list(sm.numerical_eigenvalues)
                                    ).reshape(n_eq, 1))
        else:
            eig_expr = sp.Array([[0]] * n_eq)
        blocks.append(
            self._kernel(
                "eigenvalues", eig_expr, (n_eq, 1),
                self._operator_arg_keys("eigenvalues")
            )
        )

        # ``source(Q, Qaux, p, time, dt, X)`` — declared signature.  The
        # coordinate map (t→time, x/y/z→X.x()/…) is scoped-pushed at highest
        # precedence only here, so it never shadows ``interpolate_to_3d``'s
        # z-mapping.  Bodies bind only what they reference.
        self.symbol_maps.insert(0, self._coord_symbol_map())
        try:
            blocks.append(
                self._kernel("source", sm.source, (n_eq, 1),
                             self._operator_arg_keys("source"))
            )
        finally:
            self.symbol_maps.pop(0)

        # REQ-40 (a): the mass matrix ``M(Q, Qaux, p)`` — the predictor
        # sub-system carries the non-trivial ``μ_k·h`` diagonal the driver
        # inverts when advancing the moments; pressure/corrector rows are
        # algebraic (all-zero rows).  Always emitted; for the single-system
        # interface it is the (often identity) operator matrix.
        blocks.append(self._emit_mass_matrix())

        # REQ-40 (c): the per-cell ``update_variables(Q, Qaux, p, dt)`` — for a
        # full model the state remap (h-clamp); for a corrector sub-system the
        # closed-form projection ``U_k ← U_k − dt/M_kk · T_u[k](P)`` (one entry
        # per row, scattered to ``equation_to_state_index``).  ALWAYS emitted:
        # the slot is ``None`` for any model without a per-cell remap — which,
        # with the wet/dry cap off by default, includes derived SME(level=0) —
        # and zoomyFoam.C:352 calls Model::update_variables unconditionally.
        # ``update_variables_or_identity`` supplies the documented identity.
        blocks.append(self._emit_update_variables())

        blocks.extend(self._emit_reconstruction_kernels())

        blocks.extend(self._emit_projection_kernels())

        blocks.append(self._emit_boundary_conditions())

        # Mesh-aware aux refresh, folded in from the (deleted) standalone
        # FoamUpdateAuxPrinter: ``update_aux_variables(Q, Qaux, dt, mesh)`` is a
        # sequence of ``numerics::compute_derivative(...)`` calls read straight
        # from ``sm.aux_registry`` (already minimal after the Phase-1 split — no
        # state_index_filter).  Lives inside the model namespace so its name
        # qualifies as ``{namespace}::update_aux_variables``.
        blocks.append(self._emit_update_aux_variables())

        # Mesh-free companion of the same algebraic closures, for a caller that
        # holds one state and no field — the preCICE interface Riemann flux.
        blocks.append(self._emit_pointwise_aux())

        # Companion for a split sub-model: fill the FROZEN predictor-forcing auxes
        # (aux_input_registry) that update_aux_variables does NOT re-derive — the
        # Poisson RHS driver.  Empty for a plain model (REQ-147).
        aux_inputs = self._emit_update_aux_input_variables()
        if aux_inputs:
            blocks.append(aux_inputs)

        blocks.append(f"}} // namespace {self.namespace_name}")
        return "\n".join(blocks)

    def _emit_update_aux_variables(self):
        """Emit `update_aux_variables(Q, Qaux, dt, mesh)` from
        `sm.aux_registry` — one `numerics::compute_derivative(...)` per
        derivative-kind aux (the solver-side helper does the LSQ / Gauss-grad
        on the OpenFOAM mesh).  Emitted verbatim from the registry, which the
        Phase-1 split already made minimal (no `state_index_filter`)."""
        sm = self.sm
        lines = [
            "inline void update_aux_variables(",
            "    const Foam::List<Foam::volScalarField*>& Q,",
            "    const Foam::List<Foam::volScalarField*>& Qaux,",
            "    const Foam::List<Foam::scalar>& p,",
            "    const Foam::scalar dt,",
            "    const Foam::fvMesh& mesh)",
            "{",
        ]
        lines += self._emit_deriv_aux_lines(sm.aux_registry)

        # ── Algebraic auxes (e.g. the KP-desingularized ``hinv``) ──────────
        # These are POINTWISE closures, not spatial derivatives, so the
        # mesh-aware ``compute_derivative`` leg above never assigns them — yet
        # every operator reads them (``Qaux[hinv]·q²`` …), so without this they
        # stay uninitialised garbage.  Emit each non-derivative
        # ``update_aux_variables`` row as a field-algebra assignment, AFTER the
        # derivative auxes (a closure may read one) and before any operator
        # consumes it.  ``Q[i]`` / ``Qaux[i]`` are ``volScalarField*`` handles
        # here, so the state/aux symbols lower to the dereferenced ``*Q[i]`` /
        # ``*Qaux[i]`` (OpenFOAM field algebra), mirroring the per-cell
        # algebraic leg the numpy/jax emitters write.
        state_syms, aux_syms = list(sm.state), list(sm.aux_state)
        param_syms = self._parameter_symbols()
        # REQ-183 (corrected): parameters are a RUNTIME INPUT to every
        # operator — the printer contract (``p`` is always in the interface).
        # The field-level refresh is no exception: it now takes ``p`` and each
        # parameter symbol lowers to its ``p[idx]`` slot, so a parameterized
        # aux (the KP ``hinv``'s ``wet_dry_eps``, Manning ``n_m`` …) resolves
        # at runtime and can be varied — NOT baked as literals.  Same p-index
        # order as the per-cell operators (``register_map("p", ...)``).
        deref = [
            {s: f"*Q[{i}]" for i, s in enumerate(state_syms)},
            {s: f"*Qaux[{i}]" for i, s in enumerate(aux_syms)},
            {s: f"p[{i}]" for i, s in enumerate(param_syms)},
        ]
        # REQ-185: a time/space-dependent aux (rain rate ``r_o =
        # Piecewise((rate, t<T),(0,True))``, a manufactured ``S(x)``) binds
        # ``t`` and the position components ``x/y/z``.  The field-level
        # refresh has the OpenFOAM ``mesh`` in scope, so ``t`` lowers to the
        # runtime time ``mesh.time().value()`` and ``x/y/z`` to the cell-
        # centre field components ``mesh.C().component(i)`` (field algebra) —
        # NOT a compile error.
        extra = {sm.time: "mesh.time().value()"}
        pos = getattr(sm, "position", None)
        if pos is not None:
            for i, s in enumerate(pos.values()):
                extra[s] = f"mesh.C().component({i})"
        deref.append(extra)
        lines += self._emit_algebraic_aux_lines(
            deref, set(extra), "*Qaux[{row}]", "the foam field-level aux refresh")
        lines.append("}")
        return "\n".join(lines)

    def _emit_algebraic_aux_lines(self, deref, extra_allowed, lhs, where):
        """Assignment lines for the ALGEBRAIC (non-derivative) aux rows.

        These are POINTWISE closures, not spatial derivatives, so the
        derivative leg of `_emit_update_aux_variables` never assigns
        them — yet every operator reads them (`Qaux[hinv]·q²` …), so without
        this they stay uninitialised garbage.  Emitted AFTER the derivative
        auxes (a closure may read one) and before any operator consumes it.

        ONE body, two lowerings: `deref` supplies the symbol → accessor maps
        and `lhs` the assignment target, so the mesh-aware field refresh
        (`*Qaux[i]` OpenFOAM field algebra) and the mesh-free
        `_emit_pointwise_aux` (`Qaux[i]` scalar list) share the same
        closure expressions instead of each carrying its own copy.
        """
        sm = self.sm
        uav = getattr(sm, "update_aux_variables", None)
        if uav is None or len(sp.flatten(uav)) == 0:
            return []
        rows = sp.flatten(uav)
        deriv_rows = {e["row"] for e in sm.aux_registry
                      if e["kind"] in ("derivative", "limited_derivative")}
        aux_syms = list(sm.aux_state)
        allowed = (set(sm.state) | set(aux_syms) | set(self._parameter_symbols())
                   | set(extra_allowed))
        out, saved_maps = [], self.symbol_maps
        try:
            self.symbol_maps = deref
            for row in range(len(aux_syms)):
                if row in deriv_rows or row >= len(rows):
                    continue
                expr = sp.sympify(rows[row])
                if expr == aux_syms[row]:
                    continue                           # identity passthrough
                unknown = expr.free_symbols - allowed
                if unknown:
                    raise NotImplementedError(
                        f"update_aux_variables row {row} ({aux_syms[row]}) "
                        f"references {unknown}; {where} cannot resolve them.")
                out.append(f"    // Qaux[{row}] ({aux_syms[row]}) = {expr}"
                           "  (algebraic closure)")
                out.append(f"    {lhs.format(row=row)} = {self.doprint(expr)};")
        finally:
            self.symbol_maps = saved_maps
        return out

    def _emit_pointwise_aux(self):
        """Emit `pointwise_aux(Q, p) -> Qaux` — the algebraic aux closures on
        a SINGLE scalar-list state, with no mesh.

        A caller that holds one state vector and no field (the preCICE coupling
        evaluates an interface Riemann flux on states RECONSTRUCTED from an
        exchanged water column, which belong to no cell) still needs the
        algebraic auxes the flux reads — `hinv` in every depth-based model.
        Derivative auxes stay 0: they are spatial, and a pointwise flux has no
        neighbourhood to take them from.

        The closure expressions come from the model's own
        `update_aux_variables` via `_emit_algebraic_aux_lines`, so the
        formula lives in core and is never restated backend-side.  A model
        whose algebraic aux binds `t` or a position raises rather than
        silently emitting a mesh-free approximation of it.
        """
        n_aux = len(self.sm.aux_state)
        deref = [
            {s: f"Q[{i}]" for i, s in enumerate(self.sm.state)},
            {s: f"Qaux[{i}]" for i, s in enumerate(self.sm.aux_state)},
            {s: f"p[{i}]" for i, s in enumerate(self._parameter_symbols())},
        ]
        lines = [
            "inline Foam::List<Foam::scalar> pointwise_aux(",
            "    const Foam::List<Foam::scalar>& Q,",
            "    const Foam::List<Foam::scalar>& p)",
            "{",
            f"    Foam::List<Foam::scalar> Qaux({n_aux}, 0.0);",
        ]
        lines += self._emit_algebraic_aux_lines(
            deref, set(), "Qaux[{row}]",
            "a pointwise (mesh-free) aux, which has no time or position")
        lines.append("    return Qaux;")
        lines.append("}")
        return "\n".join(lines)

    def _emit_deriv_aux_lines(self, registry):
        """`compute_derivative` (+ function-comment) lines for a derivative-kind
        aux registry — shared by `update_aux_variables` (live) and
        `update_aux_input_variables` (frozen predictor forcing).  `row` indexes
        the FULL `aux_state` (both registries share it) so the slots line up."""
        sm = self.sm
        aux_names = [str(s) for s in sm.aux_state]
        state_names = [str(s) for s in sm.state]

        def _resolve_source(entry):
            name = entry["target_name"]
            if name in state_names:
                return "Q", state_names.index(name)
            if name in aux_names:
                return "Qaux", aux_names.index(name)
            raise KeyError(
                f"aux registry entry {entry['name']!r} references unknown "
                f"target {name!r} — not found in state or aux_state."
            )

        out = []
        for entry in registry:
            row = entry["row"]
            name = entry["name"]
            if entry["kind"] in ("derivative", "limited_derivative"):
                src_container, src_idx = _resolve_source(entry)
                mi = entry["multi_index"]
                pad = tuple(mi) + (0,) * (3 - len(mi))
                out.append(f"    // Qaux[{row}] ({name}) = D^{mi} {src_container}[{src_idx}]")
                out.append(
                    f"    numerics::compute_derivative"
                    f"(*Qaux[{row}], *{src_container}[{src_idx}], "
                    f"{pad[0]}, {pad[1]}, {pad[2]}, mesh);")
            elif entry["kind"] == "function":
                out.append(
                    f"    // Qaux[{row}] ({name}) — user-supplied function "
                    f"loaded by the case directory; no computation.")
        return out

    def _emit_update_aux_input_variables(self):
        """Emit `update_aux_input_variables(Q, Qaux, mesh)` — fills the FROZEN
        predictor-forcing auxes (`sm.aux_input_registry`, e.g. the Chorin
        pressure's q/h/b derivatives that drive the Poisson RHS) via
        `compute_derivative`.  These slots are declared in `n_dof_qaux` but
        NOT re-derived by `update_aux_variables` (which only owns the live
        pressure derivatives), so without this they stay 0 and the pressure
        residual/RHS is identically 0 → P≡0 (REQ-147).  The driver calls this ONCE
        per step (inputs constant across the Krylov iterations) before the solve.
        Empty output for a plain model with no input registry (non-split path
        unchanged)."""
        reg = getattr(self.sm, "aux_input_registry", None) or []
        if not reg:
            return ""
        lines = [
            "inline void update_aux_input_variables(",
            "    const Foam::List<Foam::volScalarField*>& Q,",
            "    const Foam::List<Foam::volScalarField*>& Qaux,",
            "    const Foam::fvMesh& mesh)",
            "{",
        ]
        lines += self._emit_deriv_aux_lines(reg)
        lines.append("}")
        return "\n".join(lines)

    def _emit_mass_matrix(self):
        """Emit `mass_matrix(Q, Qaux, p) -> List[n_eq][n_state]`."""
        sm = self.sm
        return self._kernel(
            "mass_matrix", sm.mass_matrix,
            (sm.n_equations, len(sm.state)),
            self._operator_arg_keys("mass_matrix"),
        )

    def _emit_update_variables(self):
        """Emit the per-cell `update_variables(Q, Qaux, p, dt) -> List[n]`
        from `sm.update_variables_or_identity()`.  For a full model the values
        are the whole state remap; for a corrector sub-system one updated value
        per row, in the order of `equation_to_state_index`; for a model with
        no remap at all the identity (`Q` unchanged)."""
        sm = self.sm
        uv = sp.Array(sp.flatten(sm.update_variables_or_identity()))
        n = len(uv)
        return self._kernel(
            "update_variables", uv, (n,),
            self._operator_arg_keys("update_variables"))

    def _emit_reconstruction_kernels(self):
        """Emit `Model::reconstruction_variables(Q, Qaux, p)` (forward)
        and `Model::state_from_reconstruction(W, Qaux, p)` (inverse).

        Forward map uses the same Q/Qaux/p symbol scope as every other
        operator kernel — no extra symbol map needed.

        Inverse map is parameterised by fresh `WB_<state_name>` symbols
        generated by `reconstruction_inverse.invert_reconstruction`.
        Push a temporary symbol map for the emission so each WB symbol
        prints as `W[i]` (where `i` is the index of the matching
        state slot), then pop.
        """
        sm = self.sm
        # A SystemModel may carry no reconstruction maps (e.g. VAM and the
        # Chorin sub-systems use the default conservative reconstruction);
        # emit nothing then, mirroring ``_emit_projection_kernels``' skip.
        if (sm.reconstruction_variables is None
                or sm.state_from_reconstruction is None):
            return []
        n_state = len(sm.state)
        shape = (n_state,)

        # Forward map.
        fwd = self._kernel(
            "reconstruction_variables",
            sm.reconstruction_variables,
            shape,
            ["Q", "Qaux", "p"],
        )

        # Inverse map — build WB_* → W[i] using the *actual* symbols
        # that invert_reconstruction created (assumptions like real=True
        # mean a freshly-constructed Symbol("WB_b") would not match).
        wb_map = {}
        free = set()
        for expr in sp.flatten(sm.state_from_reconstruction):
            if hasattr(expr, "free_symbols"):
                free |= expr.free_symbols
        wb_by_name = {str(s): s for s in free if str(s).startswith("WB_")}
        for i, state_sym in enumerate(sm.state):
            wb_name = f"WB_{state_sym}"
            if wb_name in wb_by_name:
                wb_map[wb_by_name[wb_name]] = f"W[{i}]"
        self.symbol_maps.append(wb_map)
        try:
            inv = self._kernel(
                "state_from_reconstruction",
                sm.state_from_reconstruction,
                shape,
                ["W", "Qaux", "p"],
            )
        finally:
            self.symbol_maps.pop()

        return [fwd, inv]

    def _emit_projection_kernels(self):
        """Emit the coupling projections on one interface COLUMN, when defined:

        * `Model::interpolate_to_3d(Q, Qaux, p, z[N]) -> field[N][6]` — the
          canonical 3D field `[b,h,u,v,w,p]` evaluated at every z of the
          column.  Loops the per-position kernel `interpolate_to_3d_at`
          (from `sm.interpolate_to_3d`; `sm.position[2]` → scalar `z`).
        * `Model::project_from_3d(column, p) -> Q` — the inverse: reduce one
          sampled interface column back to the 2D state.  The project rows are
          the model's Integral-FREE, fixed-node Galerkin reduction (plain
          arithmetic in the column samples), so this lowers through the SAME
          generic kernel path as every other operator — no `Integral`/`I[]`
          quadrature accumulator and no foam-only `project_from_3d_at` +
          column-quadrature wrapper.  `_lower_project_from_3d` (shared with
          the generic-C printer) binds each sampled value `P3_<field>(ζ_j)`
          to its column slot `column[j][slot]`; the C++ driver supplies the
          resolved `column` and calls this directly.

        interpolate_to_3d and project_from_3d are inverse on a column.  A model
        with neither defined emits nothing (uncoupled cases unchanged).
        """
        sm = self.sm
        blocks = []

        p2 = sm.interpolate_to_3d
        # The base model returns zeros(6); only emit a real reconstruction.
        if p2 is not None and any(e != 0 for e in sp.flatten(p2)):
            shape = (len(sp.flatten(p2)),)
            z_map = {}
            if sm.position is not None:
                z_map[sm.position[2]] = "z"
            self.symbol_maps.append(z_map)
            try:
                blocks.append(self._kernel(
                    "interpolate_to_3d_at", p2, shape,
                    ["Q", "Qaux", "p", "z"],
                ))
            finally:
                self.symbol_maps.pop()
            blocks.append(_INTERPOLATE_COLUMN_WRAPPER)        # column wrapper

        lowered = self._lower_project_from_3d(self.project_from_3d)
        if lowered is not None:
            rows, prof_map, at_args = lowered
            self.symbol_maps.append(prof_map)
            try:
                blocks.append(self._kernel(
                    "project_from_3d", sp.Matrix(rows), (len(rows),), at_args,
                ))
            finally:
                self.symbol_maps.pop()

        return blocks

    def _emit_boundary_conditions(self):
        """Emit ``Model::boundary_conditions(bc_idx, time, X, dX, Q,
        Qaux, p, n)`` from the SystemModel's symbolic Piecewise kernel.

        The argument list is read off the BC Function's DECLARED `args`
        (idx, time, position, distance, variables, aux_variables,
        parameters, normal) — the same single-source pattern as
        `_operator_arg_keys` — and spelled in Foam syntax via
        `ARG_MAPPING` + `_FOAM_ARG`.

        Returns a `Foam::List<scalar>` of size `n_eq` — the
        boundary state for the branch matching `bc_idx`.
        """
        bc = self.sm.boundary_conditions
        # The Q / Qaux / p / n symbols are already mapped via __init__'s
        # register_map (they share Symbol identity with sm.state etc.).
        # Add scalar / position symbols specific to the BC kernel.
        extra_map = {}
        if bc.args.contains("idx"):
            extra_map[bc.args["idx"]] = "bc_idx"
        if bc.args.contains("time"):
            extra_map[bc.args["time"]] = "time"
        if bc.args.contains("distance"):
            extra_map[bc.args["distance"]] = "dX"
        if bc.args.contains("position"):
            pos = bc.args["position"]
            for axis in ("x", "y", "z"):
                if hasattr(pos, axis):
                    extra_map[getattr(pos, axis)] = f"X.{axis}()"

        self.symbol_maps.append(extra_map)
        try:
            # The BC kernel returns the full face state (one entry per state
            # variable).  For a square ``from_model`` system this equals
            # ``n_equations``; for a rectangular Chorin sub-system the row
            # count differs from the state count, so size off the state.
            shape = (len(self.sm.state),)
            body = self.convert_expression_body(bc.definition, shape)
            sig = ",\n    ".join(
                _FOAM_ARG[self.ARG_MAPPING[k]] for k in bc.args.keys()
            )
            return self.wrap_function_signature(
                "boundary_conditions", sig, body, shape
            )
        finally:
            self.symbol_maps.pop()

    @classmethod
    def write_code(cls, sm, output_path, **opts):
        with open(output_path, "w") as f:
            f.write(cls(sm, **opts).create_code())
        return output_path


# ── Numerics (Riemann) printer ───────────────────────────────────────────


# Args carried by symbolic Riemann functions → Foam parameter declaration.
# Keys match ``func_obj.args.keys()`` for the Numerics-registered functions.
_FOAM_NUMERICS_ARG = {
    "q_minus": "const Foam::List<Foam::scalar>& Q_minus",
    "q_plus": "const Foam::List<Foam::scalar>& Q_plus",
    "aux_minus": "const Foam::List<Foam::scalar>& Qaux_minus",
    "aux_plus": "const Foam::List<Foam::scalar>& Qaux_plus",
    "Q": "const Foam::List<Foam::scalar>& Q",
    "Qaux": "const Foam::List<Foam::scalar>& Qaux",
    "p": "const Foam::List<Foam::scalar>& p",
    "normal": "const Foam::vector& n",
    "n": "const Foam::vector& n",
}


class FoamNumericsPrinter(GenericCppBase):
    """Foam printer for a symbolic `Numerics` object (Rusanov,
    HLL, NonconservativeRusanov, …).

    Emits `Numerics.H` with one kernel per entry in
    `numerics.functions` — typically `numerical_flux`,
    `numerical_fluctuations`, `local_max_abs_eigenvalue`.  Body
    expressions are CSE-optimised by the inherited
    `convert_expression_body`; signatures use the Foam type
    aliases above.
    """

    _output_subdir = ".foam_interface"
    real_type = "Foam::scalar"
    math_namespace = "Foam::"
    c_functions = {**GenericCppBase.c_functions, **_NUMERICS_QUALIFIED}

    #: C++ namespace the kernels are emitted into, and the model header they
    #: are paired with.  Overridable exactly like
    #: :attr:`FoamSystemModelPrinter.namespace_name`, so a translation unit can
    #: carry a SECOND scheme beside its own — the preCICE interface, which
    #: solves the shared Riemann problem with the designated high-fidelity
    #: model's kernels while the bulk keeps its own.  The emitted body is
    #: self-contained (no ``Model::`` references), so only the header line and
    #: the namespace name differ.
    namespace_name = "Numerics"
    model_header = "Model.H"

    def __init__(self, numerics, **opts):
        super().__init__()
        self.numerics = numerics
        # NOTE on promotion: unlike ``FoamSystemModelPrinter``, this printer
        # CANNOT defend itself against an un-promoted ``numerics.model`` —
        # ``Numerics.__init__`` (riemann_solvers.py) ALREADY snapshotted
        # ``local_max_abs_eigenvalue`` / ``numerical_flux`` from it by the
        # time a ``Numerics`` instance reaches here, so promoting
        # ``numerics.model`` NOW would not change the (already-baked) kernel
        # bodies — and re-deriving a fresh NSM here (the pre-3337fe5
        # approach) builds a symbol twin that prints identically but compares
        # unequal to the symbols those bodies are expressed in terms of (see
        # the symbol-map comment below, cid=87).  A hard type check was tried
        # here and reverted: this printer is also legitimately constructed
        # directly from a ``Numerics(model=<bare SystemModel>)`` in tests that
        # probe raw (un-regularised) kernel shapes, so rejecting a non-NSM
        # ``numerics.model`` breaks that contract.  The actual fix is at the
        # CONSTRUCTION site — whoever builds ``numerics`` must promote first
        # (see ``_promote_with_dry_gate`` above and
        # ``zoomy_foam._pipeline._codegen`` / ``_codegen_chorin``, both of
        # which now do).
        self.sm = numerics.model
        # State / aux / parameter / normal symbol maps — sourced from
        # ``numerics.variables`` / ``.aux_variables`` / ``.parameters`` /
        # ``.normal``, NOT from a freshly re-derived NSM.
        #
        # ``numerics.functions`` (``numerical_flux``, ``local_max_abs_eigenvalue``,
        # …) are ``Function`` objects whose bodies were snapshotted at
        # ``Numerics.__init__`` time from EXACTLY these symbol objects
        # (``self.variables = ZArray(list(self.model.state))`` there — see
        # ``riemann_solvers.Numerics.__init__``). Re-deriving a NEW NSM here
        # (formerly via ``_promote_with_dry_gate(numerics.model)``) builds a
        # SEPARATE ``state`` list: ``construct_numerical`` (the Model ->
        # SystemModel -> NSM boundary, ``numerical_system_model.py``) replaces
        # every state/aux symbol that carries a sign assumption (e.g. depth
        # ``h = Symbol("h", positive=True)`` — every depth-based model declares
        # it thus) with a bare, assumption-free twin. That twin is a DIFFERENT
        # sympy object with the SAME name: it prints identically but compares
        # unequal (exactly the trap ``construct_numerical``'s own docstring
        # warns about, cid=87). A symbol map keyed by the twin then silently
        # misses the original in ``numerics.functions`` bodies whenever the two
        # diverge — e.g. ``numerics`` was built from a SystemModel that was
        # never itself promoted/derived through the NSM boundary (as
        # ``thesis/notebooks/coupling/cases/confluence/model_sme.py`` used to)
        # — and ``generic_c.GenericCppBase._print_Symbol`` falls through to
        # printing the BARE symbol name instead of an accessor (``h`` instead
        # of ``Q[1]``), which does not compile.
        #
        # Sourcing the map from ``numerics.*`` instead is unconditionally
        # correct: those are (by construction, in ``Numerics.__init__``) the
        # very objects every registered function is expressed in terms of, so
        # the map can never miss regardless of what ``numerics.model`` is or
        # how/whether it was promoted before ``Numerics`` was built.
        self.register_map("Q", list(numerics.variables))
        self.register_map("Qaux", list(numerics.aux_variables))
        self.register_map("n", list(numerics.normal))
        self.register_map("p", list(numerics.parameters))
        # Face-state symbols carried by the symbolic Numerics — wired
        # into the printer so they print as ``Q_minus[i]`` etc.
        self.register_map("Q_minus", list(numerics.variables_minus))
        self.register_map("Q_plus", list(numerics.variables_plus))
        self.register_map("Qaux_minus", list(numerics.aux_variables_minus))
        self.register_map("Qaux_plus", list(numerics.aux_variables_plus))
        self.register_map("flux_minus", list(numerics.flux_minus))
        self.register_map("flux_plus", list(numerics.flux_plus))
        for k, v in opts.items():
            setattr(self, k, v)

    # ── Foam syntax (shared with the SystemModel printer) ────────────────

    def format_accessor(self, var, idx):
        if var in ("n", "X") and idx < 3:
            return f"{var}.{('x()', 'y()', 'z()')[idx]}"
        return f"{var}[{idx}]"

    def format_assignment(self, target, indices, value, shape):
        return f"{target}{''.join(f'[{i}]' for i in indices)} = {value};"

    def _print_Abs(self, expr):
        """Foam has no `Foam::fabs`; canonical abs is `Foam::mag`."""
        return f"Foam::mag({self._print(expr.args[0])})"

    def _foam_type(self, shape):
        if not shape:
            return self.real_type
        return f"Foam::List<{self._foam_type(shape[1:])}>"

    def _foam_init(self, shape):
        if len(shape) == 1:
            return f"Foam::List<{self.real_type}>({shape[0]}, 0.0)"
        return (
            f"Foam::List<{self._foam_type(shape[1:])}>"
            f"({shape[0]}, {self._foam_init(shape[1:])})"
        )

    def get_array_declaration(self, target, shape, init_zero=False):
        return f"auto {target} = {self._foam_init(shape)};"

    def wrap_function_signature(self, name, args_str, body_str, shape):
        return (
            f"\ninline {self._foam_type(shape)} {name}(\n"
            f"    {args_str})\n"
            f"{{\n"
            f"{body_str}\n"
            f"}}\n"
        )

    # ── Emission ─────────────────────────────────────────────────────────

    def _generate_signature_from_function(self, func_obj):
        """Foam-typed parameter list built from `func_obj.args.keys()`."""
        return ",\n    ".join(_FOAM_NUMERICS_ARG[k] for k in func_obj.args.keys())

    def create_code(self):
        sm = self.sm
        blocks = [
            "#pragma once",
            '#include "List.H"',
            '#include "vector.H"',
            '#include "scalar.H"',
            f'#include "{self.model_header}"',
            "",
            f"namespace {self.namespace_name}",
            "{",
            f"constexpr int n_dof_q = {sm.n_equations};",
        ]
        for _name, func_obj in self.numerics.functions.items():
            blocks.extend(self._process_kernel_from_function(func_obj))
        blocks.append(f"}} // namespace {self.namespace_name}")
        return "\n".join(blocks)

    @classmethod
    def write_code(cls, numerics, output_path, **opts):
        with open(output_path, "w") as f:
            f.write(cls(numerics, **opts).create_code())
        return output_path


# ── Chorin projection split headers (REQ-40) ────────────────────────────


def write_chorin_split_headers(split, output_dir, dt_symbol, *,
                               predictor_ns="ChorinPredictor",
                               pressure_ns="ChorinPressure",
                               corrector_ns="ChorinCorrector"):
    """Write the three sub-model headers a C++ Chorin-projection driver needs
    from a VAM pressure split (`model.chorin_split(dt)` →
    `split_for_pressure_structural` → `(SM_pred, SM_press, SM_corr)`).

    No foam-only fork: this just LOOPS the ordinary
    `FoamSystemModelPrinter` over the three self-contained sub-systems
    and writes one full model header each.  The pressure-aux refresh is NOT a
    separate emission anymore — it rides inside `SM_press`'s own (already
    minimal) `update_aux_variables`, folded into its model header.

    * `Model.H`     — `namespace {predictor_ns}` : predictor ops
      (pressure-zeroed flux / NCP / source + `mass_matrix`);
    * `Pressure.H`  — `namespace {pressure_ns}`  : the elliptic
      `source(Q, Qaux, p)` linear in `(P, P_x, P_xx)` + the pressure-mode
      `equation_to_state_index` + its minimal `update_aux_variables`
      (P-derivative computes only);
    * `Corrector.H` — `namespace {corrector_ns}` : `update_variables`
      `U_k ← U_k − dt/M_kk · T_u[k](P)` + its `equation_to_state_index`.

    The corrector takes dt as an explicit kernel argument
    (`update_variables(Q, Qaux, p, dt)`), so dt is NOT baked into its
    parameter vector — only the pressure block carries `dt_symbol`.

    Returns the list of written paths.
    """
    import os
    specs = [
        ("Model.H", split.SM_pred, {"namespace_name": predictor_ns}),
        ("Pressure.H", split.SM_press,
         {"namespace_name": pressure_ns, "dt_symbol": dt_symbol}),
        ("Corrector.H", split.SM_corr, {"namespace_name": corrector_ns}),
    ]
    paths = []
    for fname, sub_sm, opts in specs:
        path = os.path.join(output_dir, fname)
        FoamSystemModelPrinter.write_code(sub_sm, path, **opts)
        paths.append(path)
    return paths

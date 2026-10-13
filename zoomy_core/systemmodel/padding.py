"""Square a rectangular SystemModel (`pad_to_square`): the Chorin predictor of a split acts on the
full state, its pressure rows frozen.
"""
from __future__ import annotations

import sympy as sp

from zoomy_core.misc.misc import ZArray
from zoomy_core.systemmodel.system_model import SystemModel


def pad_to_square(sm: SystemModel) -> SystemModel:
    """Return a square SystemModel whose operators have `n_state` rows.

    Makes a rectangular sub-system (the Chorin predictor of a split) consumable by a
    hyperbolic template (which assumes `n_equations == n_state`).  Rows at state indices
    *not* in `sm.equation_to_state_index` get all-zero operators
    (flux / source / NCP / etc.) — they contribute zero RHS to the RK,
    so the corresponding state slots stay frozen through the predictor
    substep.  This is precisely the Chorin semantics: predictor doesn't
    touch pressure modes.
    """
    n_eq = sm.n_equations
    n_st = sm.n_state
    if n_eq == n_st:
        return sm

    e2s = list(sm.equation_to_state_index)
    n_dim = sm.n_dim

    # Map: dest_state_row → src_eq_row.  Missing states get None.
    src_of_dest = {state_i: src_i for src_i, state_i in enumerate(e2s)}

    def _pad_rank2(arr, n_cols):
        """`(n_eq, n_cols)` → `(n_st, n_cols)` ZArray with zero rows
        at missing dest indices.  `arr` may be ZArray or sp.Matrix."""
        if arr is None:
            return None
        out_rows = []
        for state_i in range(n_st):
            if state_i in src_of_dest:
                src = src_of_dest[state_i]
                row = [sp.sympify(arr[src, j]) for j in range(n_cols)]
            else:
                row = [sp.S.Zero] * n_cols
            out_rows.append(row)
        return ZArray(out_rows, shape=(n_st, n_cols))

    def _pad_rank1(arr):
        """`(n_eq,)` → `(n_st,)` ZArray with zeros at missing indices."""
        if arr is None:
            return None
        out = []
        for state_i in range(n_st):
            if state_i in src_of_dest:
                src = src_of_dest[state_i]
                # arr may be rank-1 or rank-2 (n_eq, 1) — handle both.
                if len(arr.shape) == 1:
                    out.append(sp.sympify(arr[src]))
                else:
                    out.append(sp.sympify(arr[src, 0]))
            else:
                out.append(sp.S.Zero)
        return ZArray(out, shape=(n_st,))

    def _pad_rank3(arr):
        """`(n_eq, n_st_cols, n_dim)` → `(n_st, n_st_cols, n_dim)`."""
        if arr is None:
            return None
        n_st_cols = arr.shape[1]
        out = []
        for state_i in range(n_st):
            row = []
            if state_i in src_of_dest:
                src = src_of_dest[state_i]
                for j in range(n_st_cols):
                    inner = [sp.sympify(arr[src, j, d]) for d in range(n_dim)]
                    row.append(inner)
            else:
                for j in range(n_st_cols):
                    row.append([sp.S.Zero] * n_dim)
            out.append(row)
        return ZArray(out, shape=(n_st, n_st_cols, n_dim))

    # Pad each operator.  Source can be rank-1 (post-refactor) or (n_eq,1).
    new_flux = _pad_rank2(sm.flux, n_dim)
    new_P    = _pad_rank2(sm.hydrostatic_pressure, n_dim)
    new_B    = _pad_rank3(sm.nonconservative_matrix)
    new_S    = (_pad_rank2(sm.source, sm.source.shape[1])
                if len(sm.source.shape) == 2
                else _pad_rank1(sm.source))
    new_M    = _pad_rank2(sm.mass_matrix, n_st)
    new_E    = _pad_rank1(sm.eigenvalues) if sm.eigenvalues is not None else None

    # cid=50: the split stages arrive here ALREADY swept by the NSM default
    # operations (split_for_pressure_structural routes every stage through
    # ``to_numerical_system_model``), so their derived operators were
    # FROZEN-then-substituted (``−q²/h² → −q²·hinv²``; see
    # ``regularize_pow``).  Carry them through the padding — letting the
    # padded copy recompute them from the swept primaries would drop the
    # exact ``∂(q²/h)/∂h`` terms (``hinv`` is an independent symbol) and
    # corrupt every wavespeed.  The same goes for ``update_aux_variables``
    # (aux-indexed, no padding needed): dropping it would leave the ``hinv``
    # aux row at 0 and zero every reconstructed velocity.
    new_SJV  = _pad_rank2(sm.source_jacobian_wrt_variables, n_st)
    new_SJA  = _pad_rank2(sm.source_jacobian_wrt_aux_variables,
                          len(sm.aux_state))

    sm_square = SystemModel(
        time=sm.time,
        space=list(sm.space),
        state=list(sm.state),
        aux_state=list(sm.aux_state),
        parameters=sm.parameters,
        parameter_values=sm.parameter_values,
        flux=new_flux,
        hydrostatic_pressure=new_P,
        nonconservative_matrix=new_B,
        source=new_S,
        mass_matrix=new_M,
        eigenvalues=new_E,
        source_jacobian_wrt_variables=new_SJV,
        source_jacobian_wrt_aux_variables=new_SJA,
        equation_to_state_index=list(range(n_st)),
        boundary_conditions=sm.boundary_conditions,
        aux_boundary_conditions=sm.aux_boundary_conditions,
        boundary_gradients=sm.boundary_gradients,
        initial_conditions=sm.initial_conditions,
        aux_initial_conditions=sm.aux_initial_conditions,
        update_variables=sm.update_variables,
        update_aux_variables=sm.update_aux_variables,
        reconstruction_variables=sm.reconstruction_variables,
        state_from_reconstruction=sm.state_from_reconstruction,
    )
    sm_square.aux_registry = list(getattr(sm, "aux_registry", []) or [])
    sm_square.aux_input_registry = list(
        getattr(sm, "aux_input_registry", []) or [])
    # The WB mode is a NUMERICS choice that must survive padding.  This object
    # — not the caller's ``SM_pred`` — is what reaches the solver's
    # well-balanced hook, which reads ``getattr(sm, "equilibrium_reconstruction")``.
    # ``SystemModel(...)`` above defaults it to "none", so a model declaring
    # 'audusse' / 'bernoulli' had it silently dropped HERE even after the
    # splitter propagates it onto SM_pred: padding rebuilds the object.  Only
    # the square path was affected — ``n_eq == n_state`` returns ``sm`` itself
    # and so never lost it, which is why 1-D SWE/SME suites never caught this.
    sm_square.equilibrium_reconstruction = getattr(
        sm, "equilibrium_reconstruction", "none")
    # Frozen flux Jacobian (see the cid=50 note above): pad the stage's
    # materialized quasilinear cache row-wise so the padded copy NEVER
    # re-materializes it from the swept flux.
    if getattr(sm, "_quasilinear_matrix", None) is not None:
        sm_square.quasilinear_matrix = _pad_rank3(sm._quasilinear_matrix)
    sm_square.equation_names = (
        [sm.equation_names[src_of_dest[i]] if i in src_of_dest
         else f"_pad_{i}"
         for i in range(n_st)]
        if getattr(sm, "equation_names", None) is not None
        else None
    )
    sm_square.history = list(sm.history) + [{
        "name": "pad_to_square",
        "description": (
            f"padded {n_eq} → {n_st} rows; "
            f"zero ops at state indices "
            f"{[i for i in range(n_st) if i not in src_of_dest]}"
        ),
    }]
    # Propagate the BoundaryConditions CONTAINER (``_bc_source`` /
    # ``_aux_bc_source``) — dynamic attributes, NOT constructor fields, so the
    # ``SystemModel(...)`` above copies the lambdified BC *kernel*
    # (``boundary_conditions``) but silently drops the container.  The
    # boundary-tag remap (``boundary_face_branches``) reads
    # ``_bc_source.list_sorted_function_names`` to align the mesh's positional
    # face-tag order (create_2d: left,right,bottom,top) with the BC kernel's
    # ALPHABETICAL Piecewise-branch order (bottom,left,right,top).  Without the
    # container that remap is skipped and 2-D boundary faces route by mesh
    # position against alphabetical branches — inflow/outflow/lateral BCs
    # rotate (e.g. the outflow face inherits the inflow Dirichlet ``q=Q_IN``,
    # leaking mass out of the dry-shelf boundary cell until ``h`` goes
    # negative).  1-D is accidentally safe because mesh order == alphabetical
    # for {left,right}.  Restoring the container makes the padded predictor
    # route boundaries exactly like every other system.
    sm_square._bc_source = getattr(sm, "_bc_source", None)
    sm_square._aux_bc_source = getattr(sm, "_aux_bc_source", None)
    return sm_square

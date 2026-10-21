"""Solver-unification machinery: the Procedure/Statement IR, the solver-level
argument vocabulary and the `ExternalProcedure` contract.

Walkers live with the other emitters, in
`zoomy_core.transformation.procedure_c` (C family) and
`zoomy_core.transformation.procedure_python` (numpy / jax).

Import note: `zoomy_core.solver.external` is exposed LAZILY (PEP 562).
Building `REQUIRED_PROCEDURES` runs the
name-collision guard, which reads `GenericCppBase.c_functions` — and
`generic_c` itself imports the argument vocabulary from this package.  Eager
re-export would close that cycle at interpreter start; deferring the external
declarations to first attribute access keeps ``import
zoomy_core.transformation.generic_c`` working while the guard still runs (just
at first use, never skipped).
"""
from zoomy_core.solver.arg_slots import (
    SOLVER_ARG_KINDS,
    SOLVER_ARG_MAPPING,
    SOLVER_ARG_SLOTS,
    SOLVER_DECL_KINDS,
    solver_arg_kind,
)
from zoomy_core.solver.ir import (
    SOLVER_PREFIX,
    Assign,
    Call,
    ForEachFace,
    IfStatic,
    Procedure,
    ProcedureNameError,
    Statement,
    While,
    check_procedure_name,
    require_resolved,
    split_target,
    written_names,
)

_LAZY = {
    "REQUIRED_PROCEDURES", "ExternalProcedure", "MissingProcedureBody",
    "assert_procedure_bodies", "required_procedure_gaps",
}

#: The emitted march lives one level up from the IR and pulls in the NSM, so
#: it is deferred for the same reason ``external`` is.
_LAZY_MARCH = {
    "MARCH_FLAGS", "MarchProgram", "TABLEAUS", "TABLEAU_EULER",
    "TABLEAU_SSPRK2", "build_hyperbolic_step", "build_march",
    "build_should_write", "emit_march", "flags_from_nsm", "tableau_for",
}

_LAZY_CONSTANTS = {
    "ConstantResolutionError", "EPS_H_FALLBACK", "MarchConstants",
    "WRITE_EPS", "eigen_wave_speed_floor", "kp_eps", "march_constants",
}


def __getattr__(name):
    """Defer the `external` / `march` modules until first use (see the
    module note)."""
    if name in _LAZY:
        from zoomy_core.solver import external
        return getattr(external, name)
    if name in _LAZY_MARCH:
        from zoomy_core.solver import march
        return getattr(march, name)
    if name in _LAZY_CONSTANTS:
        from zoomy_core.solver import constants
        return getattr(constants, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _LAZY | _LAZY_MARCH | _LAZY_CONSTANTS)


__all__ = [
    "SOLVER_ARG_KINDS", "SOLVER_ARG_MAPPING", "SOLVER_ARG_SLOTS",
    "SOLVER_DECL_KINDS", "solver_arg_kind", "REQUIRED_PROCEDURES",
    "ExternalProcedure", "MissingProcedureBody", "assert_procedure_bodies",
    "required_procedure_gaps", "SOLVER_PREFIX", "Assign", "Call",
    "ForEachFace", "IfStatic", "Procedure", "ProcedureNameError", "Statement",
    "While", "check_procedure_name", "require_resolved", "split_target",
    "written_names",
    "MARCH_FLAGS", "MarchProgram", "TABLEAUS", "TABLEAU_EULER",
    "TABLEAU_SSPRK2", "build_hyperbolic_step", "build_march",
    "build_should_write", "emit_march", "flags_from_nsm", "tableau_for",
    "ConstantResolutionError", "EPS_H_FALLBACK", "MarchConstants",
    "WRITE_EPS", "eigen_wave_speed_floor", "kp_eps", "march_constants",
]

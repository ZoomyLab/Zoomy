"""Solver template: IR, walkers and authoring classes (backend-agnostic).

Layers (docs/dev/solver_template/SPEC.md):
  IR          `.ir` statements + Procedure, `.registry` typed block
              signatures, `.check` scope / kind / arity / liveness
  walkers     `.walker_python` (readable Python source per backend),
              `.walker_c` (C++)
  authoring   `.scheme`, `.stage`, `.plugin`, `.builders`,
              `.solver`
  backends    `.backend_numpy` here; jax in `zoomy_jax.template`

Never imports jax.
"""
from zoomy_core.solver.template.kinds import Port, ports
from zoomy_core.solver.template.expr import (
    AllTrue, AnyTrue, FullLike, IsFinite, MaskLike, ReduceMax, ReduceMin,
    ReduceSum, S, SafeDiv, Where, ZerosLike)
from zoomy_core.solver.template.ir import (
    Assign, Call, Callback, ForEachFace, Guard, Hook, IfStatic, LinSolve,
    LinSolveOpts, Procedure, Repeat, Root, RootOpts, Select, Seq, Statement,
    While, collect_procedures, fill_hooks, resolve_static, static_flags)
from zoomy_core.solver.template.registry import BLOCKS, BlockRegistry, BlockSpec
from zoomy_core.solver.template.consts import Const, ConstTable
from zoomy_core.solver.template.check import (
    ProgramInfo, TemplateCheckError, check_program)
from zoomy_core.solver.template.errors import GuardError
from zoomy_core.solver.template.show import show_procedure, show_program
from zoomy_core.solver.template.walker_python import (
    Generated, LoweringError, MissingBodyError, PythonWalker)
from zoomy_core.solver.template.backend_numpy import NumpyBackend
from zoomy_core.solver.template.scheme import (
    ARS232, ARS343, EULER, SSPRK2, SSPRK3, ButcherTableau, Compose, Imex,
    ImexTableau, ShuOsher, lie, strang)
from zoomy_core.solver.template.stage import (
    BuildContext, CompositeStage, Hyperbolic, Stage)
from zoomy_core.solver.template.plugin import Observer, Output, Plugin
from zoomy_core.solver.template.builders import (
    build_split_step, build_step, build_time_loop)
from zoomy_core.solver.template.solver import CATALOGUE, Setup, Solver


def __getattr__(name):
    # the C walker pulls the generic_c / NSM stack in: load it on demand
    if name in ("CWalker", "CProgram"):
        from zoomy_core.solver.template import walker_c
        return getattr(walker_c, name)
    raise AttributeError(name)

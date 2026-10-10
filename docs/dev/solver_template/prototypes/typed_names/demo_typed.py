"""Feasibility checks for the string-free authoring layer."""
import sys
import numpy as np

from zoomy_core.solver.template.expr import ZerosLike
from zoomy_core.solver.template.ir import Procedure
from zoomy_core.solver.template.show import show_procedure
from zoomy_core.solver.template.stage import (BuildContext, Hyperbolic,
                                              STAGE_INPUTS, STAGE_OUTPUTS)
from zoomy_core.solver.template.registry import BLOCKS
from zoomy_core.solver.template.toy import advection_bodies

from typed import (V, Block, BlockSet, Comm, Flag, HookPoint, assign, body,
                   if_static)


class HyperbolicBlocks(BlockSet):
    halo_bc = Block((V.Q, V.Qaux, V.p, V.time, V.geom), (V.bc_Q, V.bc_aux),
                    comm=Comm.neighbor)
    reconstruct = Block((V.Q, V.Qaux, V.bc_Q, V.bc_aux, V.p, V.geom, V.o1),
                        (V.Q_L, V.Q_R, V.Qaux_L, V.Qaux_R, V.grad))
    flux_pass = Block((V.Q_L, V.Q_R, V.Qaux_L, V.Qaux_R, V.p, V.time, V.geom),
                      (V.F, V.Dp, V.Dm))
    cell_ncp = Block((V.Q, V.grad, V.Qaux, V.p, V.geom), (V.cell_term,))
    gather_update = Block((V.Q, V.F, V.Dp, V.Dm, V.cell_term, V.Qaux, V.p,
                           V.time, V.dt, V.geom), (V.Q_cand, V.troubled))


class Hooks:
    face_equilibrium = HookPoint((V.Q_L, V.Q_R, V.Qaux_L, V.Qaux_R, V.p, V.geom),
                                 (V.Q_L, V.Q_R),
                                 doc="well-balancing (model-supplied), empty by default")
    face_fixup = HookPoint((V.F, V.Dp, V.Dm), (V.F, V.Dp, V.Dm),
                           doc="coupling face fixup")


class Flags:
    cell_ncp = Flag("interior NCP integral (order >= 2)")


B, H = HyperbolicBlocks, Hooks


def hyperbolic_stage():
    with body() as stmts:
        bc_Q, bc_aux = B.halo_bc(V.Q, V.Qaux, V.p, V.time, V.geom,
                                 note="D5/D6 ghost state at the stage time")
        Q_L, Q_R, A_L, A_R, grad = B.reconstruct(V.Q, V.Qaux, bc_Q, bc_aux,
                                                 V.p, V.geom, V.o1)
        Q_L, Q_R = H.face_equilibrium(Q_L, Q_R, A_L, A_R, V.p, V.geom)
        F, Dp, Dm = B.flux_pass(Q_L, Q_R, A_L, A_R, V.p, V.time, V.geom)
        F, Dp, Dm = H.face_fixup(F, Dp, Dm)
        if_static(Flags.cell_ncp,
                  then=lambda: B.cell_ncp(V.Q, grad, V.Qaux, V.p, V.geom),
                  otherwise=lambda: assign(V.cell_term, ZerosLike(V.Q)))
        B.gather_update(V.Q, F, Dp, Dm, V.cell_term, V.Qaux, V.p, V.time,
                        V.dt, V.geom)
    return stmts


# 1. same IR as the current string-based Hyperbolic stage
ctx = BuildContext()
old = Procedure("explicit_stage", STAGE_INPUTS, STAGE_OUTPUTS, Hyperbolic().build(ctx))
new = Procedure("explicit_stage", STAGE_INPUTS, STAGE_OUTPUTS, hyperbolic_stage())
so, sn = show_procedure(old), show_procedure(new)
print("1. IR text identical:", so == sn)
if so != sn:
    import difflib
    print("\n".join(difflib.unified_diff(so.splitlines(), sn.splitlines(), lineterm="")))

# 2. registry derived from the class == the current string registry
same = all(HyperbolicBlocks.registry()[n] == BLOCKS[n].__class__(
    BLOCKS[n].name, BLOCKS[n].inputs, BLOCKS[n].outputs, comm=BLOCKS[n].comm,
    role=BLOCKS[n].role, doc="") for n in HyperbolicBlocks.blocks())
print("2. block signatures identical to registry.py:", same)

# 3. typo in a name -> AttributeError with a suggestion
for expr in ("V.Qax", "HyperbolicBlocks.flux_pas", "Flags.cell_npc"):
    try:
        eval(expr)
    except AttributeError as e:
        import traceback
        msg = "".join(traceback.format_exception_only(e)).strip()
        print(f"3. {expr:28s} -> {msg}")

# 4. wrong kind at a call -> build-time error naming the port
try:
    with body():
        HyperbolicBlocks.flux_pass(V.Q, V.Q_R, V.Qaux_L, V.Qaux_R, V.p, V.time, V.geom)
except TypeError as e:
    print("4.", e)


# 5. bodies as a class: closest match for a misspelt method
class AdvBodies:
    def __init__(self):
        self._b = advection_bodies(np, order=2)

    def halo_bc(self, Q, Qaux, p, time, geom): return self._b["halo_bc"](Q, Qaux, p, time, geom)
    def reconstrut(self, *a): return self._b["reconstruct"](*a)  # typo


try:
    HyperbolicBlocks.bodies(AdvBodies())
except TypeError as e:
    print("5.", e)

# 6. static autocompletion (jedi, what IDEs use) -- no code is run
import jedi
src = open(__file__).read().split("\n# 1.")[0]
for prefix in ("V.Qa", "HyperbolicBlocks.fl", "Hooks.fa", "Flags.c"):
    code = src + "\n" + prefix
    lines = code.splitlines()
    comps = jedi.Script(code, path="x.py").complete(len(lines), len(lines[-1]))
    print(f"6. jedi completes {prefix!r:24s} -> {[c.name for c in comps][:6]}")

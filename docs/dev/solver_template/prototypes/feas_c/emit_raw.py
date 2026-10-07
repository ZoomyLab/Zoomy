"""Step 1: emit the march for SME(0, d=2) at order 1/2 x flag variants, print
with the UNMODIFIED core CProcedurePrinter.  Writes raw/<variant>.inc"""
import os, sys, pathlib, itertools, time
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "raw"; OUT.mkdir(exist_ok=True)

from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, StressFree
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.solver.march import emit_march, MARCH_FLAGS
from zoomy_core.transformation.procedure_c import CProcedurePrinter


def build_nsm(order):
    t0 = time.time()
    model = SME(level=0, dimension=2, closures=[Newtonian(), StressFree()])
    sm = SystemModel.from_model(model)
    kw = {} if order == 1 else dict(reconstruction=ReconstructionSpec(order=2))
    nsm = NumericalSystemModel.from_system_model(sm, **kw).derive()
    print(f"nsm order={order} built in {time.time()-t0:.1f}s; state={list(nsm.state)}",
          file=sys.stderr)
    return nsm

VARIANTS = {
    # name: (order, flag overrides)
    "o1_plain": (1, dict()),
    "o2_plain": (2, dict()),
    "o1_mood": (1, dict(mood=True)),
    "o2_mood": (2, dict(mood=True)),
    "o2_mood_coupled": (2, dict(mood=True, coupled=True)),
    "o2_all": (2, dict(mood=True, coupled=True, implicit_source=True,
                       implicit_diffusion=True, interior_ncp=True)),
    "o1_dtfloor": (1, dict()),   # dt_floor via emit kwarg
}

if __name__ == "__main__":
    nsms = {o: build_nsm(o) for o in (1, 2)}
    for name, (order, fl) in VARIANTS.items():
        nsm = nsms[order]
        flags = None
        kw = dict(cfl=0.9, dimension=1, dt_max=1.0)
        if name == "o1_dtfloor":
            kw["dt_floor"] = 1e-12
        # implicit_* are NSM-derived; force via the flags override
        extra = {k: v for k, v in fl.items()
                 if k in ("implicit_source", "implicit_diffusion")}
        rest = {k: v for k, v in fl.items() if k not in extra}
        prog = emit_march(nsm, flags=extra or None, **kw, **rest)
        pr = CProcedurePrinter()
        # print order: should_write, step, march (callee before caller)
        txt = pr.print_procedures([prog.should_write, prog.step, prog.march])
        (OUT / f"{name}.inc").write_text(
            "// " + prog.report().replace("\n", "\n// ") + "\n" + txt)
        print(name, prog.flags, "tableau", prog.tableau, file=sys.stderr)
    print("MARCH_FLAGS:", sorted(MARCH_FLAGS), file=sys.stderr)

"""Step 3: emit every variant with the FIXED printer subclass.
Writes fixed/<variant>/{march_decls.hpp, march_procs.inc}."""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from emit_raw import build_nsm, VARIANTS
from zoomy_core.solver.march import emit_march
from feas_printer import FeasCProcedurePrinter

# run-time knobs for the dam-break executable (emitted constants!)
RUN_DT_MAX = 2.0 ** -10   # exact binary: t_end = 10*dt_max -> exactly 10 steps


def emit(name, order, fl, nsm):
    kw = dict(cfl=0.9, dimension=1, dt_max=RUN_DT_MAX)
    if name == "o1_dtfloor":
        kw["dt_floor"] = 1e-12
    extra = {k: v for k, v in fl.items() if k in ("implicit_source", "implicit_diffusion")}
    rest = {k: v for k, v in fl.items() if k not in extra}
    prog = emit_march(nsm, flags=extra or None, **kw, **rest)
    procs = [prog.should_write, prog.step, prog.march]
    pr = FeasCProcedurePrinter(procedures=procs)
    body = pr.print_procedures(procs)
    out = HERE / "fixed" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "march_decls.hpp").write_text(
        "// EMITTED (scratch FeasCProcedurePrinter) -- " + name + "\n#pragma once\n"
        + f"constexpr int MARCH_ORDER = {prog.order};\n"
        + pr.workspace_struct() + "\n" + pr.external_declarations())
    (out / "march_procs.inc").write_text(
        "// " + prog.report().replace("\n", "\n// ") + "\n" + body)
    return prog


def model_layout(nsm):
    """NSM-derived index/value header the vector stubs read (no restated layout)."""
    st = [str(s) for s in nsm.state]
    aux = [str(s) for s in nsm.aux_state]
    pv = nsm.parameter_values
    pnames = [str(k) for k in nsm.parameters.keys()] if hasattr(nsm.parameters, "keys") else [str(s) for s in nsm.parameters]
    pvals = [float(getattr(pv, n)) for n in pnames]
    lines = ["// EMITTED from the NSM: state/aux/parameter layout", "#pragma once",
             f"constexpr int N_VARS = {len(st)};", f"constexpr int N_AUX = {len(aux)};",
             f"constexpr int N_PARAMS = {len(pnames)};"]
    lines += [f"constexpr int IDX_{n} = {i};" for i, n in enumerate(st)]
    lines += [f"constexpr int AUX_{n} = {i};" for i, n in enumerate(aux)]
    lines += [f"constexpr int PAR_{n} = {i};" for i, n in enumerate(pnames)]
    lines.append("constexpr double PARAM_VALUES[N_PARAMS] = {" + ", ".join(repr(v) for v in pvals) + "};")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    nsms = {o: build_nsm(o) for o in (1, 2)}
    (HERE / "fixed").mkdir(exist_ok=True)
    (HERE / "fixed" / "model_layout.hpp").write_text(model_layout(nsms[1]))
    for name, (order, fl) in VARIANTS.items():
        emit(name, order, fl, nsms[order])
        print("emitted", name, file=sys.stderr)

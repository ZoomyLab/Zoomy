"""Plugins: fill template hooks and add carried state.

A plugin declares
  * `blocks()`           block / callback specs it calls (registry additions)
  * `define_constants`   named constants it needs
  * `inputs()` / `outputs()`  extra march ports (traced inputs, carried
                           results)
  * `hook_outputs(hook)` names it writes in a hook that are bound before it
  * `fills(ctx)`         `{hook_name: [statements]}`
  * `bodies()`           block bodies it supplies (e.g. observer functions)
  * `defaults()`         default values of its extra march inputs

Hooks of the time loop: `loop_init`, `pre_step`, `observe`, `output`,
`loop_finalize`; of the step: `post_step`; of the explicit stage:
`face_equilibrium`, `face_fixup`.
"""
from __future__ import annotations

from typing import Optional

import sympy as sp

from zoomy_core.solver.template.expr import Where  # noqa: F401  (re-export)
from zoomy_core.solver.template.ir import Assign, Call, Callback, Select
from zoomy_core.solver.template.registry import BlockSpec


class Plugin:
    name = "plugin"

    def blocks(self) -> list:
        return []

    def define_constants(self, consts) -> None:
        return None

    def inputs(self) -> tuple:
        return ()

    def outputs(self) -> tuple:
        return ()

    def hook_outputs(self, hook: str) -> tuple:
        return ()

    def fills(self, ctx) -> dict:
        return {}

    def bodies(self) -> dict:
        return {}

    def defaults(self) -> dict:
        return {}

    def __repr__(self):
        return f"{type(self).__name__}()"


class Output(Plugin):
    """Snapshot output with a drift-free cadence.

    Snapshot `i` is due at the stamp `t_ref + (i - i_ref) * write_interval`
    (recomputed from the index every step, never accumulated); the first step
    reaching it writes it.  The initial state (at `time0`) and the final
    state are written unless disabled.  `write_interval` and the starting
    index `i_snapshot` are TRACED march inputs; with `snapshots=N` the
    interval is `(time_end - time0) / (N - 1)`, computed in-template (so
    `time_end` stays traced).  `write_fields` is an idempotent callback
    keyed by `i_snapshot`."""

    name = "output"

    def __init__(self, snapshots: Optional[int] = None, *,
                 write_initial: bool = True, write_final: bool = True,
                 callback: str = "write_fields"):
        if snapshots is not None and int(snapshots) < 2:
            raise ValueError("Output(snapshots=N) needs N >= 2")
        self.snapshots = None if snapshots is None else int(snapshots)
        self.write_initial = write_initial
        self.write_final = write_final
        self.callback = callback

    def define_constants(self, consts):
        consts.define("c_write_eps", 1e-12, "real",
                      "relative tolerance of the snapshot stamp comparison")
        if self.snapshots is not None:
            consts.define("c_out_intervals", self.snapshots - 1, "int",
                          f"Output(snapshots={self.snapshots})")

    def inputs(self):
        ins = [("i_snapshot", "int")]
        if self.snapshots is None:
            ins.append(("write_interval", "real"))
        return tuple(ins)

    def outputs(self):
        return (("i_snapshot", "int"),)

    def defaults(self):
        return {"i_snapshot": 0}

    def hook_outputs(self, hook):
        if hook == "loop_init":
            return ("i_snapshot",)
        if hook in ("output", "loop_finalize"):
            return ("i_snapshot", "t_last_write")
        return ()

    def _write(self):
        i = sp.Symbol("i_snapshot")
        return [Callback(self.callback, ("Q", "Qaux", "time", "i_snapshot"),
                         kind="idempotent"),
                Assign("i_snapshot", i + 1),
                Assign("t_last_write", sp.Symbol("time"))]

    def fills(self, ctx):
        time, i = sp.symbols("time i_snapshot")
        t_ref, i_ref, wi = sp.symbols("t_ref i_ref write_interval")
        init = [Assign("t_ref", time),
                Assign("i_ref", i if self.write_initial else i - 1),
                Assign("t_last_write", sp.S.NegativeInfinity, kind="real")]
        if self.snapshots is not None:
            init.append(Assign("write_interval",
                               (sp.Symbol("time_end") - time)
                               / sp.Symbol("c_out_intervals")))
        if self.write_initial:
            init += self._write()
        stamp = t_ref + (i - i_ref) * wi
        due = time + sp.Symbol("c_write_eps") * (1 + sp.Abs(stamp)) >= stamp
        out = {"loop_init": [(self.name, init)],
               "output": [(self.name, [Select(due, then=self._write(),
                                              note="drift-free write cadence")])]}
        if self.write_final:
            out["loop_finalize"] = [(self.name, [Select(
                time > sp.Symbol("t_last_write"), then=self._write(),
                note="final snapshot")])]
        return out


class Observer(Plugin):
    """Run accumulator `acc` (any pytree of fixed shapes), carried by the
    loop and differentiable like the state:

        acc = init(Q, Qaux, p)                       before the loop
        acc = update(acc, time, dt, Q, Qaux, p)      after every step

    `observer` is any object with `init` / `update` (e.g.
    `zoomy_jax.fvm.observers.Observer`)."""

    def __init__(self, observer=None, *, init=None, update=None,
                 name: str = "observer", acc: str = "acc"):
        if observer is not None:
            init, update = observer.init, observer.update
        if init is None or update is None:
            raise ValueError("Observer needs init and update functions")
        self.init, self.update = init, update
        self.name = name
        self.acc = acc

    def blocks(self):
        return [
            BlockSpec(f"{self.name}_init", "Q:cell_state Qaux:cell_aux p:params",
                      f"{self.acc}:obs", doc="observer init"),
            BlockSpec(f"{self.name}_update",
                      f"{self.acc}:obs time:real dt:real Q:cell_state "
                      "Qaux:cell_aux p:params", f"{self.acc}:obs",
                      doc="observer update after each step"),
        ]

    def outputs(self):
        return ((self.acc, "obs"),)

    def hook_outputs(self, hook):
        return (self.acc,) if hook == "observe" else ()

    def fills(self, ctx):
        return {
            "loop_init": [(self.name, [Call(f"{self.name}_init",
                                            ("Q", "Qaux", "p"), (self.acc,))])],
            "observe": [(self.name, [Call(f"{self.name}_update",
                                          (self.acc, "time", "dt", "Q", "Qaux",
                                           "p"), (self.acc,))])],
        }

    def bodies(self):
        return {f"{self.name}_init": self.init,
                f"{self.name}_update": self.update}


__all__ = ["Plugin", "Output", "Observer"]

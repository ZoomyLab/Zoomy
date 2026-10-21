"""`Solver`: assemble scheme + stages + plugins into a template, show it,
lower it for a backend, run it.

A solver VARIANT is a catalogue entry (constructor arguments); there is no
capability-checking code -- an impossible combination fails where a walker or
backend has no lowering for a statement it meets.

Presets (`HyperbolicSolver` ...) subclass `Solver` and implement
`Solver.setup_bodies` (mesh + NSM -> block bodies, flags, geometry,
initial state); everything else is generic.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional

from zoomy_core.solver.template.builders import (
    build_split_step, build_step, build_time_loop, define_loop_constants)
from zoomy_core.solver.template.consts import ConstTable
from zoomy_core.solver.template.ir import (
    Procedure, fill_hooks, resolve_static, static_flags)
from zoomy_core.solver.template.registry import BLOCKS
from zoomy_core.solver.template.scheme import Compose, ShuOsher
from zoomy_core.solver.template.show import show_program
from zoomy_core.solver.template.stage import BuildContext, Hyperbolic, Stage
from zoomy_core.solver.template.walker_python import Generated, PythonWalker


def resolve_backend(backend):
    """A backend primitive object, or a name: `numpy` | `jax` /
    `jax-forward` | `jax-reverse` (the jax names import zoomy_jax lazily;
    zoomy_core itself never imports jax)."""
    if backend is None or backend == "numpy":
        from zoomy_core.solver.template.backend_numpy import NumpyBackend
        return NumpyBackend()
    if isinstance(backend, str):
        if backend in ("jax", "jax-forward", "jax_fwd"):
            from zoomy_jax.template.backend import JaxBackend
            return JaxBackend(mode="forward")
        if backend in ("jax-reverse", "jax_rev"):
            from zoomy_jax.template.backend import JaxBackend
            return JaxBackend(mode="reverse")
        raise ValueError(f"unknown backend {backend!r}")
    return backend


@dataclasses.dataclass
class Setup:
    """What a preset's `Solver.setup_bodies` returns."""

    bodies: dict
    flags: dict
    geom: Any
    Q: Any
    Qaux: Any
    parameters: Any
    dimension: int
    degree: int
    order: int
    dt_max: Optional[float] = None
    extra: dict = dataclasses.field(default_factory=dict)


#: Variants = plain catalogue entries (constructor arguments).
CATALOGUE = {
    "hyperbolic": dict(stages=(Hyperbolic,), scheme=None, mood="d1"),
    "hyperbolic_guard_only": dict(stages=(Hyperbolic,), scheme=None,
                                  mood="guard"),
}


class Solver:
    """Template assembly + lowering + run.

    Parameters
    ----------
    scheme    : `ShuOsher` or None (default from the reconstruction
                order: EULER / SSPRK2 / SSPRK3)
    stages    : stage objects (or classes); one RK stage applies them as a
                composite in order.  Default `[Hyperbolic()]`.
    plugins   : `Plugin` objects (Output, Observer, ...)
    backend   : backend primitive object or name (see `resolve_backend`)
    time_end  : default end time (TRACED: changing it does not recompile)
    cfl       : CFL safety factor (emitted constant)
    dt_max    : dt cap (emitted constant; None = no cap)
    mood      : `d1` | `guard` | `off`
    """

    name = "solver"

    def __init__(self, *, scheme: Optional[ShuOsher] = None, stages=None,
                 plugins=(), backend=None, time_end: float = 1.0,
                 cfl: float = 0.9, dt_max: Optional[float] = None,
                 mood: str = "d1", order: Optional[int] = None,
                 registry=None, lowering: str = "auto",
                 cache_dir: Optional[str] = None, name: Optional[str] = None):
        self.scheme = scheme
        stages = list(stages) if stages is not None else self.default_stages()
        self.stages = [s() if isinstance(s, type) else s for s in stages]
        self.plugins = list(plugins)
        self.backend = resolve_backend(backend)
        self.time_end = float(time_end)
        self.cfl = float(cfl)
        self.dt_max = dt_max
        self.mood = mood
        self.order = order
        self.registry = registry or BLOCKS
        self.lowering = lowering
        self.cache_dir = cache_dir
        if name:
            self.name = name
        self._compiled: dict = {}
        self._setup: Optional[Setup] = None
        self.last_outputs: dict = {}

    # -- catalogue ------------------------------------------------------------
    @classmethod
    def variant(cls, name: str, **overrides) -> "Solver":
        kw = dict(CATALOGUE[name])
        kw.update(overrides)
        return cls(**kw)

    def default_stages(self):
        return [Hyperbolic()]

    # -- assembly ---------------------------------------------------------------
    def scheme_for(self, order: Optional[int]) -> ShuOsher:
        if self.scheme is not None:
            return self.scheme
        return ShuOsher.for_order(order or self.order or 2)

    def stage(self) -> Stage:
        if len(self.stages) == 1:
            return self.stages[0]
        from zoomy_core.solver.template.stage import CompositeStage
        return CompositeStage(self.stages)

    def context(self, *, order=None, dimension=None, degree=None,
                dt_max=None) -> BuildContext:
        reg = self.registry.extended([b for pl in self.plugins for b in pl.blocks()])
        ctx = BuildContext(registry=reg, constants=ConstTable(),
                           options={"recon_order": order or self.order or 2})
        dtm = self.dt_max if dt_max is None else dt_max
        define_loop_constants(ctx, cfl=self.cfl, dimension=dimension,
                              degree=degree,
                              dt_max=float("inf") if dtm is None and dimension
                              is not None else dtm)
        for pl in self.plugins:
            pl.define_constants(ctx.constants)
        return ctx

    def template(self, *, order=None, dimension=None, degree=None,
                 dt_max=None):
        """`(march Procedure, BuildContext)`: hooks filled by the plugins,
        `IfStatic` still unresolved."""
        order = order or self.order or 2
        ctx = self.context(order=order, dimension=dimension, degree=degree,
                           dt_max=dt_max)
        scheme = self.scheme_for(order)
        if isinstance(scheme, Compose):
            step = build_split_step(scheme, ctx, mood=self.mood)
        elif isinstance(scheme, ShuOsher):
            step = build_step(scheme, self.stage(), ctx, mood=self.mood)
        else:
            raise NotImplementedError(
                f"no step builder for {type(scheme).__name__} yet (IMEX "
                "lowering is sub-stage 1b)")
        march = build_time_loop(step, ctx, plugins=self.plugins)
        fills: dict = {}
        for pl in self.plugins:
            for hook, parts in pl.fills(ctx).items():
                fills.setdefault(hook, []).extend(parts)
        return fill_hooks(march, fills), ctx

    def show(self, *, flags=None, order=None, file=None, **kw) -> str:
        """Print (and return) the assembled template."""
        march, ctx = self.template(order=order, **kw)
        if flags is not None:
            march = resolve_static(march, flags)
        text = show_program(march, ctx.constants, ctx.registry)
        print(text, file=file)
        return text

    def required_flags(self, order=None) -> tuple:
        return static_flags(self.template(order=order)[0])

    def compile(self, bodies, *, flags, dimension, degree, order=None,
                dt_max=None, backend=None) -> Generated:
        """Lower the template for `backend` (default: the solver's) and
        load it with `bodies`; cached per configuration."""
        B = resolve_backend(backend) if backend is not None else self.backend
        key = (id(B), tuple(sorted(flags.items())), dimension, degree, order,
               dt_max, id(bodies))
        g = self._compiled.get(key)
        if g is None:
            march, ctx = self.template(order=order, dimension=dimension,
                                       degree=degree, dt_max=dt_max)
            table = dict(bodies)
            for pl in self.plugins:
                table.update(pl.bodies())
            g = PythonWalker(B, lowering=self.lowering,
                             cache_dir=self.cache_dir).build(
                march, table, registry=ctx.registry, constants=ctx.constants,
                flags=flags, solver_name=self.name)
            self._compiled[key] = g
        return g

    # -- running ----------------------------------------------------------------
    def setup_bodies(self, mesh, nsm, backend) -> Setup:
        """Preset hook: build the block bodies, build flags, geometry and the
        initial state for `mesh` / `nsm` on `backend`."""
        raise NotImplementedError(
            f"{type(self).__name__} has no setup_bodies(); a preset (e.g. "
            "HyperbolicSolver) supplies the block bodies")

    def setup_simulation(self, mesh, nsm):
        self._setup = self.setup_bodies(mesh, nsm, self.backend)
        return self._setup.Q, self._setup.Qaux

    @property
    def parameters(self):
        return None if self._setup is None else self._setup.parameters

    def _entry(self, aux_from_state: bool):
        s = self._setup
        if s is None:
            raise RuntimeError("call setup_simulation(mesh, nsm) first")
        flags = dict(s.flags)
        flags["aux_from_state"] = bool(aux_from_state)
        return self.compile(s.bodies, flags=flags, dimension=s.dimension,
                            degree=s.degree, order=s.order, dt_max=s.dt_max)

    def _call(self, Q, Qaux, parameters, time0, time_end, extra):
        s = self._setup
        g = self._entry(Qaux is None)
        p = s.parameters if parameters is None else parameters
        root = g.info.root
        values = dict(Q=Q, Qaux=s.Qaux if Qaux is None else Qaux, p=p,
                      geom=s.geom, time0=time0,
                      time_end=self.time_end if time_end is None else time_end)
        for pl in self.plugins:
            values.update({k: v for k, v in pl.defaults().items()
                           if k not in values})
        values.update(extra)
        args = [values[pt.name] for pt in root.inputs]
        out = g.entry(*args)
        names = root.output_names
        out = out if len(names) > 1 else (out,)
        self.last_outputs = dict(zip(names, out))
        return self.last_outputs

    def run_simulation(self, Q, Qaux=None, parameters=None, time0=0.0,
                       time_end=None, **plugin_inputs):
        """March from `time0` to `time_end` (default: the solver's);
        `Qaux=None` rebuilds the aux from `Q` inside the march.  Returns
        `(Q, Qaux)`; all outputs are in `last_outputs`."""
        o = self._call(Q, Qaux, parameters, time0, time_end, plugin_inputs)
        return o["Q"], o["Qaux"]

    def run_observed(self, Q, Qaux=None, parameters=None, time0=0.0,
                     time_end=None, acc: str = "acc", **plugin_inputs):
        o = self._call(Q, Qaux, parameters, time0, time_end, plugin_inputs)
        return o["Q"], o["Qaux"], o[acc]

    def solve(self, mesh, nsm, **kw):
        Q, Qaux = self.setup_simulation(mesh, nsm)
        return self.run_simulation(Q, Qaux, **kw)


__all__ = ["Solver", "Setup", "CATALOGUE", "resolve_backend"]

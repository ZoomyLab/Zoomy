"""Stages: what one RK stage does, as template statements.

Every stage is a :class:`~.ir.Procedure` with the common signature

    (Q, Qaux, p, geom, time, dt, o1) -> (Q_cand, troubled)

``time`` is the stage time ``t + c_k dt`` (D3), ``dt`` the stage step,
``o1`` the MOOD order-1 cell mask, ``Q_cand`` the stage candidate and
``troubled`` its not-admissible / non-finite cell mask.  The step builder
calls it once per Shu-Osher stage (and again inside the MOOD ``Select``).

Stages only REFERENCE blocks by name; the bodies are per backend.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import sympy as sp

from zoomy_core.solver.template.consts import ConstTable
from zoomy_core.solver.template.expr import MaskLike, ZerosLike
from zoomy_core.solver.template.ir import (
    Assign, Call, Hook, IfStatic, Procedure)
from zoomy_core.solver.template.registry import BLOCKS, BlockRegistry

STAGE_INPUTS = ("Q:cell_state Qaux:cell_aux p:params geom:geom time:real "
                "dt:real o1:cell_mask")
STAGE_OUTPUTS = "Q_cand:cell_state troubled:cell_mask"


@dataclasses.dataclass
class BuildContext:
    """What builders may read: the block registry, the constant table (they
    may define constants), the build options and an id source."""

    registry: BlockRegistry = dataclasses.field(default_factory=lambda: BLOCKS)
    constants: ConstTable = dataclasses.field(default_factory=ConstTable)
    options: dict = dataclasses.field(default_factory=dict)
    _ids: dict = dataclasses.field(default_factory=dict)

    def unique(self, stem: str) -> str:
        n = self._ids.get(stem, 0)
        self._ids[stem] = n + 1
        return stem if n == 0 else f"{stem}_{n}"


class Stage:
    """Base class: subclasses implement :meth:`build`.

    ``supports_o1``: the stage's result depends on the ``o1`` mask (it has a
    higher-order reconstruction that MOOD can demote).  ``name``: procedure
    name."""

    name = "stage"
    supports_o1 = False
    doc = ""

    def build(self, ctx: BuildContext) -> list:
        raise NotImplementedError

    def procedure(self, ctx: BuildContext) -> Procedure:
        key = ("stage_proc", id(self))
        if key not in ctx._ids:
            ctx._ids[key] = Procedure(self.name, STAGE_INPUTS, STAGE_OUTPUTS,
                                      self.build(ctx), doc=self.doc)
        return ctx._ids[key]

    def __repr__(self):
        return f"{type(self).__name__}()"


class Hyperbolic(Stage):
    """Explicit FV stage: ``Q_cand = Q + dt L(Q; t_stage)``.

    halo/BC ghosts -> reconstruction (o1-aware) -> face_equilibrium hook ->
    numerical flux + fluctuations -> face_fixup hook (coupling) -> interior
    NCP (build flag ``cell_ncp``) -> gather with the admissibility mask."""

    name = "explicit_stage"
    supports_o1 = True
    doc = ("Explicit hyperbolic stage: Q_cand = Q + dt*L(Q; time), troubled = "
           "not admissible or non-finite.")

    def build(self, ctx):
        Q = sp.Symbol("Q")
        return [
            Call("halo_bc", ("Q", "Qaux", "p", "time", "geom"),
                 ("bc_Q", "bc_aux"), note="D5/D6 ghost state at the stage time"),
            Call("reconstruct",
                 ("Q", "Qaux", "bc_Q", "bc_aux", "p", "geom", "o1"),
                 ("Q_L", "Q_R", "Qaux_L", "Qaux_R", "grad")),
            Hook("face_equilibrium",
                 inputs=("Q_L", "Q_R", "Qaux_L", "Qaux_R", "p", "geom"),
                 outputs=("Q_L", "Q_R"),
                 note="well-balancing (model-supplied), empty by default"),
            Call("flux_pass",
                 ("Q_L", "Q_R", "Qaux_L", "Qaux_R", "p", "time", "geom"),
                 ("F", "Dp", "Dm")),
            Hook("face_fixup", inputs=("F", "Dp", "Dm"),
                 outputs=("F", "Dp", "Dm"), note="coupling face fixup"),
            IfStatic("cell_ncp",
                     then=[Call("cell_ncp", ("Q", "grad", "Qaux", "p", "geom"),
                                ("cell_term",))],
                     otherwise=[Assign("cell_term", ZerosLike(Q))]),
            Call("gather_update",
                 ("Q", "F", "Dp", "Dm", "cell_term", "Qaux", "p", "time", "dt",
                  "geom"),
                 ("Q_cand", "troubled")),
        ]


class CompositeStage(Stage):
    """Sub-stages chained within ONE RK stage: ``Q -> S_1 -> S_2 -> ...``
    (D8: Chorin's [hyperbolic(Euler) -> elliptic -> pointwise]).  The troubled
    masks are OR-combined; ``refresh_aux`` refreshes the aux between
    sub-stages at the stage time."""

    def __init__(self, stages, *, name: Optional[str] = None,
                 refresh_aux: bool = True):
        self.stages = tuple(stages)
        if not self.stages:
            raise ValueError("CompositeStage needs at least one stage")
        self.name = name or "composite_" + "_".join(s.name for s in self.stages)
        self.refresh_aux = refresh_aux
        self.supports_o1 = any(s.supports_o1 for s in self.stages)
        self.doc = "Composite stage: " + " -> ".join(s.name for s in self.stages)

    def build(self, ctx):
        Qs, tr, tri = sp.symbols("Q_s troubled troubled_i")
        body = [Assign("Q_s", sp.Symbol("Q")),
                Assign("troubled", MaskLike(sp.Symbol("Q"), False))]
        for i, st in enumerate(self.stages):
            if i > 0 and self.refresh_aux:
                body.append(Call("update_aux", ("Q_s", "Qaux", "p", "geom", "time"),
                                 ("Qaux",), note="aux of the sub-stage input"))
            body.append(Call(st.procedure(ctx),
                             ("Q_s", "Qaux", "p", "geom", "time", "dt", "o1"),
                             ("Q_s", "troubled_i")))
            body.append(Assign("troubled", sp.Or(tr, tri)))
        body.append(Assign("Q_cand", Qs))
        return body

    def __repr__(self):
        return f"CompositeStage({list(self.stages)})"


__all__ = ["Stage", "Hyperbolic", "CompositeStage", "BuildContext",
           "STAGE_INPUTS", "STAGE_OUTPUTS"]

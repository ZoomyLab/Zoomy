"""The typed block registry: the SINGLE source of block signatures.

A *block* is an opaque operation the template calls by name; its body is
supplied per backend (one shared xp body for numpy and jax, native code for
C++).  Its :class:`BlockSpec` states:

* ``inputs`` / ``outputs`` -- typed :class:`~.kinds.Port` lists.  A port name
  that appears in BOTH lists is an in/out port (passed once, by mutable
  reference, in C);
* ``comm`` -- ``none`` (pointwise / local), ``neighbor`` (reads halo / ghost
  data), ``global`` (collective reduction).  The type checker's
  replicated-predicate rule reads it;
* ``role`` -- ``block`` (called with ``Call``) or ``callback`` (called with
  ``Callback``; a host side effect);
* ``doc`` -- the contract a backend body implements.

Every ``Call`` / ``Callback`` is checked against its spec at build time
(arity and kind).  Body contract (Python backends): a body is called with the
input values in port order and returns its outputs -- the bare value for one
output, a tuple for several, ``None`` for none.
"""
from __future__ import annotations

import dataclasses

from zoomy_core.solver.template.kinds import ports

COMM_CLASSES = ("none", "neighbor", "global")
ROLES = ("block", "callback")


class RegistryError(KeyError):
    pass


@dataclasses.dataclass(frozen=True)
class BlockSpec:
    name: str
    inputs: tuple = ()
    outputs: tuple = ()
    comm: str = "none"
    role: str = "block"
    doc: str = ""

    def __post_init__(self):
        if not self.name.isidentifier():
            raise ValueError(f"block name must be an identifier: {self.name!r}")
        object.__setattr__(self, "inputs", ports(self.inputs))
        object.__setattr__(self, "outputs", ports(self.outputs))
        if self.comm not in COMM_CLASSES:
            raise ValueError(f"comm must be one of {COMM_CLASSES}")
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        for p in self.outputs:
            q = next((q for q in self.inputs if q.name == p.name), None)
            if q is not None and q.kind != p.kind:
                raise ValueError(f"block {self.name}: in/out port {p.name!r} "
                                 f"has kinds {q.kind!r} vs {p.kind!r}")

    @property
    def input_names(self):
        return tuple(p.name for p in self.inputs)

    @property
    def output_names(self):
        return tuple(p.name for p in self.outputs)

    def signature(self) -> str:
        ins = ", ".join(str(p) for p in self.inputs)
        outs = ", ".join(str(p) for p in self.outputs)
        return f"{self.name}({ins}) -> ({outs})  [{self.role}, comm={self.comm}]"


class BlockRegistry:
    """Name -> :class:`BlockSpec`.  ``extended(...)`` returns a child registry
    (parent specs visible, additions local) so plugins and tests never mutate
    the shared default."""

    def __init__(self, specs=(), parent=None):
        self._specs: dict = {}
        self._parent = parent
        for s in specs:
            self.register(s)

    def register(self, spec: BlockSpec, *, replace: bool = False) -> BlockSpec:
        if not isinstance(spec, BlockSpec):
            raise TypeError(f"not a BlockSpec: {spec!r}")
        old = self.get(spec.name, None)
        if old is not None and old != spec and not replace:
            raise RegistryError(
                f"block {spec.name!r} is already registered with a different "
                f"signature:\n  have {old.signature()}\n  new  {spec.signature()}")
        self._specs[spec.name] = spec
        return spec

    def get(self, name, default=...):
        if name in self._specs:
            return self._specs[name]
        if self._parent is not None:
            return self._parent.get(name, default)
        if default is ...:
            raise RegistryError(f"no block {name!r} in the registry")
        return default

    def __getitem__(self, name):
        return self.get(name)

    def __contains__(self, name):
        return self.get(name, None) is not None

    def names(self) -> tuple:
        out = list(self._parent.names()) if self._parent is not None else []
        for n in self._specs:
            if n not in out:
                out.append(n)
        return tuple(out)

    def extended(self, specs=()) -> "BlockRegistry":
        return BlockRegistry(specs, parent=self)

    def report(self) -> str:
        return "\n".join(self.get(n).signature() for n in self.names())


def _spec(name, inputs, outputs, comm="none", doc="", role="block"):
    return BlockSpec(name, inputs, outputs, comm=comm, role=role, doc=doc)


#: The template's default blocks.  Signature order follows the template:
#: ``(state..., Qaux, p, ..., time, dt, geom, ...)``.
BLOCKS = BlockRegistry([
    _spec("prepare_mesh", "mesh:any", "topology:any geom:geom", doc=(
        "Host setup: mesh object -> (topology, geom).  topology is static "
        "integer connectivity (numpy, closed over by the bodies); geom is the "
        "vectorised xp geometry pytree computed from node coordinates "
        "(differentiable w.r.t. them).")),
    _spec("initialize_state", "geom:geom p:params", "Q:cell_state Qaux:cell_aux",
          doc="Initial state from the model's initial conditions."),
    _spec("compute_dt",
          "Q:cell_state Qaux:cell_aux p:params geom:geom cfl:real "
          "dimension:int degree_factor:int dt_max:real",
          "dt:real", comm="global", doc=(
              "D4: dt = min_f cfl*2 r_f / (dimension*degree_factor*|lambda_f|) "
              "over all faces (local per-face law, wave-free faces drop out), "
              "capped by dt_max.  Divisions are safe (double where).  Global "
              "reduction (AD-safe reduce_min).")),
    _spec("halo_bc", "Q:cell_state Qaux:cell_aux p:params time:real geom:geom",
          "bc_Q:bc_state bc_aux:bc_aux", comm="neighbor", doc=(
              "Batched BC kernel over all boundary faces at the stage time: "
              "GHOST values (D5: Dirichlet 2v - Q_in, wall reflection, "
              "extrapolation Q_in, ...), routed by tag NAME; ghost aux "
              "recomputed from the ghost state (D6).  Halo exchange under "
              "domain decomposition.")),
    _spec("reconstruct",
          "Q:cell_state Qaux:cell_aux bc_Q:bc_state bc_aux:bc_aux p:params "
          "geom:geom o1:cell_mask",
          "Q_L:face_state Q_R:face_state Qaux_L:face_aux Qaux_R:face_aux "
          "grad:cell_grad", doc=(
              "Face states: constant or LSQ-MUSCL with the NSM limiter, in the "
              "model's reconstruction variables (W(Q)/Q(W) maps), positivity "
              "scaling over the admissibility list.  Cells flagged in o1 are "
              "reconstructed piecewise constant (MOOD order-1 repair).  grad "
              "is the limited gradient (zero at o1 cells / order 1).")),
    _spec("flux_pass",
          "Q_L:face_state Q_R:face_state Qaux_L:face_aux Qaux_R:face_aux "
          "p:params time:real geom:geom",
          "F:face_state Dp:face_state Dm:face_state", doc=(
              "Batched numerical flux and path-conservative fluctuations, each "
              "face evaluated once.")),
    _spec("cell_ncp", "Q:cell_state grad:cell_grad Qaux:cell_aux p:params geom:geom",
          "cell_term:cell_state", doc=(
              "Cell-interior non-conservative integral (Gauss-Legendre form of "
              "numerical_cell_edge_source); order >= 2 only.")),
    _spec("gather_update",
          "Q:cell_state F:face_state Dp:face_state Dm:face_state "
          "cell_term:cell_state Qaux:cell_aux p:params time:real dt:real "
          "geom:geom",
          "Q_cand:cell_state troubled:cell_mask", doc=(
              "Q_cand = Q + dt*(-div(F + D) + cell_term + explicit source / "
              "diffusion) via scatter_add; troubled = not admissible "
              "(NSM admissible() < 0) or non-finite, per cell.")),
    _spec("update_aux", "Q:cell_state Qaux:cell_aux p:params geom:geom time:real",
          "Qaux:cell_aux", comm="neighbor", doc=(
              "Full-length aux contract: algebraic aux map then LSQ derivative "
              "aux rows; time/position threaded.")),
    _spec("update_variables",
          "Q:cell_state Qaux:cell_aux p:params geom:geom time:real dt:real",
          "Q:cell_state", doc=(
              "The model's per-cell update_variables slot (identity when the "
              "model declares none).")),
    _spec("write_fields", "Q:cell_state Qaux:cell_aux time:real i_snapshot:int", "",
          role="callback", doc=(
              "Idempotent snapshot write keyed by i_snapshot (an INPUT; the "
              "body never re-derives it from time).")),
])


__all__ = ["BlockSpec", "BlockRegistry", "BLOCKS", "COMM_CLASSES", "ROLES",
           "RegistryError"]

"""Codegen driver for the SWE browser game.

This is the app-side entry point of the Zoomy codegen pipeline: it
**(a) creates the model** — a Shallow Water `Model` with the game's
parameter set and boundary conditions — and **(b) prints it to code**,
running zoomy_core's JS and GLSL printers to emit the kernels the
``library/zoomy_js`` solver consumes.

Run from the ``zoomy`` conda env::

    python apps/swe-game/generate.py

Writes ``apps/swe-game/generated/swe2d.js`` and ``…/swe2d.glsl``
(plus a small ``swe2d_meta.json`` describing the parameter layout and
boundary-tag indices the runtime adapter needs).

The Q(t) inflow curve is *not* baked in: the model carries a fixed
20-point timeline as parameters (``q_t_0…q_t_19`` / ``q_v_0…q_v_19``),
the BC interpolates over them symbolically, and the game writes the
live editor curve into that parameter block at runtime.
"""

import json
import re
from pathlib import Path

import numpy as np
from sympy import Matrix, sqrt

from zoomy_core.misc.misc import ZArray
from zoomy_core.model.basemodel import Model
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions,
    Extrapolation,
    InflowOutflow,
    Wall,
    _sympy_interpolate_data,
)
from zoomy_core.fvm.riemann_solvers import HLLC
from zoomy_core.transformation.to_js import JsModel, JsNumerics
from zoomy_core.transformation.to_glsl import GlslModel, GlslNumerics

# Number of (time, value) pairs in the inflow timeline parameter block.
# The game resamples its live Q(t) editor curve to exactly this many
# points (duplicating points when the user has fewer).
N_TIMELINE = 20

# Physical defaults (mirror the current hand-written game worker).
G_DEFAULT = 9.81
H_IN_DEFAULT = 0.1     # target inflow depth
Q_IN_DEFAULT = 0.01    # base inflow discharge (scaled by the Q(t) curve)
END_TIME = 60.0        # the timeline spans [0, END_TIME] by default


class SWEGameModel(Model):
    """2D Shallow Water Equations for the irrigation game.

    State ``Q = [h, hu, hv]``; conservative form, flat bottom (no
    bathymetry, hence no well-balancing).  Parameters: ``g``, the
    inflow constants ``h_in`` / ``q_in``, and a fixed ``N_TIMELINE``
    -point Q(t) timeline block (``q_t_*`` / ``q_v_*``).
    """

    def __init__(self, **kwargs):
        params = {
            "g": (G_DEFAULT, "positive"),
            "h_in": (H_IN_DEFAULT, "positive"),
            "q_in": (Q_IN_DEFAULT, "positive"),
        }
        # Auto-generated timeline block — never hand-referenced by name;
        # the runtime adapter writes it as two contiguous slices of `p`
        # (all q_t_* then all q_v_*).
        for i in range(N_TIMELINE):
            params[f"q_t_{i}"] = (END_TIME * i / (N_TIMELINE - 1), "positive")
        for i in range(N_TIMELINE):
            params[f"q_v_{i}"] = (1.0, "positive")
        super().__init__(
            dimension=2,
            variables=["h", "hu", "hv"],
            parameters=params,
            eigenvalue_mode="symbolic",
            **kwargs,
        )

    def flux(self):
        h = self.variables[0]
        hu = Matrix(self.variables[1:])
        u = hu / h
        g = self._parameter_symbols.g
        F = Matrix.zeros(self.n_variables, self.dimension)
        F[0, :] = hu.T
        F[1:, :] = h * u * u.T + 0.5 * g * h**2 * Matrix.eye(self.dimension)
        return ZArray(F)

    def eigenvalues(self):
        h = self.variables[0]
        hu = Matrix(self.variables[1:])
        n = Matrix(self.normal[:2])
        g = self._parameter_symbols.g
        un = (hu.T * n)[0] / h
        c = sqrt(g * h)
        return ZArray([un - c, un + c, un])


def _q_scale_expr(model):
    """Symbolic piecewise-linear Q(t) multiplier interpolated over the
    model's timeline parameter block."""
    ps = model._parameter_symbols
    t = model.time
    q_t = np.array([ps[f"q_t_{i}"] for i in range(N_TIMELINE)])
    q_v = np.array([ps[f"q_v_{i}"] for i in range(N_TIMELINE)])
    return _sympy_interpolate_data(t, q_t, q_v)


def build_model():
    """Create the SWE game Model with its boundary-condition set."""
    base = SWEGameModel()
    ps = base._parameter_symbols
    q_scale = _q_scale_expr(base)

    # inflow: soft depth blend toward `h_in`, momentum prescribed
    # (OpenFOAM inletOutlet style) to `q_in * q_scale`.
    inflow = InflowOutflow(
        tag="inflow",
        prescribe_fields={
            0: {"mode": "blend", "target": ps.h_in, "weight": q_scale},
            1: {"mode": "inlet_outlet", "value": ps.q_in * q_scale},
        },
    )
    bcs = BoundaryConditions([
        Wall(tag="wall", momentum_field_indices=[[1, 2]]),
        Extrapolation(tag="outflow"),     # zero-Neumann
        inflow,
    ])
    return SWEGameModel(boundary_conditions=bcs)


def _exported(code):
    """Append an ESM ``export { … }`` of every generated function."""
    names = sorted(set(re.findall(r"^function (\w+)\(", code, re.M)))
    return code + "\n\nexport { " + ", ".join(names) + " };\n"


def main():
    model = build_model()
    numerics = HLLC(model)
    out_dir = Path(__file__).parent / "generated"
    out_dir.mkdir(exist_ok=True)

    js = "\n\n".join([
        "// AUTO-GENERATED by apps/swe-game/generate.py — do not edit.",
        JsModel(model).generate(),
        JsNumerics(numerics).generate(),
        JsModel(model).generate_boundary_conditions(),
    ])
    (out_dir / "swe2d.js").write_text(_exported(js))

    glsl = "\n\n".join([
        "// AUTO-GENERATED by apps/swe-game/generate.py — do not edit.",
        GlslModel(model).generate(),
        GlslNumerics(numerics).generate(),
        GlslModel(model).generate_boundary_conditions(),
    ])
    (out_dir / "swe2d.glsl").write_text(glsl)

    # Metadata the runtime adapter needs: ordered parameter names (so
    # the timeline block is a known contiguous slice of `p`) and the
    # boundary-tag → index map (BoundaryConditions sorts tags).
    param_names = list(model._parameter_symbols.keys())
    meta = {
        "nVars": model.n_variables,
        "nAux": model.n_aux_variables,
        "dimension": model.dimension,
        "parameterNames": param_names,
        "parameterDefaults": [float(v) for v in model.parameters.values()],
        "timelineBlock": {
            "nPoints": N_TIMELINE,
            "timeStart": param_names.index("q_t_0"),
            "valueStart": param_names.index("q_v_0"),
        },
        "boundaryTags": list(
            model.boundary_conditions.list_sorted_function_names
        ),
        "numerics": numerics.__class__.__name__,
    }
    (out_dir / "swe2d_meta.json").write_text(json.dumps(meta, indent=2))

    # The WebGL engine lives in index.html's classic <script>, which
    # cannot `import`.  Emit the GLSL kernels + metadata as a global
    # the page can pull in with a plain <script src> tag.
    glsl_js = (
        "// AUTO-GENERATED by apps/swe-game/generate.py — do not edit.\n"
        "// GLSL kernels + metadata for the WebGL engine.\n"
        "window.SWE2D = {\n"
        f"  meta: {json.dumps(meta)},\n"
        f"  glsl: {json.dumps(glsl)},\n"
        "};\n"
    )
    (out_dir / "swe2d_glsl.js").write_text(glsl_js)

    print(f"wrote {out_dir}/swe2d.js, swe2d.glsl, swe2d_meta.json, swe2d_glsl.js")
    print(f"  boundary tags: {meta['boundaryTags']}")
    print(f"  parameters: {len(param_names)} "
          f"(timeline block at {meta['timelineBlock']})")


if __name__ == "__main__":
    main()

"""Module `zoomy_core.model.derivative_workflow`."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import ast
import inspect
import textwrap
import numpy as np
import param
import sympy as sp

from zoomy_core.fvm.solver_numpy import HyperbolicSolver
from zoomy_core.mesh.lsq_reconstruction import compute_derivatives
from zoomy_core.model.basemodel import Model
from zoomy_core.misc.misc import ZArray, Zstruct


@dataclass(frozen=True)
class DerivativeSpec:
    """DerivativeSpec. (class)."""
    field: str
    axes: Tuple[str, ...]

    @property
    def key(self) -> Tuple[str, Tuple[str, ...]]:
        """Key."""
        return (self.field, self.axes)

    def aux_name(self) -> str:
        """Aux name."""
        axis_code = "".join(self.axes)
        return f"d_{axis_code}_{self.field}"


class DerivativeNamespace:
    """DerivativeNamespace. (class)."""
    def __init__(self, model: "StructuredDerivativeModel"):
        """Initialize the instance."""
        self._model = model

    def diff(self, field, axes: Iterable[str]):
        """Diff."""
        axes_tuple = tuple(axes)
        field_name = self._model.resolve_field_name(field)
        key = (field_name, axes_tuple)
        if key not in self._model.derivative_key_to_symbol:
            raise KeyError(
                f"Derivative {key} was not declared in requested_derivatives()."
            )
        return self._model.derivative_key_to_symbol[key]

    def dt(self, field):
        """Dt."""
        return self.diff(field, ("t",))

    def dx(self, field):
        """Dx."""
        return self.diff(field, ("x",))

    def dxx(self, field):
        """Dxx."""
        return self.diff(field, ("x", "x"))

    def dtxx(self, field):
        """Dtxx."""
        return self.diff(field, ("t", "x", "x"))


class StructuredDerivativeModel(Model):
    """Compatibility layer for named fields + derivative mapping."""

    user_aux_variables = param.Parameter(default=0)
    auto_requested_derivatives = param.Boolean(default=False)
    derivative_canonicalization = param.Selector(
        default="time_first",
        objects=["none", "time_first"],
    )
    aux_variables = lambda self: self.derivative_buffer_aux_names()

    def requested_derivatives(self) -> List[DerivativeSpec]:
        """Requested derivatives."""
        return []

    def _canonicalize_axes(self, axes: Tuple[str, ...]) -> Tuple[str, ...]:
        """Internal helper `_canonicalize_axes`."""
        mode = self.derivative_canonicalization
        if mode == "none":
            return axes
        if mode == "time_first":
            t_count = sum(a == "t" for a in axes)
            rest = tuple(a for a in axes if a != "t")
            return tuple(["t"] * t_count) + rest
        return axes

    def _canonicalize_specs(self, specs: List[DerivativeSpec]) -> List[DerivativeSpec]:
        """Internal helper `_canonicalize_specs`."""
        dedup: Dict[Tuple[str, Tuple[str, ...]], DerivativeSpec] = {}
        for spec in specs:
            axes = self._canonicalize_axes(spec.axes)
            key = (spec.field, axes)
            dedup[key] = DerivativeSpec(field=spec.field, axes=axes)
        return list(dedup.values())

    def _infer_requested_derivatives(self) -> List[DerivativeSpec]:
        """Infer derivative specs from calls like self.D.dx(self.Q.h) in model methods."""
        methods = []
        for method_name in ("flux", "source", "nonconservative_matrix"):
            fn = getattr(type(self), method_name, None)
            if fn is not None:
                methods.append(fn)

        specs: List[DerivativeSpec] = []

        def _axes_from_helper(name: str):
            """Internal helper `_axes_from_helper`."""
            if not name.startswith("d"):
                return None
            if name == "diff":
                return "diff"
            axes_chars = name[1:]
            if not axes_chars:
                return None
            allowed = {"t", "x", "y", "z"}
            if all(ch in allowed for ch in axes_chars):
                return tuple(axes_chars)
            return None

        for method in methods:
            try:
                source = inspect.getsource(method)
            except (OSError, TypeError):
                continue
            source = textwrap.dedent(source)
            tree = ast.parse(source)
            alias_to_field: Dict[str, str] = {}
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                if len(node.targets) != 1:
                    continue
                target = node.targets[0]
                value = node.value
                if not isinstance(target, ast.Name):
                    continue
                if (
                    isinstance(value, ast.Attribute)
                    and isinstance(value.value, ast.Attribute)
                    and isinstance(value.value.value, ast.Name)
                    and value.value.value.id == "self"
                    and value.value.attr == "Q"
                ):
                    alias_to_field[target.id] = value.attr
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not isinstance(func, ast.Attribute):
                    continue
                d_obj = func.value
                if not (
                    isinstance(d_obj, ast.Attribute)
                    and d_obj.attr == "D"
                    and isinstance(d_obj.value, ast.Name)
                    and d_obj.value.id == "self"
                ):
                    continue

                op = func.attr
                axes_info = _axes_from_helper(op)
                if axes_info is None:
                    continue
                if not node.args:
                    continue

                # Expect first arg like self.Q.<field>
                field_arg = node.args[0]
                field_name = None
                if (
                    isinstance(field_arg, ast.Attribute)
                    and isinstance(field_arg.value, ast.Attribute)
                    and isinstance(field_arg.value.value, ast.Name)
                    and field_arg.value.value.id == "self"
                    and field_arg.value.attr == "Q"
                ):
                    field_name = field_arg.attr
                elif isinstance(field_arg, ast.Name) and field_arg.id in alias_to_field:
                    field_name = alias_to_field[field_arg.id]
                elif isinstance(field_arg, ast.Constant) and isinstance(field_arg.value, str):
                    field_name = field_arg.value

                if field_name is None:
                    continue

                if axes_info == "diff":
                    if len(node.args) < 2:
                        continue
                    axes_arg = node.args[1]
                    if isinstance(axes_arg, (ast.Tuple, ast.List)):
                        axes = []
                        ok = True
                        for elt in axes_arg.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                axes.append(elt.value)
                            else:
                                ok = False
                                break
                        if not ok:
                            continue
                        specs.append(DerivativeSpec(field=field_name, axes=tuple(axes)))
                else:
                    specs.append(DerivativeSpec(field=field_name, axes=axes_info))

        return specs

    def _user_aux_names(self) -> List[str]:
        """Internal helper `_user_aux_names`."""
        definition = self._resolve_input(self.user_aux_variables)
        if isinstance(definition, int):
            return [f"aux_{i}" for i in range(definition)]
        if isinstance(definition, (list, tuple)):
            return [str(v) for v in definition]
        if isinstance(definition, dict):
            return [str(k) for k in definition.keys()]
        if isinstance(definition, Zstruct):
            return [str(k) for k in definition.keys()]
        raise TypeError(
            "user_aux_variables must be int/list/tuple/dict/Zstruct compatible."
        )

    def derivative_buffer_aux_names(self) -> List[str]:
        """Derivative buffer aux names."""
        user_names = self._user_aux_names()
        specs = list(self.requested_derivatives())
        if self.auto_requested_derivatives:
            specs.extend(self._infer_requested_derivatives())
        specs = self._canonicalize_specs(specs)
        deriv_names = [spec.aux_name() for spec in specs]
        return user_names + deriv_names

    def _initialize_derived_properties(self):
        """Internal helper `_initialize_derived_properties`."""
        super()._initialize_derived_properties()
        self.Q = self.variables
        self.A = self.aux_variables
        self.params = self.parameters
        self._symbol_to_field: Dict[sp.Symbol, str] = {
            symbol: name for name, symbol in zip(self.variables.keys(), self.variables.values())
        }
        specs = list(self.requested_derivatives())
        if self.auto_requested_derivatives:
            specs.extend(self._infer_requested_derivatives())
        specs = self._canonicalize_specs(specs)
        self.derivative_specs = specs
        self.n_user_aux_variables = len(self._user_aux_names())
        self.derivative_key_to_index: Dict[Tuple[str, Tuple[str, ...]], int] = {}
        self.derivative_key_to_symbol: Dict[Tuple[str, Tuple[str, ...]], sp.Symbol] = {}
        for idx, spec in enumerate(specs):
            key = spec.key
            i_aux = self.n_user_aux_variables + idx
            self.derivative_key_to_index[key] = i_aux
            self.derivative_key_to_symbol[key] = self.aux_variables[i_aux]
        self.D = DerivativeNamespace(self)

    def resolve_field_name(self, field) -> str:
        """Resolve field name."""
        if isinstance(field, str):
            return field
        if field in self._symbol_to_field:
            return self._symbol_to_field[field]
        raise KeyError(f"Unknown field reference '{field}'.")


class DerivativeAwareSolverMixin:
    """Computes declared derivative buffer entries and fills Qaux."""

    def update_qaux(self, Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt):
        """Run the canonical aux legs, then overlay any `derivative_specs`.

        COMPOSE, don't shadow.  `super().update_qaux` resolves (via the MRO
        of every concrete user — IMEXSolver, DerivativeAwareSolver — past this
        mixin to `Solver.update_qaux`) and fills the two canonical legs: the
        local `update_aux_variables` formula (e.g. KP `hinv`) and the
        non-local `aux_registry` spatial-derivative rows.  Canonical
        `aux_registry` models (SME/VAM/…) carry no `derivative_specs`, so
        they get *exactly* the explicit `HyperbolicSolver` aux path —
        previously this method early-returned `Qaux` unchanged and silently
        froze their aux at the IC.  On top, any model that DOES declare
        `derivative_specs` (StructuredDerivativeModel / Green-Naghdi) gets
        its declared buffer rows — including the `(Q-Qold)/dt` time
        derivative, the one capability the canonical leg lacks — overlaid here.
        """
        out = super().update_qaux(
            Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt)
        symbolic_model = model.model if hasattr(model, "model") else model
        specs = getattr(symbolic_model, "derivative_specs", None)
        if not specs:
            return out

        out = np.array(out, copy=True)
        var_keys = symbolic_model.variables.keys()
        field_to_index = {name: i for i, name in enumerate(var_keys)}
        for spec in specs:
            i_aux = symbolic_model.derivative_key_to_index[spec.key]
            i_q = field_to_index[spec.field]
            out[i_aux, :] = self._compute_derivative(spec.axes, Q[i_q], Qold[i_q], mesh, dt)
        return out

    def _compute_derivative(self, axes, q_now, q_old, mesh, dt):
        """Internal helper `_compute_derivative`."""
        n_t = sum(a == "t" for a in axes)
        n_x = sum(a == "x" for a in axes)
        if len(axes) != (n_t + n_x):
            raise NotImplementedError("Only axes in {'t', 'x'} are supported in this demo.")
        if n_t > 1:
            raise NotImplementedError("Only first-order time derivatives are supported.")

        data = q_now
        if n_t == 1:
            data = (q_now - q_old) / max(float(dt), 1e-14)

        if n_x == 0:
            return data

        deriv = compute_derivatives(
            data,
            mesh,
            derivatives_multi_index=[[n_x]],
            u_boundary_face="extrapolation",
        )[:, 0]
        return deriv


class DerivativeAwareSolver(DerivativeAwareSolverMixin, HyperbolicSolver):
    """Generic derivative-aware solver based on the NumPy hyperbolic base."""


class DerivativeAwareHyperbolicSolver(DerivativeAwareSolver):
    """Backward-compatible alias."""


def print_model_functions(model: Model, function_names=None):
    """Compatibility wrapper; prefer model.print_model_functions()."""
    return model.print_model_functions(function_names=function_names)

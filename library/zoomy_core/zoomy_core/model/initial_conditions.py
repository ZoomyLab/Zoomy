"""Initial conditions (constant, Riemann problems, radial dam break, user function, 3-D
projection, restart) applied to the state array of a Model.
"""

import numpy as np
import param
import sympy as sp
from typing import Callable, Optional

from zoomy_core.misc.custom_types import FArray
from zoomy_core.misc.misc import ZArray
from zoomy_core.mesh.lsq_mesh import LSQMesh as Mesh
import zoomy_core.misc.io as io
import zoomy_core.misc.interpolation as interpolate_mesh

# --- Helper Functions for Defaults ---


def default_constant_func(n_variables: int) -> FArray:
    """Default constant state `[1, 0, 0, ...]`."""
    return np.array([1.0] + [0.0 for i in range(n_variables - 1)])


def default_low_state(n_variables: int) -> FArray:
    """Default low (right) state `[1, 0, ...]`."""
    return np.array([1.0 * (i == 0) for i in range(n_variables)])


def default_high_state(n_variables: int) -> FArray:
    """Default high (left) state `[2, 0, ...]`."""
    return np.array([2.0 * (i == 0) for i in range(n_variables)])


def default_user_function(x: FArray) -> FArray:
    """Default user function, identically zero."""
    return 0.0


# --- Classes ---


class InitialConditions(param.Parameterized):
    """Abstract base of initial conditions; subclasses implement `apply` and
    `get_definition`.
    """
    def apply(self, X, Q):
        """Abstract: fill `Q` from `X`."""
        assert False, "InitialConditions is an abstract class."
        return Q

    def get_definition(self, X, p, n_variables):
        """Symbolic initial state for C++ code generation; the default is a zero state that the
        solver overwrites at runtime.
        """
        # Default fallback: Return 0. Allows compilation even if IC is non-symbolic.
        # The solver is expected to overwrite this at runtime if needed.
        return ZArray.zeros(n_variables)


class Constant(InitialConditions):
    """Spatially constant initial state given by `constants(n_variables)`."""
    constants = param.Callable(default=default_constant_func)

    def __init__(self, constants=None, **params):
        """Set `constants` (callable `n_variables -> state`) if given."""
        if constants is not None:
            params["constants"] = constants
        super().__init__(**params)

    def apply(self, X, Q):
        """Fill `Q` (shape `(n_variables, n_cells)`) in place from the cell coordinates
        `X` (shape `(dim, n_cells)`).

        Returns:
            The filled `Q`.
        """
        n_variables = Q.shape[0]
        const_vals = self.constants(n_variables)
        for i in range(Q.shape[1]):
            Q[:, i] = const_vals
        return Q

    def get_definition(self, X, p, n_variables):
        """Symbolic initial state `ZArray` of length `n_variables` for code generation."""
        vals = self.constants(n_variables)
        return ZArray(vals)


class RP(InitialConditions):
    """1-D Riemann problem: `high` state for `x < jump_position_x`, `low` state otherwise."""
    low = param.Callable(default=default_low_state)
    high = param.Callable(default=default_high_state)
    jump_position_x = param.Number(default=0.0)

    def apply(self, X, Q):
        """Fill `Q` (shape `(n_variables, n_cells)`) in place from the cell coordinates
        `X` (shape `(dim, n_cells)`).

        Returns:
            The filled `Q`.
        """
        assert X.shape[1] == Q.shape[1]
        n_variables = Q.shape[0]
        val_low = self.low(n_variables)
        val_high = self.high(n_variables)
        for i in range(Q.shape[1]):
            if X[0, i] < self.jump_position_x:
                Q[:, i] = val_high
            else:
                Q[:, i] = val_low
        return Q

    def get_definition(self, X, p, n_variables):
        """Symbolic initial state `ZArray` of length `n_variables` for code generation."""
        val_low = self.low(n_variables)
        val_high = self.high(n_variables)

        # Symbolic Condition
        cond = X[0] < self.jump_position_x

        out = []
        for i in range(n_variables):
            out.append(sp.Piecewise((val_high[i], cond), (val_low[i], True)))
        return ZArray(out)


class RP2d(InitialConditions):
    """2-D Riemann problem: `high` state in the quadrant `x < jump_position_x` and
    `y < jump_position_y`, `low` otherwise.
    """
    low = param.Callable(default=default_low_state)
    high = param.Callable(default=default_high_state)
    jump_position_x = param.Number(default=0.0)
    jump_position_y = param.Number(default=0.0)

    def apply(self, X, Q):
        """Fill `Q` (shape `(n_variables, n_cells)`) in place from the cell coordinates
        `X` (shape `(dim, n_cells)`).

        Returns:
            The filled `Q`.
        """
        assert X.shape[1] == Q.shape[1]
        n_variables = Q.shape[0]
        val_low = self.low(n_variables)
        val_high = self.high(n_variables)
        for i in range(Q.shape[1]):
            if X[0, i] < self.jump_position_x and X[1, i] < self.jump_position_y:
                Q[:, i] = val_high
            else:
                Q[:, i] = val_low
        return Q

    def get_definition(self, X, p, n_variables):
        """Symbolic initial state `ZArray` of length `n_variables` for code generation."""
        val_low = self.low(n_variables)
        val_high = self.high(n_variables)
        cond = sp.And(X[0] < self.jump_position_x, X[1] < self.jump_position_y)
        out = []
        for i in range(n_variables):
            out.append(sp.Piecewise((val_high[i], cond), (val_low[i], True)))
        return ZArray(out)


class RP3d(InitialConditions):
    """3-D Riemann problem: `high` state in the octant below all three jump positions, `low`
    otherwise.
    """
    low = param.Callable(default=default_low_state)
    high = param.Callable(default=default_high_state)
    jump_position_x = param.Number(default=0.0)
    jump_position_y = param.Number(default=0.0)
    jump_position_z = param.Number(default=0.0)

    def apply(self, X, Q):
        """Fill `Q` (shape `(n_variables, n_cells)`) in place from the cell coordinates
        `X` (shape `(dim, n_cells)`).

        Returns:
            The filled `Q`.
        """
        assert X.shape[1] == Q.shape[1]
        n_variables = Q.shape[0]
        val_low = self.low(n_variables)
        val_high = self.high(n_variables)
        for i in range(Q.shape[1]):
            if (
                X[0, i] < self.jump_position_x
                and X[1, i] < self.jump_position_y
                and X[2, i] < self.jump_position_z
            ):
                Q[:, i] = val_high
            else:
                Q[:, i] = val_low
        return Q

    def get_definition(self, X, p, n_variables):
        """Symbolic initial state `ZArray` of length `n_variables` for code generation."""
        val_low = self.low(n_variables)
        val_high = self.high(n_variables)
        cond = sp.And(
            X[0] < self.jump_position_x,
            X[1] < self.jump_position_y,
            X[2] < self.jump_position_z,
        )
        out = []
        for i in range(n_variables):
            out.append(sp.Piecewise((val_high[i], cond), (val_low[i], True)))
        return ZArray(out)


class RadialDambreak(InitialConditions):
    """Radial dam break: `high` state within `radius` of the mesh centroid, `low` outside."""
    low = param.Callable(default=default_low_state)
    high = param.Callable(default=default_high_state)
    radius = param.Number(default=0.1)

    def apply(self, X, Q):
        """Fill `Q` (shape `(n_variables, n_cells)`) in place from the cell coordinates
        `X` (shape `(dim, n_cells)`).

        Returns:
            The filled `Q`.
        """
        dim = X.shape[0]
        center = np.zeros(dim)
        for d in range(dim):
            center[d] = X[d, :].mean()
        assert X.shape[1] == Q.shape[1]
        n_variables = Q.shape[0]
        val_low = self.low(n_variables)
        val_high = self.high(n_variables)
        for i in range(Q.shape[1]):
            if np.linalg.norm(X[:, i] - center) <= self.radius:
                Q[:, i] = val_high
            else:
                Q[:, i] = val_low
        return Q

    def get_definition(self, X, p, n_variables):
        # NOTE: For C++ generation, we assume center is at (0,0,0) or needs explicit params.
        # This implementation assumes origin for simplicity.
        """Symbolic initial state, with the circle centred at the origin (unlike `apply`,
        which uses the centroid).
        """
        dist_sq = X[0] ** 2 + X[1] ** 2 + X[2] ** 2
        cond = dist_sq <= self.radius**2

        val_low = self.low(n_variables)
        val_high = self.high(n_variables)

        out = []
        for i in range(n_variables):
            out.append(sp.Piecewise((val_high[i], cond), (val_low[i], True)))
        return ZArray(out)


class UserFunction(InitialConditions):
    """Initial state from a user callable `function(x) -> state` evaluated per cell."""
    function = param.Callable(default=None)

    def __init__(self, function=None, **params):
        """Set `function` if given."""
        if function is not None:
            params["function"] = function
        super().__init__(**params)

    def apply(self, X, Q):
        """Fill `Q` (shape `(n_variables, n_cells)`) in place from the cell coordinates
        `X` (shape `(dim, n_cells)`).

        Returns:
            The filled `Q`.
        """
        assert X.shape[1] == Q.shape[1]
        func_to_use = self.function
        if func_to_use is None:
            func_to_use = lambda x: np.zeros(Q.shape[0])
        for i, x in enumerate(X.T):
            Q[:, i] = func_to_use(x)
        return Q

    def get_definition(self, X, p, n_variables):
        """Symbolic initial state obtained by calling `function` on symbolic `X`; a zero
        state if it is not symbolic.
        """
        if self.function is None:
            return ZArray.zeros(n_variables)
        try:
            # Try passing symbolic X
            res = self.function(X)
            return ZArray(res)
        except Exception:
            # Fallback for non-symbolic functions (e.g. restarts or complex logic)
            # We return 0, expecting the C++ solver to OVERWRITE this via file IO.
            return ZArray.zeros(n_variables)


class Project3D(InitialConditions):
    """Initial condition from a 3-D profile through the model's own `project_from_3d` map,
    sampled at its projection nodes. Covers the horizontal-velocity moments (SME family); the
    profile is given as `height(x)`, optional `bed(x)` and `velocity(x, zeta)`.

    Args:
        model: the Model providing `project_from_3d`.
        height: callable `x -> h`.
        velocity: callable `(x, zeta) -> u`, the vertical velocity profile.
        bed: callable `x -> b`; defaults to a flat bed at zero.
    """

    def __init__(self, model, height, velocity, bed=None, **params):
        """Store the model and the profile callables."""
        super().__init__(**params)
        self._model = model
        self._height = height
        self._velocity = velocity
        self._bed = bed if bed is not None else (lambda x: 0.0)
        self._rows = None
        self._nodes = None

    def _build(self):
        """Lambdify the projection rows and collect the `P3_u` node positions."""
        import sympy as sp
        pf = self._model.project_from_3d
        p3u = sorted({a for e in pf for a in sp.sympify(e).atoms(sp.Function)
                      if a.func.__name__ == "P3_u"},
                     key=lambda a: float(a.args[0]))
        self._nodes = [float(a.args[0]) for a in p3u]
        args = [sp.Symbol("P3_b"), sp.Symbol("P3_h")] + list(p3u)
        self._rows = [sp.lambdify(args, sp.sympify(pf[i]), "numpy")
                      for i in range(len(pf))]

    def apply(self, X, Q):
        """Fill `Q` (shape `(n_variables, n_cells)`) in place from the cell coordinates
        `X` (shape `(dim, n_cells)`).

        Returns:
            The filled `Q`.
        """
        if self._rows is None:
            self._build()
        assert X.shape[1] == Q.shape[1]
        for i, x in enumerate(X.T):
            b, h = float(self._bed(x)), float(self._height(x))
            u = [float(self._velocity(x, z)) for z in self._nodes]
            Q[:, i] = [float(np.squeeze(r(b, h, *u))) for r in self._rows]
        return Q


class RestartFromHdf5(InitialConditions):
    # Parameters omitted for brevity, functionality preserved
    """Restart from an HDF5 snapshot; a runtime operation handled by the solver, so `apply` is
    a no-op.
    """
    path_to_fields = param.String(default=None)
    mesh_new = param.ClassSelector(class_=Mesh, default=None)
    mesh_identical = param.Boolean(default=False)
    path_to_old_mesh = param.String(default=None)
    snapshot = param.Integer(default=-1)
    map_fields = param.Dict(default=None)

    def apply(self, X, Q):
        # (Standard Python implementation)
        """No-op placeholder; the solver performs the HDF5 load."""
        pass

    def get_definition(self, X, p, n_variables):
        # Restart is a RUNTIME operation.
        # Generate a dummy zero state for the compiled model.
        # The C++ VirtualSolver must handle the actual HDF5 loading.
        """Zero state for the compiled model; the solver overwrites it from the HDF5 file."""
        return ZArray.zeros(n_variables)

"""Load a trained 1D Poisson V-cycle checkpoint and build GMRES x0 for GN-style stacked Q.

GN `Q` is shaped `(n_vars, n_cells)` with C-order ravel **variable-major**
(`flat = Q.ravel()` is all cells for var 0, then var 1, …).  The V-cycle uses **cell-major**
stacking `[cell0: v0,v1,…, cell1: …]`.  We convert, run `forward_vcycle(b_p)` on **inner**
cells only (columns `0 .. n_inner-1`), embed the result into a full-length vector with **zeros
on ghost columns**, and scale by `guess_scale`.

The learned operator is still the **synthetic** $L\\otimes I$ Poisson proxy; using GMRES RHS
as its `f` is a **heuristic** initial guess for the GN Newton Jacobian, not an exact solve.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from zoomy_jax.gnn_blueprint.mg_structured_hierarchy import restrict_field_to_coarser_levels
from zoomy_jax.gnn_blueprint.mg_structured_hierarchy_1d import (
    build_poisson_hierarchy_1d,
    build_poisson_hierarchy_1d_vector,
)
from zoomy_jax.gnn_blueprint.vcycle_structured_gnn import forward_vcycle


def inner_var_major_to_poisson_flat(v_inner: np.ndarray, n_inner: int, n_vars: int) -> np.ndarray:
    g = np.asarray(v_inner, dtype=np.float64).reshape(n_vars, n_inner)
    return np.swapaxes(g, 0, 1).reshape(-1)


def poisson_flat_to_inner_var_major(p: np.ndarray, n_inner: int, n_vars: int) -> np.ndarray:
    arr = np.asarray(p, dtype=np.float64).reshape(n_inner, n_vars)
    return arr.T.reshape(-1)


@dataclass
class VcycleImexContext:
    """JIT’d V-cycle + hierarchy; `guess_x0` maps full GMRES `b` and current `Q`."""

    n_interior: int
    n_components: int
    use_bump: bool
    bump_from_bathymetry: bool
    params: Any
    a_list: tuple[jnp.ndarray, ...]
    r_list: tuple[jnp.ndarray, ...]
    p_list: tuple[jnp.ndarray, ...]
    edges_list: tuple[jnp.ndarray, ...]
    static_b_list: tuple[jnp.ndarray, ...]
    r_scalar_np: list[np.ndarray]
    n_mp: int
    hid: int
    nu1: int
    nu2: int
    coarsest_iters: int
    coarsest_omega: float
    _jit_forward: Any

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, mesh_n_inner: int) -> VcycleImexContext:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"vcycle checkpoint not found: {p}")
        with open(p, "rb") as f:
            bundle: dict[str, Any] = pickle.load(f)
        n_in = int(bundle["n_interior"])
        if n_in != int(mesh_n_inner):
            raise ValueError(
                f"checkpoint n_interior={n_in} != mesh n_inner_cells={mesh_n_inner}; "
                "retrain with matching --n-interior."
            )
        d = int(bundle["n_components"])
        use_bump = bool(bundle.get("use_bump", False))
        bump_from_bathymetry = bool(bundle.get("bump_from_bathymetry", True))
        params_np = bundle["params"]
        params = jax.tree.map(lambda a: jnp.asarray(a, dtype=jnp.float64), params_np)

        a_list_np, r_list_np, p_list_np, edges_list_np, _ = build_poisson_hierarchy_1d_vector(
            n_in, d
        )
        _, r_scalar_np, _, _, _ = build_poisson_hierarchy_1d(n_in)

        a_list = tuple(jnp.asarray(x, dtype=jnp.float64) for x in a_list_np)
        r_list = tuple(jnp.asarray(x, dtype=jnp.float64) for x in r_list_np)
        p_list = tuple(jnp.asarray(x, dtype=jnp.float64) for x in p_list_np)
        edges_list = tuple(jnp.asarray(e, dtype=jnp.int32) for e in edges_list_np)

        n_levels = len(a_list_np)
        if use_bump and not bump_from_bathymetry:
            bl = bundle.get("static_b_per_level")
            if bl is None:
                raise ValueError("checkpoint use_bump without bathymetry needs static_b_per_level")
            static_b_list = tuple(jnp.asarray(x, dtype=jnp.float64) for x in bl)
        else:
            static_b_list = tuple(
                jnp.zeros(a_list_np[i].shape[0] // d, dtype=jnp.float64) for i in range(n_levels)
            )

        n_mp = int(bundle.get("n_mp_layers", 3))
        hid = int(bundle.get("hidden", 32))
        nu1 = int(bundle.get("nu1", 1))
        nu2 = int(bundle.get("nu2", 1))
        coarsest_iters = int(bundle.get("coarsest_iters", 40))
        coarsest_omega = float(bundle.get("coarsest_omega", 0.5))

        def _one(
            fvec: jnp.ndarray,
            p: Any,
            bl: tuple[jnp.ndarray, ...],
        ) -> jnp.ndarray:
            return forward_vcycle(
                fvec,
                p,
                a_list,
                r_list,
                p_list,
                edges_list,
                bl,
                n_mp,
                hid,
                nu1,
                nu2,
                coarsest_iters,
                coarsest_omega,
                d,
            )

        jit_forward = jax.jit(_one)

        return cls(
            n_interior=n_in,
            n_components=d,
            use_bump=use_bump,
            bump_from_bathymetry=bump_from_bathymetry,
            params=params,
            a_list=a_list,
            r_list=r_list,
            p_list=p_list,
            edges_list=edges_list,
            static_b_list=static_b_list,
            r_scalar_np=list(r_scalar_np),
            n_mp=n_mp,
            hid=hid,
            nu1=nu1,
            nu2=nu2,
            coarsest_iters=coarsest_iters,
            coarsest_omega=coarsest_omega,
            _jit_forward=jit_forward,
        )

    def _b_list_for_q(self, Q_np: np.ndarray) -> tuple[jnp.ndarray, ...]:
        n_levels = len(self.static_b_list)
        if not self.use_bump:
            return self.static_b_list
        if self.bump_from_bathymetry:
            bf = np.asarray(Q_np[0, : self.n_interior], dtype=np.float64)
            b_per = restrict_field_to_coarser_levels(bf, self.r_scalar_np)
            return tuple(jnp.asarray(x, dtype=jnp.float64) for x in b_per)
        return self.static_b_list

    def guess_x0(
        self,
        b_flat: np.ndarray,
        Q_np: np.ndarray,
        q_shape: tuple[int, ...],
        guess_scale: float,
    ) -> np.ndarray:
        n_vars, n_tot = int(q_shape[0]), int(q_shape[1])
        if n_vars != self.n_components:
            raise ValueError(
                f"Q has n_vars={n_vars} but checkpoint n_components={self.n_components}"
            )
        if n_tot < self.n_interior:
            raise ValueError("mesh too small for checkpoint n_interior")
        b3 = np.asarray(b_flat, dtype=np.float64).reshape(q_shape)
        Q3 = np.asarray(Q_np, dtype=np.float64).reshape(q_shape)
        b_inner = b3[:, : self.n_interior].reshape(-1)
        b_p = inner_var_major_to_poisson_flat(b_inner, self.n_interior, n_vars)
        bl = self._b_list_for_q(Q3)
        x0_p = np.asarray(
            self._jit_forward(jnp.asarray(b_p, dtype=jnp.float64), self.params, bl),
            dtype=np.float64,
        )
        x0_inner = poisson_flat_to_inner_var_major(x0_p, self.n_interior, n_vars)
        x0_full = np.zeros_like(b_flat, dtype=np.float64)
        shp = (n_vars, n_tot)
        x0_full.reshape(shp)[:, : self.n_interior] = x0_inner.reshape(n_vars, self.n_interior)
        return float(guess_scale) * x0_full


def save_vcycle_checkpoint(
    path: str | Path,
    *,
    params: Any,
    n_interior: int,
    n_components: int,
    use_bump: bool,
    bump_from_bathymetry: bool = True,
    static_b_per_level: list[np.ndarray] | None = None,
    n_mp_layers: int = 3,
    hidden: int = 32,
    nu1: int = 1,
    nu2: int = 1,
    coarsest_iters: int = 40,
    coarsest_omega: float = 0.5,
) -> None:
    """Pickle training hyperparameters and `params` as NumPy arrays (host tree)."""
    params_np = jax.tree.map(lambda a: np.asarray(a), params)
    bundle = {
        "params": params_np,
        "n_interior": int(n_interior),
        "n_components": int(n_components),
        "use_bump": bool(use_bump),
        "bump_from_bathymetry": bool(bump_from_bathymetry),
        "static_b_per_level": static_b_per_level,
        "n_mp_layers": int(n_mp_layers),
        "hidden": int(hidden),
        "nu1": int(nu1),
        "nu2": int(nu2),
        "coarsest_iters": int(coarsest_iters),
        "coarsest_omega": float(coarsest_omega),
    }
    outp = Path(path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

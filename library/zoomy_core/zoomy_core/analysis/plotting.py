"""Plotting helpers for `zoomy_core.analysis`.

Lightweight wrappers around matplotlib.  Lazy-imported so the package
stays usable without matplotlib installed.

Two main entry points:

  `plot_dispersion(result, k_range, ...)` —
       line plot of phase velocity `C(k) = ω/k` (or ω(k)) for each
       propagating mode in a `plane_wave_dispersion` result.  Accepts
       reference curves (e.g. Airy `tanh(kH)/(kH)`) for overlay
       comparison.

  `plot_hyperbolic_region_2d(M_x, M_t, axis_a, axis_b, fixed_subs, ...)` —
       2D color map of the hyperbolic region over a chosen parameter
       pair, with all other parameters held at user-specified values.
       Mirrors the Koellermeier–Torrilhon-style figures used to
       characterise the loss-of-hyperbolicity region in the SME family.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import sympy as sp


def _pyplot():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:                            # pragma: no cover
        raise ImportError(
            "matplotlib is required for plotting; install via "
            "`pip install matplotlib`"
        ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_lambdify(symbols, expr, modules='numpy'):
    """Lambdify with vectorised numeric error suppression."""
    f = sp.lambdify(symbols, expr, modules=modules)

    def safe(*args):
        with np.errstate(divide='ignore', invalid='ignore'):
            try:
                v = f(*args)
            except (ZeroDivisionError, FloatingPointError):
                return np.nan
            return v
    return safe


def _detect_k_symbol(expr) -> sp.Symbol:
    candidates = [s for s in expr.free_symbols if str(s) == 'k']
    if not candidates:
        raise ValueError(
            "Could not auto-detect a wavenumber symbol named 'k' in the "
            "result; pass k_var=Symbol('k') explicitly."
        )
    return candidates[0]


# ---------------------------------------------------------------------------
# 1D dispersion plot
# ---------------------------------------------------------------------------

def plot_dispersion(result: Dict, k_range: Tuple[float, float], *,
                    k_var: Optional[sp.Symbol] = None,
                    ax=None,
                    n_points: int = 200,
                    mode: str = 'phase_velocity',
                    fixed_subs: Optional[Dict] = None,
                    references: Optional[Dict[str, Callable]] = None,
                    nondimensionalise_by: Optional[Tuple[str, Any]] = None,
                    squared: bool = False,
                    title: Optional[str] = None,
                    mode_labels: Optional[List[str]] = None,
                    drop_zero_modes: bool = True,
                    **plot_kwargs):
    """Plot dispersion curves from a `plane_wave_dispersion` result.

    Args:
        result:    dict from `plane_wave_dispersion`.
        k_range:   `(k_lo, k_hi)` numeric range over which to evaluate.
        k_var:     wavenumber symbol; auto-detected as `Symbol('k')`
                   if absent.
        ax:        matplotlib axis to draw on; new figure created if None.
        n_points:  grid resolution along k.
        mode:      `'phase_velocity'` (C = ω/k) or `'omega'`.
        fixed_subs: dict `{Symbol: number}` for the non-k free symbols.
        references: dict `{label: callable(k_arr)}` for overlay curves
                    (e.g. `{'Airy': lambda k: np.sqrt(np.tanh(k*H)/(k*H))}`).
        nondimensionalise_by: tuple `(label, expr)`; if given, the y-axis
                              becomes `y / expr` and the label includes
                              `label` (e.g. `('gH', g*H)`).
        squared:   if True, plot `y**2` instead of `y` (for `C²/(gH)`).
        title:     optional plot title.
        mode_labels: if given, list of length `len(solutions)` for the
                     legend; otherwise `mode 0, mode 1, …`.
        drop_zero_modes: if True (default), skip ω-solutions that are
                         identically zero (trivial constraint modes).

    Returns:
        `{'k', 'curves', 'figure', 'axis'}`.
    """
    plt = _pyplot()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    fixed_subs = fixed_subs or {}
    references = references or {}

    if k_var is None:
        k_var = _detect_k_symbol(result['determinant'])

    sols = list(result.get('solutions') or [])
    k_arr = np.linspace(k_range[0], k_range[1], n_points)
    curves = []

    # Optional non-dimensionalisation factor.
    if nondimensionalise_by is not None:
        norm_label, norm_expr = nondimensionalise_by
        norm_val = float(sp.sympify(norm_expr).xreplace(fixed_subs))
    else:
        norm_label, norm_val = None, 1.0

    for i, sol in enumerate(sols):
        if drop_zero_modes and sp.simplify(sol) == 0:
            continue
        sol_sub = sol.xreplace(fixed_subs)
        if mode == 'phase_velocity':
            y_expr = sol_sub / k_var
        elif mode == 'omega':
            y_expr = sol_sub
        else:
            raise ValueError(f"mode must be 'phase_velocity' or 'omega'; got {mode!r}")
        if squared:
            y_expr = y_expr ** 2
        f = _safe_lambdify((k_var,), y_expr)
        y = np.empty(n_points, dtype=complex)
        for j, kv in enumerate(k_arr):
            try:
                y[j] = complex(f(kv))
            except Exception:
                y[j] = np.nan
        y_real = np.real(y)
        y_imag = np.imag(y)
        ok = np.isfinite(y_real) & (np.abs(y_imag) < 1e-9)
        if not np.any(ok):
            continue
        if norm_val != 1.0:
            y_real = y_real / norm_val
        label = (mode_labels[i] if mode_labels and i < len(mode_labels)
                 else f'mode {i}')
        ax.plot(k_arr[ok], y_real[ok], label=label, **plot_kwargs)
        curves.append((k_arr[ok], y_real[ok]))

    for ref_label, ref_fn in references.items():
        y_ref = ref_fn(k_arr)
        ax.plot(k_arr, y_ref, '--', label=ref_label, alpha=0.7, linewidth=1.5)

    ax.set_xlabel(str(k_var))
    if mode == 'phase_velocity':
        ylabel = r'$C^2 = (\omega/k)^2$' if squared else r'$C = \omega / k$'
    else:
        ylabel = r'$\omega^2$' if squared else r'$\omega$'
    if norm_label:
        ylabel = ylabel + f' / {norm_label}'
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return {'k': k_arr, 'curves': curves, 'figure': fig, 'axis': ax}


# ---------------------------------------------------------------------------
# 2D hyperbolicity region scan
# ---------------------------------------------------------------------------

def plot_hyperbolic_region_2d(M_x, M_t,
                              axis_a: Tuple[sp.Symbol, float, float],
                              axis_b: Tuple[sp.Symbol, float, float],
                              fixed_subs: Dict,
                              *, ax=None,
                              n_a: int = 100, n_b: int = 100,
                              tol: float = 1e-9,
                              drop_infinite: bool = True,
                              show: str = 'binary',
                              title: Optional[str] = None,
                              cmap: Optional[str] = None,
                              ):
    """2D color map of hyperbolicity over a chosen parameter pair.

    Each `(a_val, b_val)` grid point is evaluated by:

      1. Substituting all of `fixed_subs` plus ``{axis_a[0]: a_val,
         axis_b[0]: b_val}`` into `M_x` and `M_t`.
      2. Computing the generalised eigenvalues of the pencil
         `(M_x_num, M_t_num)` via `scipy.linalg.eig`.
      3. Filtering infinite eigenvalues (constraint modes); the state
         is **hyperbolic** iff every remaining eigenvalue has
         `|imag| < tol`.

    Args:
        M_x, M_t: pencil matrices (sympy).
        axis_a:   `(symbol_a, lo, hi)` for the x-axis parameter.
        axis_b:   `(symbol_b, lo, hi)` for the y-axis parameter.
        fixed_subs: dict of values for every other free symbol in
                    `M_x` and `M_t`.
        n_a, n_b: grid resolution.
        show:     `'binary'` (default — green=hyperbolic, red=non) or
                  `'imag_max'` (color by `max |Im λ|`).
        cmap:     optional matplotlib colormap name.

    Returns:
        dict with `a_grid`, `b_grid`, `is_hyperbolic` (bool 2D
        array), `imag_max` (float 2D array), `figure`, `axis`.
    """
    plt = _pyplot()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    sym_a, a_lo, a_hi = axis_a
    sym_b, b_lo, b_hi = axis_b
    a_arr = np.linspace(a_lo, a_hi, n_a)
    b_arr = np.linspace(b_lo, b_hi, n_b)

    M_x_pre = M_x.subs(fixed_subs)
    M_t_pre = M_t.subs(fixed_subs)
    f_Mx = sp.lambdify([sym_a, sym_b], M_x_pre, modules='numpy')
    f_Mt = sp.lambdify([sym_a, sym_b], M_t_pre, modules='numpy')

    try:
        from scipy.linalg import eig as scipy_eig
    except ImportError:                                # pragma: no cover
        scipy_eig = None

    is_hyp = np.zeros((n_b, n_a), dtype=bool)
    imag_max = np.full((n_b, n_a), np.nan, dtype=float)

    for ib, b_val in enumerate(b_arr):
        for ia, a_val in enumerate(a_arr):
            try:
                A_num = np.array(f_Mx(a_val, b_val), dtype=complex)
                B_num = np.array(f_Mt(a_val, b_val), dtype=complex)
                if scipy_eig is not None:
                    lam = scipy_eig(A_num, B_num, left=False, right=False)
                else:
                    lam = np.linalg.eigvals(
                        np.linalg.solve(B_num + 1e-12 * np.eye(B_num.shape[0]),
                                        A_num)
                    )
                if drop_infinite:
                    lam = lam[np.isfinite(lam)]
                if len(lam) == 0:
                    is_hyp[ib, ia] = True       # vacuous: no propagating mode
                    imag_max[ib, ia] = 0.0
                else:
                    im = float(np.max(np.abs(np.imag(lam))))
                    is_hyp[ib, ia] = (im < tol)
                    imag_max[ib, ia] = im
            except Exception:
                is_hyp[ib, ia] = False
                imag_max[ib, ia] = np.nan

    if show == 'binary':
        from matplotlib.colors import ListedColormap
        cm = ListedColormap(['lightcoral', 'lightgreen'])
        im = ax.pcolormesh(a_arr, b_arr, is_hyp.astype(int),
                           shading='auto', cmap=cm, vmin=0, vmax=1)
        cbar = fig.colorbar(im, ax=ax, ticks=[0.25, 0.75])
        cbar.set_ticklabels(['non-hyperbolic', 'hyperbolic'])
    elif show == 'imag_max':
        cm = cmap or 'viridis'
        im = ax.pcolormesh(a_arr, b_arr, imag_max,
                           shading='auto', cmap=cm)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r'$\max |\Im(\lambda)|$')
    else:
        raise ValueError(f"show must be 'binary' or 'imag_max'; got {show!r}")

    # Overlay the hyperbolic / non-hyperbolic boundary as a contour line
    # when in binary mode.
    if show == 'binary':
        # Use imag_max == tol as the implicit boundary.
        try:
            ax.contour(a_arr, b_arr, imag_max, levels=[tol],
                       colors='k', linewidths=1.0, linestyles='-')
        except Exception:
            pass

    ax.set_xlabel(str(sym_a))
    ax.set_ylabel(str(sym_b))
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3, color='k', linewidth=0.5)
    return {
        'a_grid': a_arr,
        'b_grid': b_arr,
        'is_hyperbolic': is_hyp,
        'imag_max': imag_max,
        'figure': fig,
        'axis': ax,
    }

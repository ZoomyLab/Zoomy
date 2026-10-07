"""Observers: quantities accumulated over a whole run of the time loop.

An :class:`Observer` is a pair of pure functions,

    init(Q, Qaux, parameters) -> acc
    update(acc, time, dt, Q, Qaux, parameters) -> acc

``acc`` is any pytree of fixed shapes and dtypes.
:meth:`HyperbolicSolver.run_observed` evaluates ``init`` on the initial state
and ``update`` after every step, on the post-step state at the new ``time``
(``dt`` is the step just taken).  ``acc`` rides in the loop state, so it is
differentiable like the final state: ``jax.jvp`` with ``ad_mode="forward"``,
``jax.grad`` / ``jax.vjp`` with ``ad_mode="reverse"``.

The observer is part of the compiled-loop cache key.  :class:`Observer` is a
NamedTuple, so two observers are the same key iff they hold the same two
function objects: reuse ONE observer object across runs to reuse the compiled
loop; every call of a factory below builds new functions and compiles anew.

``Q`` is the state of the inner cells, shape ``(n_state, n_cells)``.
"""

from typing import Callable, NamedTuple

import jax.numpy as jnp
import numpy as np


class Observer(NamedTuple):
    """``init(Q, Qaux, parameters) -> acc`` and
    ``update(acc, time, dt, Q, Qaux, parameters) -> acc``."""
    init: Callable
    update: Callable


def running_max(row):
    """Maximum of state row ``row`` over all cells and all time levels,
    the initial state included.  ``acc``: scalar."""
    row = int(row)

    def init(Q, Qaux, parameters):
        return jnp.max(Q[row])

    def update(acc, time, dt, Q, Qaux, parameters):
        return jnp.maximum(acc, jnp.max(Q[row]))

    return Observer(init, update)


class GaugeSeries(NamedTuple):
    """Accumulator of :func:`gauge_series`.

    values : ``(n_times, n_cells)``, state row ``row`` at the gauge cells at
        the output times; NaN for an output time the run has not reached.
    time : time of the last observed state.
    last : ``(n_cells,)``, the gauge values at ``time``.
    """
    values: jnp.ndarray
    time: jnp.ndarray
    last: jnp.ndarray


def gauge_series(row, cells, times):
    """Time series of state row ``row`` at the cells ``cells``, sampled at the
    output times ``times``.

    The value at an output time ``t_k`` is the LINEAR INTERPOLATION in time
    between the two time levels of the step that crosses it,
    ``t_old < t_k <= t_new``; ``t_k <= 0`` takes the initial state.  It is
    therefore continuous (and differentiable) in anything that moves the
    adaptive time levels.  ``acc``: :class:`GaugeSeries`; the result is
    ``acc.values``, shape ``(len(times), len(cells))``.
    """
    row = int(row)
    cells = np.asarray(cells, dtype=int).reshape(-1)
    times = np.asarray(times, dtype=float).reshape(-1)

    def init(Q, Qaux, parameters):
        g = Q[row, cells]
        t = jnp.asarray(times, dtype=Q.dtype)
        values = jnp.where((t <= 0.0)[:, None], g[None, :],
                           jnp.full((t.size, g.size), jnp.nan, Q.dtype))
        return GaugeSeries(values, jnp.zeros((), Q.dtype), g)

    def update(acc, time, dt, Q, Qaux, parameters):
        g = Q[row, cells]
        t = jnp.asarray(times, dtype=Q.dtype)
        hit = (t > acc.time) & (t <= time)
        span = jnp.where(time > acc.time, time - acc.time, 1.0)
        w = (t - acc.time) / span
        interp = acc.last[None, :] + w[:, None] * (g - acc.last)[None, :]
        values = jnp.where(hit[:, None], interp, acc.values)
        return GaugeSeries(values, jnp.asarray(time, Q.dtype), g)

    return Observer(init, update)

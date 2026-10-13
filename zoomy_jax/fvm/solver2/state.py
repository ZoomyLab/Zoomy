"""The march state `S` — the ONE cross-backend contract object (design §0).

    S = (time, iteration, i_snapshot, Q, Qaux)

Registered as a JAX pytree so it can be the `lax.while_loop` carry
*unchanged* — this is literally today's `solver_jax.py:1158` carry tuple,
promoted to a named type so every block can be written `S -> S`.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class MarchState(NamedTuple):
    """`(time, iteration, i_snapshot, Q, Qaux)` — logical contents fixed by
    the design; the *container* is the backend's own (here: a jax pytree)."""

    time: Any
    iteration: Any
    i_snapshot: Any
    Q: Any
    Qaux: Any


def proceed(S: MarchState, time_end) -> bool:
    """Loop-continue predicate (design §2).  `time < time_end`.

    Kept a block of its own — a coupled backend replaces the body with
    "coupling ongoing" and nothing else in the march changes.
    """
    return bool(S.time < time_end)


def should_write(time, dt, i_snapshot, write_interval):
    """Canonical drift-free write gate (design §2 / core inventory).

    Returns `(write?, i_snapshot')`.  The next stamp is recomputed as
    `i_snapshot * write_interval` every call rather than accumulated, which
    is what makes it drift-free (foam's gate; the accumulate form was the
    measured double-clamp bug).
    """
    stamp = i_snapshot * write_interval
    write = bool(time + 1e-14 >= stamp)
    return write, (i_snapshot + 1 if write else i_snapshot)

# NOTE: no explicit ``register_pytree_node`` — jax treats ``NamedTuple``
# subclasses as pytrees natively, and a second registration RAISES
# ("Duplicate custom PyTreeDef type registration").

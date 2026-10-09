"""The declared vocabulary a solver author imports (would live in zoomy_core).

Each class is one building block; a backend implements a callable with the
same name.  No strings in authoring code: typos fail at import/attribute.
"""
from symsolve import Block, Effect, Kernel


# ── mesh operations ─────────────────────────────────────────────────────────
class ghost(Block): """Q -> Q with ghost cells from the BC kernels."""
class set_ghost(Block): """(Qg, side, q) -> Qg with the ghost on `side` := q."""
class edge(Block): """(Q, side) -> state of the cell next to boundary `side`."""
class reconstruct(Block): """(Qg, order) -> (Q_L, Q_R) face traces."""
class face_map(Block): """(kernel, Q_L, Q_R, par) -> per-face values."""
class div(Block): """(F, dx) -> cell divergence of face values."""
class cell_map(Block): """(kernel, Q, par) -> per-cell values."""
class reduce_max(Block): """field -> global max (MPI allreduce)."""
class reduce_any(Block): """mask -> global any."""
class select(Block): """(mask, A, B) -> A where mask else B, per cell."""
class comp(Block): """(Q, i) -> row i."""
class assemble(Block): """(rows...) -> Q."""
class lap(Block): """(x, dx) -> discrete Laplacian."""


# ── pointwise kernels (in the real system: the NSM slots) ──────────────────
class kernels:
    rusanov = Kernel()      # sys.numerical_flux + numerical_fluctuations
    wavespeed = Kernel()    # sys.eigenvalues
    troubled = Kernel()     # sys.admissible


# ── effects (ordered automatically) ─────────────────────────────────────────
class io:
    write = Effect()                 # io_write(tok, t, Q)


class couple:                        # preCICE, one participant
    init = Effect(1)                 # (q_edge) -> ongoing
    max_dt = Effect(1)               # () -> window dt
    read = Effect(1)                 # (dt) -> peer state
    write = Effect()                 # (q_edge)
    advance = Effect(1)              # (dt) -> ongoing

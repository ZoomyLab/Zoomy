"""Host side of a preCICE participant, shared by the Python backends (NumPy, JAX): the preCICE
API calls of the `couple_*` effects, and the overlap of a Schwarz-coupled participant
(`ChorinTemplate.precice_march`).

References:
    Chourdakis et al. 2022, https://doi.org/10.12688/openreseurope.14445.2
"""
from __future__ import annotations

import numpy as np


class PreciceAdapter:
    """Host side of the `couple_*` effects: the preCICE API calls only.  `comm` (mpi4py) makes a
    distributed participant, one preCICE process per rank, so it must span the world (one
    `mpiexec` per participant).

    References:
        Chourdakis et al. 2022, https://doi.org/10.12688/openreseurope.14445.2
    """

    def __init__(self, name, config, mesh_name, write, read, comm=None):
        """Open participant `name` from `config` on mesh `mesh_name`, writing the data
        `write` and reading `read`.
        """
        import petsc4py  # noqa: F401  (must precede precice)
        import precice
        self.mesh, self.wn, self.rn = mesh_name, list(write), list(read)
        rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
        if size > 1:
            from mpi4py import MPI
            if MPI.Comm.Compare(comm, MPI.COMM_WORLD) not in (MPI.IDENT, MPI.CONGRUENT):
                raise NotImplementedError("preCICE participant on a sub-communicator "
                                          "(run it as its own mpiexec)")
        self.p = precice.Participant(name, config, rank, size)

    def set_vertices(self, coords):
        """The coupling vertices `(n, >= dim)`, in the order of the data."""
        dim = self.p.get_mesh_dimensions(self.mesh)
        self.ids = self.p.set_mesh_vertices(self.mesh, np.asarray(coords)[:, :dim])

    def _write(self, q):
        """Write the rows of `q` as the write data."""
        ids = getattr(self, "wids", self.ids)
        for i, n in enumerate(self.wn):
            self.p.write_data(self.mesh, n, ids, np.asarray(q[i]))

    def init(self, q):
        """Write the initial data if required, initialise; returns whether coupling is ongoing."""
        if self.p.requires_initial_data():
            self._write(q)
        self.p.initialize()
        self._checkpoints()
        return np.bool_(self.p.is_coupling_ongoing())

    def max_dt(self):
        """Largest time step left in the window."""
        return np.float64(self.p.get_max_time_step_size())

    def set_overlap_vertices(self, write, read):
        """An overlap participant (`ChorinTemplate.precice_march`): the vertices `write` (the
        mirror cells' centres, `(n, >= dim)`) and `read` (the overlap cells') of one mesh, in
        the order of the data; the window may repeat (an implicit scheme), the template
        restarting it from the window start.  `iterations`: the coupling iterations of each
        completed window."""
        dim = self.p.get_mesh_dimensions(self.mesh)
        ids = self.p.set_mesh_vertices(
            self.mesh, np.concatenate([np.asarray(write), np.asarray(read)])[:, :dim])
        n = len(write)
        self.ids, self.wids, self.rids = ids, ids[:n], ids[n:]
        self.iterative, self.iterations, self._iteration, self._repeat = True, [], 0, False

    def read(self):
        """The read data of the peer at the window start, one row per name."""
        return self._read(0.0)

    def read_iterate(self):
        """The peer's read data of its last coupling iteration (the window end), one row per
        name."""
        return self._read(self.p.get_max_time_step_size())

    def _read(self, t):
        """The read data at the time `t` in the window."""
        ids = getattr(self, "rids", self.ids)
        return np.stack([self.p.read_data(self.mesh, n, ids, t) for n in self.rn])

    def repeat(self):
        """Whether preCICE repeats the window of the last `advance` (not converged)."""
        return np.bool_(self._repeat)

    def write(self, q):
        """Write `q`; returns an empty tuple."""
        self._write(q)
        return ()

    def advance(self, dt):
        """Advance by `dt`; returns whether coupling is ongoing."""
        self.p.advance(float(dt))
        self._checkpoints()
        if getattr(self, "iterative", False):
            self._iteration += 1
            if not self._repeat:
                self.iterations.append(self._iteration)
                self._iteration = 0
        return np.bool_(self.p.is_coupling_ongoing())

    def window_complete(self):
        """Whether the last `advance` completed a time window."""
        return np.bool_(self.p.is_time_window_complete())

    def _checkpoints(self):
        """Answer preCICE's checkpoint queries (an implicit-class scheme, e.g. `multi`, asks at
        every window start): writing one is a no-op (an overlap participant's template keeps
        the window start); a window is repeated only by an overlap participant."""
        self._repeat = bool(self.p.requires_reading_checkpoint())
        if self._repeat and not getattr(self, "iterative", False):
            raise NotImplementedError(
                "preCICE asks to repeat a window (implicit coupling iterations): this "
                "participant keeps no checkpoint; use an explicit scheme or max-iterations 1")
        self.p.requires_writing_checkpoint()

    def finalize(self):
        """Finalize the participant."""
        self.p.finalize()


def overlap_participant(cfg, mesh, n_state):
    """The overlap of a Schwarz-coupled participant (`cfg`: `adapter`, `tag`, `overlap` layers):
    sets the adapter's vertices (mirror cells written, overlap cells read, at their centres) and
    returns `(overlap, mirror)` (`zoomy_core.mesh.overlap.overlap_layers`)."""
    from zoomy_core.mesh.overlap import overlap_layers
    a = cfg["adapter"]
    if len(a.wn) != len(a.rn) or len(a.wn) < n_state:
        raise ValueError(f"an overlap participant writes and reads its {n_state} state rows "
                         "(then, with several SSP-RK stages, each stage's pressure rows), as "
                         f"many as the peer; got {len(a.wn)} / {len(a.rn)} data names")
    over, mirror = overlap_layers(mesh, cfg["tag"], cfg["overlap"])
    cc = np.asarray(mesh.cell_centers).T
    a.set_overlap_vertices(cc[mirror], cc[over])
    return over, mirror

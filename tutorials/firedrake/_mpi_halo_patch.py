"""Monkey-patch Firedrake's halo SF calls to work around aliased
``rootdata == leafdata`` buffer rejection under recent PETSc / MPI.

────────────────────────────────────────────────────────────────────
WHEN CAN THIS PATCH BE DELETED?
────────────────────────────────────────────────────────────────────

The patch is needed because Firedrake's ``firedrake/halo.py`` calls
``self.sf.bcast{Begin,End}`` and ``self.sf.reduce{Begin,End}`` with
``dat._data`` as **both** ``rootdata`` and ``leafdata`` (the same
NumPy buffer for source and destination — an in-place SF graph
operation).  Recent PETSc / MPI stacks (observed with PETSc 3.24.5
+ OpenMPI 4.1.6) reject this aliasing and abort with messages of the
form::

    PETSc ERROR: Object is in wrong state
    PETSc ERROR: ... rootdata and leafdata cannot match ...

Concretely, the offending lines in upstream Firedrake (as of
2025-10 release-candidate ``connorjward/2025.10.3rc``,
commit ``9bbb9a83c``) are in :func:`firedrake.halo.Halo.global_to_local_begin`
/ ``global_to_local_end`` / ``local_to_global_begin`` /
``local_to_global_end``::

    self.sf.bcastBegin (mtype, dat._data, dat._data, MPI.REPLACE)
    self.sf.bcastEnd   (mtype, dat._data, dat._data, MPI.REPLACE)
    self.sf.reduceBegin(mtype, dat._data, dat._data, op)
    self.sf.reduceEnd  (mtype, dat._data, dat._data, op)

A clean upstream fix would copy ``dat._data`` into a temporary source
buffer (or use a dedicated send buffer) before the bcast / reduce,
so the source and destination pointers differ.

**To detect when the upstream fix lands**, grep the installed
Firedrake for the aliased pattern::

    python3 -c "import firedrake, pathlib;
                print(pathlib.Path(firedrake.__file__).parent /'halo.py')"
    grep -nE 'bcast(Begin|End).*dat\\._data, dat\\._data' \\
        $(python3 -c 'import firedrake, os; print(os.path.dirname(firedrake.__file__))')/halo.py

If the grep finds nothing, the upstream code has been refactored —
re-test ``mpirun -n 2`` on
``tutorials/firedrake/malpasset_viscous_v2.py`` *without* importing
this module.  When it runs cleanly, **delete this file** and remove
the ``import _mpi_halo_patch; _mpi_halo_patch.apply()`` line from
every notebook / script under ``tutorials/firedrake/``.

────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────

Idempotent and a no-op on ``COMM_WORLD.size == 1`` — safe to import
unconditionally at the top of any Firedrake script.  Call
:func:`apply` once **before** constructing meshes / function spaces.

Workaround mechanism: every halo exchange copies ``dat._data`` into a
fresh source buffer before handing it to ``sf.{bcast,reduce}Begin``,
so the rootdata and leafdata pointers differ and PETSc accepts the
call.  One extra ``memcpy`` per exchange — small compared to the
communication itself.
"""

from __future__ import annotations


_APPLIED = False


def apply():
    """Install the halo-exchange workaround.  Idempotent."""
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    from firedrake import halo as _halo
    from firedrake.petsc import PETSc as _PETSc
    if _PETSc.COMM_WORLD.Get_size() == 1:
        # Patch is unnecessary serially — leave Firedrake's defaults
        # in place so we don't pay the copy cost.
        return

    import numpy as _np
    from mpi4py import MPI as _MPI
    try:
        import pyop2.types.access as _OP2
    except Exception:  # pragma: no cover — older PyOP2 fallback
        from pyop2 import op2 as _OP2

    _orig_get_mtype = _halo._get_mtype

    def _bcast(self, dat):
        mtype, _ = _orig_get_mtype(dat)
        src = _np.array(dat._data, copy=True)
        self.sf.bcastBegin(mtype, src, dat._data, _MPI.REPLACE)
        self.sf.bcastEnd(mtype, src, dat._data, _MPI.REPLACE)

    def _reduce(self, dat, op):
        mtype, _ = _orig_get_mtype(dat)
        src = _np.array(dat._data, copy=True)
        self.sf.reduceBegin(mtype, src, dat._data, op)
        self.sf.reduceEnd(mtype, src, dat._data, op)

    def global_to_local_begin(self, dat, insert_mode):
        assert insert_mode is _OP2.WRITE
        if self.comm.size == 1:
            return
        _bcast(self, dat)

    def global_to_local_end(self, dat, insert_mode):
        return

    def local_to_global_begin(self, dat, insert_mode):
        if self.comm.size == 1:
            return
        mtype, builtin = _orig_get_mtype(dat)
        op_map = {
            (False, _OP2.INC): _MPI.SUM,
            (True,  _OP2.INC): _MPI.SUM,
            (False, _OP2.MIN): _halo._contig_min_op,
            (True,  _OP2.MIN): _MPI.MIN,
            (False, _OP2.MAX): _halo._contig_max_op,
            (True,  _OP2.MAX): _MPI.MAX,
        }
        _reduce(self, dat, op_map[(builtin, insert_mode)])

    def local_to_global_end(self, dat, insert_mode):
        return

    _halo.Halo.global_to_local_begin = global_to_local_begin
    _halo.Halo.global_to_local_end   = global_to_local_end
    _halo.Halo.local_to_global_begin = local_to_global_begin
    _halo.Halo.local_to_global_end   = local_to_global_end

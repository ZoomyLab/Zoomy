"""Monkey-patch Firedrake's halo SF calls to work around
``rootdata == leafdata`` rejection in PETSc 3.20+.

PETSc 3.20+ rejects in-place ``SF.bcast`` / ``SF.reduce`` calls (where
the source and destination buffers are the same array).  Firedrake's
default halo exchange uses ``dat._data`` for both sides, so under
recent PETSc the second-and-later halo exchange of every MPI run dies
with::

    PETSc ERROR: Object is in wrong state ... rootdata and leafdata cannot match

This patch copies ``dat._data`` into a temporary source buffer before
the bcast / reduce, which PETSc accepts.  Lifted with minor cleanups
from ``tutorials/firedrake/malpasset_viscous.py``.

Idempotent and a no-op on ``COMM_WORLD.size == 1`` — safe to import
unconditionally at the top of any Firedrake script.  Call
:func:`apply` once before constructing meshes / function spaces.
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

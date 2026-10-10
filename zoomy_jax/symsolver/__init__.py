"""JAX backend of the symbolic solvers (``zoomy_core.symsolver``).

Importing this package switches XLA's HLO constant folding off (unless
``XLA_FLAGS`` already names it or ``ZOOMY_XLA_CONSTANT_FOLDING=1``): the
solvers carry the mesh topology as constant index arrays, and folding them
made GPU compiles grow with the mesh (960x96 VAM: 128 s -> 45 s, same step
time).  Effective only if JAX's backend is not initialised yet."""
import os as _os

if (_os.environ.get("ZOOMY_XLA_CONSTANT_FOLDING") != "1"
        and "constant_folding" not in _os.environ.get("XLA_FLAGS", "")):
    _os.environ["XLA_FLAGS"] = (_os.environ.get("XLA_FLAGS", "")
                                + " --xla_disable_hlo_passes=constant_folding").strip()

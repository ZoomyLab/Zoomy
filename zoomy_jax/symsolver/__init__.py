"""JAX backend of the symbolic solvers (``zoomy_core.symsolver``).  Importing it switches XLA's HLO
constant folding off (unless ``XLA_FLAGS`` names it or ``ZOOMY_XLA_CONSTANT_FOLDING=1``), because
the constant mesh topology otherwise makes GPU compiles grow with the mesh.
"""
import os as _os

if (_os.environ.get("ZOOMY_XLA_CONSTANT_FOLDING") != "1"
        and "constant_folding" not in _os.environ.get("XLA_FLAGS", "")):
    _os.environ["XLA_FLAGS"] = (_os.environ.get("XLA_FLAGS", "")
                                + " --xla_disable_hlo_passes=constant_folding").strip()

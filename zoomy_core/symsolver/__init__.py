"""Symbolic solvers: solver templates written once as sympy graphs, printed per
language, and run by a backend runtime that implements the building blocks."""
from zoomy_core.symsolver.core import (  # noqa: F401
    Block, Cond, CxxPrinter, Effect, LinSolve, PyPrinter, Root, While,
    contract, unpack)

#: Wave speed of a model without closed-form eigenvalues, on every backend and
#: device: power iteration on A^2 (``SystemModel.spectral_power_steps`` x
#: ``spectral_power_safety``).  One method everywhere, so backends and devices
#: run the same arithmetic (JAX's numerical eig is a host callback on a GPU).
WAVE_SPEED = "power"


def default_wave_speed(nsm):
    """Give ``nsm`` :data:`WAVE_SPEED` before its kernels are built, unless it
    sets ``spectral_radius_method`` itself."""
    if "spectral_radius_method" not in vars(nsm):
        nsm.spectral_radius_method = WAVE_SPEED
        nsm._call = None                                  # rebuild the table
    return nsm

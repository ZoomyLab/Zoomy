"""Symbolic solver templates (a sympy graph of one time step) and the printers that turn
them into code for the JAX, foam and AMReX backends. They sit after the
NumericalSystemModel and before the backend runtime.
"""
from zoomy_core.symsolver.core import (  # noqa: F401
    Block, Cond, CxxPrinter, Effect, LinSolve, PyPrinter, Root, While,
    contract, unpack)
from zoomy_core.systemmodel.system_model import SpectralRadius

#: Wave speed of a model without closed-form eigenvalues, on every backend and
#: device: power iteration on A^2 (``SystemModel.spectral_power_steps`` x
#: ``spectral_power_safety``).  One method everywhere, so backends and devices
#: run the same arithmetic (JAX's numerical eig is a host callback on a GPU).
WAVE_SPEED = SpectralRadius.power


def default_wave_speed(nsm):
    """Give `nsm` the wave speed `WAVE_SPEED` before its kernels are built,
    unless it sets `spectral_radius_method` itself; returns `nsm`.
    """
    if "spectral_radius_method" not in vars(nsm):
        nsm.spectral_radius_method = WAVE_SPEED
        nsm._call = None                                  # rebuild the table
    return nsm

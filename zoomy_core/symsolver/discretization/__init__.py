"""The discretization components of the solver templates, written once for every
backend: face ``reconstruction``, the semi-discrete ``residual`` with its step
pieces (RK stage, MOOD, implicit slots, hygiene), and the ``time`` step size
(SSP-RK, CFL).  The strategies (:mod:`zoomy_core.symsolver.strategies`) compose
them; nothing here knows which strategy calls it."""

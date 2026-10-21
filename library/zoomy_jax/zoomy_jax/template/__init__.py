"""jax side of the solver template: the backend primitive object and the
differentiable implicit solves.  Presets (`HyperbolicSolver` ...) are
re-exported here as they are added (sub-stage 1a-3)."""
from zoomy_jax.template.backend import JaxBackend
from zoomy_jax.template import linalg

__all__ = ["JaxBackend", "linalg"]

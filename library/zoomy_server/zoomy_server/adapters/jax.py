import logging

from zoomy_server.adapters.numpy import NumpyAdapter

logger = logging.getLogger("zoomy.adapter.jax")


class JaxAdapter(NumpyAdapter):
    tag = "jax"

    def solve(self, case, output_dir, on_progress):
        try:
            from zoomy_jax.fvm.solver_jax import HyperbolicSolver
            import zoomy_jax.fvm.flux as fvmflux
        except ImportError:
            logger.warning("zoomy_jax not available, falling back to numpy")
            return super().solve(case, output_dir, on_progress)

        # For now, JAX path uses the same NumericalModel + GeneratedModelSolver
        # pattern as numpy. JAX-specific solver integration can be added later.
        return super().solve(case, output_dir, on_progress)

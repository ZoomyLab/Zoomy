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

        from zoomy_core.misc.misc import Settings
        import zoomy_core.fvm.timestepping as timestepping

        mesh = self._build_mesh(case["mesh"])
        model = self._build_model(case["model"])

        settings = Settings.default()
        settings.output.directory = output_dir
        settings.output.filename = "simulation"
        settings.output.snapshots = case.get("solver", {}).get("output_snapshots", 10)
        settings.output.clean_directory = True

        solver = HyperbolicSolver(
            settings=settings,
            time_end=case.get("solver", {}).get("time_end", 0.1),
            compute_dt=timestepping.adaptive(CFL=case.get("solver", {}).get("cfl", 0.45)),
            flux=fvmflux.Rusanov(),
        )

        Q, Qaux = solver.solve(mesh, model)
        on_progress(-1, case.get("solver", {}).get("time_end", 0.1), 0.0)

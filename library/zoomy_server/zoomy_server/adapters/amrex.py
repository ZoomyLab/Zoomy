"""AMReX solver adapter — the shared FOLDER case format.

Matches numpy/dmplex/jax: the server hands us a ``case_dir`` containing
``model.py`` / ``mesh.py`` / ``settings.json``, and we run it. AMReX only takes
STRUCTURED meshes, so the heavy lifting — building the structured grid from the
domain/resolution (or projecting a gmsh ``.msh`` + its ``$NodeData`` initial
condition onto a structured grid), code-gen -> build -> run -> HDF5 — lives in
the amrex backend as ``zoomy_amrex.run_case`` (REQ-89). This adapter just
resolves the case and calls it, so one case runs on every backend.
"""
import logging

from zoomy_server.adapter import SolverAdapter

logger = logging.getLogger("zoomy.adapter.amrex")


class AmrexAdapter(SolverAdapter):
    tag = "amrex"

    def solve(self, case_dir, output_dir, on_progress):
        try:
            from zoomy_amrex import run_case
        except ImportError as e:
            raise RuntimeError(
                "zoomy_amrex.run_case is not available yet — the amrex backend must "
                "expose the structured-grid run entry point (see REQ-89). Once it "
                f"lands this adapter runs the shared folder case unchanged. ({e})"
            )

        settings = self.load_settings(case_dir)
        # If the case ships a mesh.py that writes a gmsh .msh with $NodeData bed/IC,
        # run it so zoomy_amrex can project it onto the structured grid. For a
        # plain analytic-IC case the grid comes from settings (domain/n_cells).
        self.run_mesh_script(case_dir)
        model = self.resolve_model(case_dir)

        logger.info("amrex: running case on a structured grid via zoomy_amrex.run_case")
        run_case(model, settings, output_dir, on_progress=on_progress)

    def list_models(self):
        try:
            from zoomy_server._registry import scan_models
            return scan_models()
        except Exception:
            return []

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

        # run_case wants a structured spec (dimension/domain/n_cells). The shared
        # folder case carries its grid in mesh.h5 instead — derive the spec from
        # it so ONE case runs on every backend (regular create_1d/create_2d grids).
        if "domain" not in settings or "n_cells" not in settings:
            derived = self._structured_spec_from_mesh(case_dir)
            if derived:
                for k, v in derived.items():
                    settings.setdefault(k, v)
                logger.info("amrex: derived structured spec from mesh.h5: %s", derived)

        logger.info("amrex: running case on a structured grid via zoomy_amrex.run_case")
        run_case(model, settings, output_dir, on_progress=on_progress)

    @staticmethod
    def _structured_spec_from_mesh(case_dir):
        """Read mesh.h5 (BaseMesh layout) and return {dimension, domain, n_cells}
        for a REGULAR create_1d/create_2d grid, else None."""
        import os
        try:
            import h5py
            import numpy as np
            path = os.path.join(case_dir, "mesh.h5")
            if not os.path.exists(path):
                return None
            with h5py.File(path, "r") as f:
                g = f["mesh"] if "mesh" in f else f
                dim = int(np.asarray(g["dimension"])[()])
                verts = np.asarray(g["vertex_coordinates"])  # (dim, n_vert)
            xs = np.unique(np.round(verts[0], 12))
            if dim == 1:
                return {"dimension": 1,
                        "domain": [float(xs.min()), float(xs.max())],
                        "n_cells": int(len(xs) - 1)}
            ys = np.unique(np.round(verts[1], 12))
            return {"dimension": 2,
                    "domain": [float(xs.min()), float(xs.max()),
                               float(ys.min()), float(ys.max())],
                    "n_cells": [int(len(xs) - 1), int(len(ys) - 1)]}
        except Exception as e:
            logger.warning("amrex: could not derive structured spec from mesh.h5: %s", e)
            return None

    def list_models(self):
        try:
            from zoomy_server._registry import scan_models
            return scan_models()
        except Exception:
            return []

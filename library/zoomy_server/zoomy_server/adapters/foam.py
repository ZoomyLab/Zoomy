"""OpenFOAM solver adapter — thin delegate to the foam backend's run entry.

Foam is NOT drivable in-process: it is codegen -> apptainer wmake -> OpenFOAM
polyMesh + 0/ fields -> zoomyFoam -> VTK (DOF count baked constexpr per model).
That pipeline lives in the foam backend as `zoomy_foam.run_case` (REQ-92, mirroring
amrex REQ-89); this adapter just resolves the case and delegates. The foam
run_case should convert its VTK output with `zoomy_prepost.vtk_to_hdf5`.
"""
import logging
import os

from zoomy_server.adapter import SolverAdapter

logger = logging.getLogger("zoomy.adapter.foam")


class FoamAdapter(SolverAdapter):
    tag = "foam"

    def solve(self, case_dir, output_dir, on_progress):
        # Generic runner: a composed case carries its own run.py (the ## Run
        # section) — execute THAT instead of translating settings into solver
        # calls. Legacy case folders (no run.py) fall through unchanged.
        # (Inert until the foam backend wrapper class lands in run.py form.)
        if os.path.exists(os.path.join(case_dir, "run.py")):
            self.run_mesh_script(case_dir)
            self.run_case_script(case_dir, output_dir, on_progress)
            return

        try:
            from zoomy_foam import run_case
        except ImportError as e:
            raise RuntimeError(
                "zoomy_foam.run_case is not available yet — the foam backend must expose "
                "the run entry (codegen -> wmake -> polyMesh + 0/ fields -> zoomyFoam -> VTK, "
                "then zoomy_prepost.vtk_to_hdf5). See REQ-92. Once it lands this adapter runs "
                f"the shared folder case unchanged. ({e})"
            )

        self.run_mesh_script(case_dir)
        settings = self.load_settings(case_dir)
        model = self.resolve_model(case_dir)

        logger.info("foam: delegating to zoomy_foam.run_case")
        run_case(model, settings, output_dir, on_progress=on_progress)

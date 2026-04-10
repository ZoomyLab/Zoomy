"""Mesh converter adapter — generates mesh files from mesh.py scripts.

Case folder: mesh.py (required), optionally .geo files and settings.json.

Server workflow:
  1. Execute mesh.py -> .h5 / .msh
  2. Run gmsh on any .geo files that lack a corresponding .msh
  3. Copy produced mesh files (.msh, .h5) to output_dir
"""

import glob
import logging
import os
import re
import shutil
import subprocess
import sys

from zoomy_server.adapter import SolverAdapter

logger = logging.getLogger("zoomy.adapter.mesh")


class MeshAdapter(SolverAdapter):
    tag = "mesh"

    def solve(self, case_dir, output_dir, on_progress):
        """Generate mesh files from a case folder.

        Parameters
        ----------
        case_dir : str
            Path containing mesh.py and optionally .geo files.
        output_dir : str
            Path where produced .msh / .h5 files are written.
        on_progress : callable(iteration, time, dt)
            Progress callback.  iteration=-1 signals completion.
        """
        on_progress(0, 0, 0)
        logger.info("Mesh generation started for %s", case_dir)

        # 1. Execute mesh.py (the base-class helper already checks existence)
        self.run_mesh_script(case_dir)
        logger.info("mesh.py executed")

        # 2. Run gmsh on any .geo files that don't have a corresponding .msh
        for geo in sorted(glob.glob(os.path.join(case_dir, "*.geo"))):
            msh = geo.replace(".geo", ".msh")
            if not os.path.exists(msh):
                dim = self._detect_geo_dimension(geo)
                logger.info("Running gmsh -%d on %s", dim, os.path.basename(geo))
                subprocess.run(
                    ["gmsh", f"-{dim}", geo, "-o", msh],
                    cwd=case_dir,
                    check=True,
                )

        # 3. Copy produced mesh files to output_dir
        copied = []
        for ext in ("*.msh", "*.h5"):
            for f in sorted(glob.glob(os.path.join(case_dir, ext))):
                dst = os.path.join(output_dir, os.path.basename(f))
                shutil.copy2(f, dst)
                copied.append(os.path.basename(f))

        if copied:
            logger.info("Produced files: %s", ", ".join(copied))
        else:
            logger.warning("No .msh or .h5 files produced in %s", case_dir)

        on_progress(-1, 0, 0)

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _detect_geo_dimension(geo_path):
        """Guess the mesh dimension from .geo file contents.

        Returns 3 if any 3-D primitives (Volume, Extrude, etc.) are found,
        otherwise 2.
        """
        pattern_3d = re.compile(
            r"\b(Volume|Extrude|Box|Sphere|Cylinder|Cone|Torus|BooleanUnion"
            r"|BooleanIntersection|BooleanDifference|BooleanFragments)\b",
            re.IGNORECASE,
        )
        try:
            with open(geo_path) as f:
                for line in f:
                    if line.lstrip().startswith("//"):
                        continue
                    if pattern_3d.search(line):
                        return 3
        except OSError:
            pass
        return 2

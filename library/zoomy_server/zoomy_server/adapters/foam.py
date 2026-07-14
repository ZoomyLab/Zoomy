"""OpenFOAM solver adapter — runs a foam folder-case's own ``run.py``.

Foam is NOT drivable in-process: it is codegen -> wmake -> OpenFOAM polyMesh +
0/ fields -> zoomyFoam -> VTK (DOF count baked constexpr per model), and a
coupled case additionally drives a second preCICE participant. That pipeline is
carried by the case's ``run.py`` (the ``## Run`` section), so this adapter just
executes it.

Host-side the case scripts wrap ``apptainer exec <sif>``; when this adapter runs
INSIDE the zoomy_openfoam container (``foamRun`` on PATH) the same scripts detect
that and run wmake / the coupled solvers directly — one image, no nested
apptainer. See ``thesis/notebooks/coupling/cases/sme_vof/{run,compile}.sh``.
"""
import logging
import os
import shutil
import subprocess
import sys

from zoomy_server.adapter import SolverAdapter

logger = logging.getLogger("zoomy.adapter.foam")


def in_container():
    """True when running inside the zoomy_openfoam image (OpenFOAM on PATH).

    The case scripts use the same predicate (``command -v foamRun``) to choose
    between wrapping ``apptainer exec`` (host) and running directly.
    """
    return shutil.which("foamRun") is not None or os.path.exists(
        "/opt/openfoam13/etc/bashrc"
    )


class FoamAdapter(SolverAdapter):
    tag = "foam"

    def solve(self, case_dir, output_dir, on_progress):
        # Generic runner: a composed case carries its own run.py — execute THAT.
        if os.path.exists(os.path.join(case_dir, "run.py")):
            self._run_inplace(case_dir, output_dir, on_progress)
            return

        # Legacy (no run.py): delegate to the foam backend's in-process entry.
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
        settings["_case_dir"] = os.path.abspath(case_dir)
        return run_case(model, settings, output_dir, on_progress=on_progress)

    # ── in-place run.py runner ───────────────────────────────────────────
    def _run_inplace(self, case_dir, output_dir, on_progress):
        """Run the case's ``run.py`` in the case folder (NOT a copy).

        Unlike the base :meth:`run_case_script`, a foam case folder is a live,
        multi-GB OpenFOAM working tree that manages its OWN ``run/`` + ``outputs/``
        subdirs and reaches sibling scripts (``../compile_sme.sh``) — copying it
        into ``output_dir`` is both wasteful and breaks those references. So run
        in place and promote the produced foam HDF5 into ``output_dir`` (the SME
        participant's ``swe_case.h5`` becomes the served ``simulation.h5``).
        """
        logger.info("foam runner: executing case run.py in place (%s)", case_dir)
        env = dict(os.environ)
        # The case scripts pick up the container python (has zoomy_core/prepost);
        # in-container they also detect foamRun on PATH and skip apptainer.
        env.setdefault("ZOOMY_PY", sys.executable)

        proc = subprocess.Popen(
            [sys.executable, "run.py"], cwd=case_dir,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
        )
        out_lines = []
        for line in proc.stdout:
            s = line.rstrip()
            out_lines.append(s)
            self._parse_progress_line(s, on_progress)
            logger.debug(s)
        proc.wait()
        with open(os.path.join(output_dir, "run.log"), "w") as f:
            f.write("\n".join(out_lines) + "\n")
        if proc.returncode != 0:
            tail = "\n".join(out_lines[-30:])
            raise RuntimeError(
                f"run.py exited with code {proc.returncode}\n"
                f"--- run.py output tail ---\n{tail}"
            )
        logger.info("foam runner: run.py finished: %s",
                    out_lines[-1] if out_lines else "")

        self._promote_foam_h5(case_dir, output_dir)
        time_end = 0.0
        try:
            time_end = float(self.load_settings(case_dir).get("time_end", 0.0))
        except Exception:
            pass
        on_progress(-1, time_end, 0.0)

    @staticmethod
    def _promote_foam_h5(case_dir, output_dir):
        """Copy the foam h5 outputs (``run/outputs/{swe,vof}_case.h5``) into the
        job output dir; the SME participant (``swe_case``) becomes the canonical
        ``simulation.h5`` the server serves."""
        out = os.path.join(case_dir, "run", "outputs")
        primary = os.path.join(out, "swe_case.h5")
        found = []
        if os.path.exists(out):
            for name in sorted(os.listdir(out)):
                if name.endswith(".h5"):
                    src = os.path.join(out, name)
                    shutil.copy2(src, os.path.join(output_dir, name))
                    found.append((name, os.path.getsize(src)))
        canonical = primary if os.path.exists(primary) else (
            os.path.join(out, found[0][0]) if found else None)
        # A case's run entry may stage the canonical artifact itself (the
        # composed live-session launcher drives the real case tree elsewhere
        # and drops simulation.h5 + figures at the case root).
        staged = os.path.join(case_dir, "simulation.h5")
        if canonical is None and os.path.exists(staged):
            canonical = staged
        for name in sorted(os.listdir(case_dir)):
            if name.endswith((".png", ".gif")):
                shutil.copy2(os.path.join(case_dir, name),
                             os.path.join(output_dir, name))
        if canonical:
            shutil.copy2(canonical, os.path.join(output_dir, "simulation.h5"))
            logger.info("foam runner: promoted %s -> simulation.h5 (%d bytes); h5=%s",
                        os.path.basename(canonical),
                        os.path.getsize(canonical),
                        ", ".join(f"{n} ({b} B)" for n, b in found))
        else:
            logger.warning("foam runner: no h5 found under %s", out)

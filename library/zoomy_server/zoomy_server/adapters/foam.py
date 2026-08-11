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
    tag = "OpenFOAM"

    def solve(self, case_dir, output_dir, on_progress):
        # Generic runner: a composed case carries its own run.py — execute THAT.
        if os.path.exists(os.path.join(case_dir, "run.py")):
            self._run_inplace(case_dir, output_dir, on_progress)
            return
        # Coupled thesis cases (and the coupled-case generator) drive both
        # participants from a run.sh instead — support it the same way.
        if os.path.exists(os.path.join(case_dir, "run.sh")):
            self._run_inplace(case_dir, output_dir, on_progress, entry=["bash", "run.sh"])
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
    def _run_inplace(self, case_dir, output_dir, on_progress, entry=None):
        """Run the case's ``run.py`` (or a given ``entry``) in the case folder.

        Unlike the base :meth:`run_case_script`, a foam case folder is a live,
        multi-GB OpenFOAM working tree that manages its OWN ``run/`` + ``outputs/``
        subdirs and reaches sibling scripts (``../compile_sme.sh``) — copying it
        into ``output_dir`` is both wasteful and breaks those references. So run
        in place and promote the produced foam HDF5 into ``output_dir`` (the SME
        participant's ``swe_case.h5`` becomes the served ``simulation.h5``).
        """
        entry = entry or [sys.executable, "run.py"]
        logger.info("foam runner: executing %s in place (%s)", entry, case_dir)
        env = dict(os.environ)
        # The case scripts pick up the container python (has zoomy_core/prepost);
        # in-container they also detect foamRun on PATH and skip apptainer.
        env.setdefault("ZOOMY_PY", sys.executable)

        proc = subprocess.Popen(
            entry, cwd=case_dir,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
        )
        # Stream the run's own output: forward every line to the GUI as the
        # progress ``message`` and append it to run.log AS IT ARRIVES.  Both
        # used to happen only at the end -- the raw line went to logger.debug
        # (invisible) and run.log was written after proc.wait() -- so a foam
        # coupling, which compiles two solvers and then marches, showed nothing
        # at all until it finished or raised.  A build that dies at minute
        # three should say so at minute three.
        out_lines = []
        log_path = os.path.join(output_dir, "run.log")
        with open(log_path, "w", buffering=1) as log:
            for line in proc.stdout:
                s = line.rstrip()
                out_lines.append(s)
                log.write(s + "\n")
                self._parse_progress_line(s, on_progress)
                if s:
                    try:
                        on_progress(-1, 0.0, 0.0, s)
                    except TypeError:
                        # Older server without the message parameter: the run
                        # still proceeds, it is just silent again.
                        pass
                logger.info(s)
        proc.wait()
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
    def _canonical_name(case_dir):
        """Which store the case DECLARES as its result.

        A coupling writes one store per participant, and which of them is the
        case's answer is a property of the coupling, not of alphabetical order —
        so ``coupling.yml``'s ``canonical_output`` decides when it is there.
        Without a declaration, the historical SME participant name.
        """
        manifest = os.path.join(case_dir, "coupling.yml")
        if os.path.exists(manifest):
            try:
                import yaml
                declared = (yaml.safe_load(open(manifest).read()) or {}).get(
                    "canonical_output")
                if declared:
                    return f"{declared}.h5"
            except Exception as e:
                logger.warning("foam runner: unreadable coupling.yml (%s)", e)
        return "swe_case.h5"

    @staticmethod
    def _promote_foam_h5(case_dir, output_dir):
        """Copy the foam h5 outputs (``run/outputs/*.h5``) into the job output
        dir; the store the case declares (see :meth:`_canonical_name`) becomes
        the ``simulation.h5`` the server serves."""
        out = os.path.join(case_dir, "run", "outputs")
        primary = os.path.join(out, FoamAdapter._canonical_name(case_dir))
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

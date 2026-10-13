"""The tested library, the backend containers and the GUI's HTTP protocol.

Environment (README.md): `E2E_CHECKOUT`, `E2E_MODE`, `E2E_RUNTIME`, `E2E_BACKENDS`,
`E2E_IMAGE_<BACKEND>`, `E2E_IMAGES`, `E2E_WORK`, `E2E_CPUS`.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # the harness's own checkout
CHECKOUT = Path(os.environ.get("E2E_CHECKOUT", ROOT)).resolve()
MODE = os.environ.get("E2E_MODE", "dev")     # dev | published
WORK = Path(os.environ.get("E2E_WORK", Path(tempfile.gettempdir()) / "zoomy_backend_e2e")).resolve()

#: the packages a dev-mode container imports from the tested checkout, in PYTHONPATH order
PACKAGES = ("zoomy_core", "zoomy_jax", "zoomy_foam", "zoomy_amrex", "zoomy_server",
            "zoomy_prepost", "zoomy_plotting", "zoomy_cli", "zoomy_gui")
#: also from the tested checkout: the matrix (cases, checks) and the hypre build script
EXTRAS = ("docs/dev/symbolic_solver/matrix", "docs/dev/symbolic_solver/gates",
          "containers/zoomy_amrex/build_hypre_cuda.sh")
#: generated in a snapshot by the backends (build caches); kept between runs
GENERATED = {".bincache", ".symcache", "_build", "tmp_build_dir", "__pycache__"}


def cpus() -> str:
    """`E2E_CPUS`, else this process's CPU affinity as a taskset list."""
    if os.environ.get("E2E_CPUS"):
        return os.environ["E2E_CPUS"]
    mine = sorted(os.sched_getaffinity(0))
    runs, start = [], mine[0]
    for a, b in zip(mine, mine[1:] + [None]):
        if b != a + 1:
            runs.append(f"{start}-{a}" if a > start else str(start))
            start = b
    return ",".join(runs)


# --------------------------------------------------------------------------- library

def _git(*args, cwd) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          check=True).stdout


def provenance(checkout: Path = CHECKOUT) -> dict:
    """`{path: "commit[+dirty]"}` of the checkout and every tested package."""
    out = {}
    for rel in (".", *(f"library/{p}" for p in PACKAGES)):
        d = checkout / rel
        if not d.is_dir():
            continue
        head = _git("rev-parse", "--short", "HEAD", cwd=d).strip()
        dirty = _git("status", "--porcelain", "--untracked-files=no", "--", ".", cwd=d).strip()
        out["superrepo" if rel == "." else rel.split("/")[1]] = head + ("+dirty" if dirty else "")
    return out


def _tracked(d: Path) -> list[str]:
    """Files git tracks under `d` (working-tree content), relative to `d`."""
    out = _git("ls-files", "-z", "-c", cwd=d)
    return [p for p in out.split("\0") if p and (d / p).is_file()]


def snapshot(checkout: Path = CHECKOUT) -> Path:
    """Copy of the checkout's tracked library files (+ EXTRAS) under WORK, mounted at /zoomy.

    A copy, not the checkout itself: the foam / AMReX code generators write into their
    package directories, and the tested checkout may change while a run is going on.
    Generated build caches (GENERATED) survive between runs.
    """
    dest = WORK / "snapshots" / checkout.name
    for rel in (*(f"library/{p}" for p in PACKAGES), *EXTRAS):
        src = checkout / rel
        if src.is_file():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / rel)
            continue
        if not src.is_dir():
            continue
        files = _tracked(src)
        (dest / rel).mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "--from0", "--files-from=-", f"{src}/", f"{dest / rel}/"],
                       input="\0".join(files), text=True, check=True)
        keep = set(files)
        for d, dirs, names in os.walk(dest / rel):
            dirs[:] = [x for x in dirs if x not in GENERATED]
            for n in names:
                p = Path(d, n)
                if str(p.relative_to(dest / rel)) not in keep:
                    p.unlink()
    (dest / "PROVENANCE.json").write_text(json.dumps(provenance(checkout), indent=1))
    return dest


# --------------------------------------------------------------------------- backends

@dataclass(frozen=True)
class Backend:
    name: str        # numpy | jax | amrex | openfoam
    container: str   # containers/<container>
    matrix: str      # matrix.py backend name

    @property
    def adapter(self) -> str:
        """The image's adapter (containers/<c>/BACKEND), = the GUI solver tag it serves."""
        return (ROOT / "containers" / self.container / "BACKEND").read_text().strip()


BACKENDS = {b.name: b for b in (Backend("numpy", "zoomy_numpy", "numpy"),
                                Backend("jax", "zoomy_jax", "jax"),
                                Backend("amrex", "zoomy_amrex", "amrex"),
                                Backend("openfoam", "zoomy_openfoam", "foam"))}


def selected_backends() -> list[str]:
    names = os.environ.get("E2E_BACKENDS")
    return [n.strip() for n in names.split(",") if n.strip()] if names else list(BACKENDS)


def runtime() -> str:
    rt = os.environ.get("E2E_RUNTIME")
    if rt:
        return rt
    return "apptainer" if shutil.which("apptainer") else "docker"


def image(backend: Backend) -> tuple[str | None, str]:
    """`(image, how it was found)`; `(None, reason)` when there is none.

    apptainer: `E2E_IMAGE_<NAME>`, `E2E_IMAGES/<container>.sif` (published SIFs pulled with
    `apptainer pull <c>.sif oras://ghcr.io/zoomylab/<c>_sif:latest`), the tested checkout's
    `containers/<c>/<c>.sif`, `~/git/Zoomy/containers/<c>/<c>.sif`.
    docker: `E2E_IMAGE_<NAME>`, else `ghcr.io/zoomylab/<c>:latest`.
    """
    env = os.environ.get(f"E2E_IMAGE_{backend.name.upper()}")
    if env:
        return env, f"E2E_IMAGE_{backend.name.upper()}"
    c = backend.container
    if runtime() == "docker":
        return f"ghcr.io/zoomylab/{c}:latest", "published (GHCR)"
    cands = []
    if os.environ.get("E2E_IMAGES"):
        cands.append(Path(os.environ["E2E_IMAGES"]) / f"{c}.sif")
    cands += [CHECKOUT / "containers" / c / f"{c}.sif",
              Path.home() / "git/Zoomy/containers" / c / f"{c}.sif"]
    for p in cands:
        if p.exists():
            return str(p), str(p)
    return None, "no image: " + ", ".join(map(str, cands)) + f" (set E2E_IMAGE_{backend.name.upper()})"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class Container:
    """A backend image under apptainer or docker: `serve()` its zoomy-server, `run()` a command.

    The library snapshot at /zoomy; `dev`: its packages first on PYTHONPATH (published: the
    image's own). Always: the harness at /e2e/harness, a scratch dir at /e2e/work (TMPDIR,
    HOME, build caches), no host home and no user site-packages (the host's editable installs
    would shadow the image's).
    """
    backend: Backend
    image: str
    snap: Path | None
    work: Path
    dev: bool = True
    rt: str = field(default_factory=runtime)
    proc: subprocess.Popen | None = None
    name: str | None = None
    port: int | None = None
    log: Path | None = None

    def env(self) -> dict:
        e = {"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg",
             "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1",
             "TMPDIR": "/e2e/work/tmp",
             "ZOOMY_AMREX_SYMCACHE": "/e2e/work/amrex_cache",
             "ZOOMY_FOAM_BINCACHE": "/e2e/work/foam_cache",
             "ZOOMY_AMREX_BUILD_CPUS": cpus(), "ZOOMY_AMREX_RUN_CPUS": cpus()}
        if self.rt == "docker":
            e["HOME"] = "/e2e/work/home"            # apptainer: --home
        if self.snap is not None and self.dev:
            e["PYTHONPATH"] = ":".join(f"/zoomy/library/{p}" for p in PACKAGES)
        if (Path.home() / ".cache/zoomy/hypre-cpu-3.2.0").is_dir() and self.rt == "apptainer":
            e["ZOOMY_HYPRE_CACHE"] = "/e2e/zcache"    # the host's hypre build (read-only)
        return e

    def binds(self) -> list[tuple[str, str, str]]:
        b = [(str(HERE), "/e2e/harness", "ro"), (str(self.work), "/e2e/work", "rw")]
        if self.snap is not None:
            b.append((str(self.snap), "/zoomy", "rw"))
        if "ZOOMY_HYPRE_CACHE" in self.env():
            b.append((str(Path.home() / ".cache/zoomy"), "/e2e/zcache", "ro"))
        return b

    def _prepare(self):
        for d in ("tmp", "home", "amrex_cache", "foam_cache"):
            (self.work / d).mkdir(parents=True, exist_ok=True)

    def _apptainer(self, verb: str, args: list[str]) -> list[str]:
        cmd = ["taskset", "-c", cpus(), "apptainer", verb, "--cleanenv", "--contain", "--pid",
               "--home", f"{self.work / 'home'}:/e2e/work/home"]
        for src, dst, mode in self.binds():
            cmd += ["--bind", f"{src}:{dst}:{mode}"]
        envfile = self.work / "apptainer.env"           # --env splits values at commas
        envfile.write_text("".join(f"{k}='{v}'\n" for k, v in self.env().items()))
        return cmd + ["--env-file", str(envfile), self.image, *args]

    def _docker(self, args: list[str], detach=False, port=None) -> list[str]:
        cmd = ["docker", "run", "--rm", "--cpuset-cpus", cpus(), "--name", self.name,
               "--user", f"{os.getuid()}:{os.getgid()}"]       # bind-mounted files stay ours
        if detach:
            cmd.append("-d")
        if port:
            cmd += ["-p", f"127.0.0.1:{port}:8080"]
        for src, dst, mode in self.binds():
            cmd += ["-v", f"{src}:{dst}:{mode}"]
        for k, v in self.env().items():
            cmd += ["-e", f"{k}={v}"]
        return cmd + [self.image, *args]

    # ---- the server ------------------------------------------------------------------

    def serve(self, timeout=300) -> str:
        """Start zoomy-server (the GUI's "Connect" target), wait for /health; returns the URL."""
        self._prepare()
        self.port = free_port()
        self.name = f"e2e-{self.backend.name}-{self.port}"
        self.log = self.work / f"server-{self.port}.log"
        if self.rt == "apptainer":
            cmd = self._apptainer("run", ["server", self.backend.adapter, str(self.port)])
        else:
            cmd = self._docker(["server", self.backend.adapter, "8080"], port=self.port)
        with open(self.log, "w") as f:
            f.write(" ".join(cmd) + "\n")
            f.flush()
            self.proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                         start_new_session=True)
        url = f"http://127.0.0.1:{self.port}"
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited ({self.proc.returncode}):\n{self.tail()}")
            try:
                h = GuiClient(url).health()
                if h.get("status") == "ok":
                    if h.get("tag") != self.backend.adapter:
                        raise RuntimeError(f"health tag {h.get('tag')!r} != {self.backend.adapter!r}")
                    return url
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                pass
            time.sleep(1)
        raise TimeoutError(f"no healthy server within {timeout}s:\n{self.tail()}")

    def tail(self, n=40, access=False) -> str:
        """The server log's last `n` lines (uvicorn's access lines only with `access`)."""
        if not self.log or not self.log.exists():
            return ""
        lines = self.log.read_text(errors="replace").splitlines()
        if not access:
            lines = [ln for ln in lines if not ln.startswith("INFO:     127.0.0.1")]
        return "\n".join(lines[-n:])

    def stop(self):
        if self.rt == "docker" and self.name:
            subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
        if self.proc is not None:
            # TERM, then KILL the whole group: the server's job workers are forked after
            # uvicorn installed its signal handlers and outlive a TERM
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(self.proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    self.proc.wait(10)
                except subprocess.TimeoutExpired:
                    pass
        self.proc = None

    # ---- a command in the image ------------------------------------------------------

    def run(self, argv: list[str], timeout: float, log: Path) -> int:
        """`argv` in the image (apptainer exec / docker run), output to `log`."""
        self._prepare()
        self.name = f"e2e-{self.backend.name}-run-{os.getpid()}-{int(time.time())}"
        cmd = self._apptainer("exec", argv) if self.rt == "apptainer" else self._docker(argv)
        with open(log, "w") as f:
            f.write(" ".join(cmd) + "\n")
            f.flush()
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                return p.wait(timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                if self.rt == "docker":
                    subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
                p.wait()
                raise


# --------------------------------------------------------------------------- the GUI's client

class JobFailed(RuntimeError):
    """The backend reported the job failed (its error text is the message)."""

    def __init__(self, msg, job_id=None):
        super().__init__(msg)
        self.job_id = job_id


class GuiClient:
    """`zoomy_cli` HttpAdapter (library/zoomy_cli/src/adapters/http_adapter.mjs), in Python.

    `submit_case` is its `submitCase` for a composed case: POST /api/v1/cases
    `{case_py, mesh_b64, mesh_name}`, poll GET /api/v1/jobs/<id> every 2 s (transient errors
    keep polling) until complete / failed / cancelled, then GET
    /api/v1/jobs/<id>/results/hdf5, the bytes the GUI writes to /tmp/zoomy_sim/<id>.h5.
    """

    def __init__(self, url: str, poll_s: float = 2.0):
        self.url = url.rstrip("/")
        self.poll_s = poll_s

    def _req(self, path, data=None, method=None, timeout=60):
        body = None if data is None else json.dumps(data).encode()
        req = urllib.request.Request(self.url + path, data=body, method=method,
                                     headers={"Content-Type": "application/json"} if body else {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()

    def health(self) -> dict:
        return json.loads(self._req("/api/v1/health", timeout=5))

    def submit_case(self, case_py: str, timeout: float, mesh: Path | None = None) -> dict:
        payload = {"case_py": case_py,
                   "mesh_b64": base64.b64encode(mesh.read_bytes()).decode() if mesh else None,
                   "mesh_name": mesh.name if mesh else None}
        job = json.loads(self._req("/api/v1/cases", payload))["job_id"]
        t0, polls, status = time.monotonic(), 0, None
        while True:
            try:
                status = json.loads(self._req(f"/api/v1/jobs/{job}"))
                polls += 1
                if status.get("status") in ("complete", "failed", "cancelled"):
                    break
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                pass
            if time.monotonic() - t0 > timeout:
                self.cancel(job)
                raise TimeoutError(f"job {job} not finished after {timeout:.0f}s "
                                   f"(last status {status})")
            time.sleep(self.poll_s)
        out = {"job_id": job, "status": status["status"], "polls": polls,
               "seconds": round(time.monotonic() - t0, 1), "hdf5": None}
        if status["status"] == "failed":
            raise JobFailed(str(status.get("error")), job)
        if status["status"] == "complete":
            out["hdf5"] = self._req(f"/api/v1/jobs/{job}/results/hdf5", timeout=300)
        return out

    def cancel(self, job):
        try:
            self._req(f"/api/v1/jobs/{job}", method="DELETE")
        except (OSError, urllib.error.URLError):
            pass


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()[:12]

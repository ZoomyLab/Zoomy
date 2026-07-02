"""Launch Jupyter Lab inside a backend container — the container's own Python
env (zoomy_core + zoomy_<backend>) becomes a usable kernel.

Exposed as the `zoomy-jupyter` console-script, so it's present in every image
that installs zoomy_server. Apptainer shares the host network namespace, so
`--ip=0.0.0.0 --port 8888` is reachable at localhost:8888 on the host — connect
from VS Code via "Jupyter: Connect to Existing Server".
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser(description="Launch Jupyter Lab (this backend's env).")
    ap.add_argument("--port", type=int, default=int(os.environ.get("ZOOMY_JUPYTER_PORT", "8888")))
    ap.add_argument("--ip", default="0.0.0.0")
    ap.add_argument("--dir", default=os.environ.get("ZOOMY_ROOT", os.getcwd()),
                    help="root directory served by Jupyter (default: $ZOOMY_ROOT or cwd)")
    args, extra = ap.parse_known_args()
    # Fall back to the cwd if the requested root (e.g. an unbound $ZOOMY_ROOT=
    # /workspace) doesn't exist, so `run img jupyter` works with or without a
    # `--bind repo:/workspace`.
    root = args.dir if os.path.isdir(args.dir) else os.getcwd()
    # exec so signals go straight to jupyter; extra args pass through.
    os.execvp("jupyter", [
        "jupyter", "lab",
        f"--ip={args.ip}", f"--port={args.port}",
        "--no-browser",
        f"--ServerApp.root_dir={root}",
        "--ServerApp.allow_root=True",   # harmless under apptainer (non-root); needed for Docker root
        *extra,
    ])


if __name__ == "__main__":
    main()

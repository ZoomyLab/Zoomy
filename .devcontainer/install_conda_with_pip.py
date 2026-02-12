import yaml
import subprocess
import sys
import os
import re


def install_deps(yaml_file):
    if not os.path.exists(yaml_file):
        print(f"Error: {yaml_file} not found.")
        sys.exit(1)

    with open(yaml_file, "r") as f:
        conf = yaml.safe_load(f)

    dependencies = conf.get("dependencies", [])
    candidates = []

    # 1. Parse and Clean Dependencies
    for dep in dependencies:
        if isinstance(dep, str):
            # Clean up conda versioning: 'numpy=1.24' -> 'numpy==1.24'
            # We ONLY replace '=' if it is NOT preceded by <, >, !, or =
            if re.search(r"(?<![<>!=])=(?![=])", dep):
                clean_dep = dep.replace("=", "==")
            else:
                clean_dep = dep
            candidates.append(clean_dep)

        elif isinstance(dep, dict) and "pip" in dep:
            # Always include explicit pip dependencies
            candidates.extend(dep["pip"])

    # 2. Try to install each one individually
    print(f"Attempting to pre-install {len(candidates)} packages from {yaml_file}...")

    successful = 0
    skipped = 0

    for pkg in candidates:
        try:
            # We use --dry-run first? No, pip dry-run is recent.
            # We just try to install. If it fails, we catch it.
            # Capturing output to avoid spamming logs with errors for skipped packages.
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--no-cache-dir", pkg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f" [OK] {pkg}")
            successful += 1
        except subprocess.CalledProcessError:
            # This handles cases like 'python=3.11' (not a pkg) or 'python-gmsh' (name mismatch)
            print(f" [SKIP] {pkg} (Not found on PyPI or incompatible)")
            skipped += 1

    print(f"\nSummary: {successful} installed, {skipped} skipped.")


if __name__ == "__main__":
    install_deps("install/Zoomy.yml")

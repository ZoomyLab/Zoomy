#!/usr/bin/env python3
import json
import os
import urllib.request

TARGET_DIR = "pyodide/packages"
os.makedirs(TARGET_DIR, exist_ok=True)

packages = ["meshio", "zoomy_core"]


def fetch_wheel(package):
    print(f"Fetching latest wheel for {package}")

    with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/json") as resp:
        data = json.load(resp)

    urls = data["urls"]

    # Prefer py3-none-any wheels (pure Python)
    wheel = next(
        (
            u
            for u in urls
            if u["packagetype"] == "bdist_wheel" and "none-any" in u["filename"]
        ),
        None,
    )

    # fallback: any wheel
    if wheel is None:
        wheel = next((u for u in urls if u["packagetype"] == "bdist_wheel"), None)

    if wheel is None:
        raise RuntimeError(f"No wheel available for {package}")

    wheel_url = wheel["url"]
    filename = os.path.join(TARGET_DIR, wheel["filename"])

    print(f"Downloading {wheel_url} → {filename}")
    urllib.request.urlretrieve(wheel_url, filename)


for pkg in packages:
    fetch_wheel(pkg)

print("Done.")

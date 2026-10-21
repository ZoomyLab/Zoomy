"""Backward-compatible free functions around `MeshCatalog`; new code should use `MeshCatalog`
directly.
"""

from __future__ import annotations

import os

from zoomy_core.mesh.mesh_catalog import (
    MeshCatalog,
    _http_get_json,
    _http_get_json_async,
    _IN_PYODIDE,
    ALLOWED_TYPES,
)

try:
    import meshio
    _HAVE_MESHIO = True
except Exception:
    _HAVE_MESHIO = False


# Module-level singleton, lazily created
_catalog: MeshCatalog | None = None


def _get_catalog() -> MeshCatalog:
    """Return the lazily created module-level `MeshCatalog`."""
    global _catalog
    if _catalog is None:
        _catalog = MeshCatalog()
    return _catalog


# ============================================================
# Mesh listing (async + sync)
# ============================================================
async def show_meshes_async(do_print=True):
    """Async: refresh the catalog, optionally print it, and return its entries."""
    catalog = MeshCatalog(auto_fetch=False)
    await catalog.refresh_async()
    meshes = catalog.entries
    if do_print:
        catalog.list()
    return meshes


def show_meshes(do_print=True):
    """Return (and optionally print) the catalog entries; returns a coroutine under Pyodide."""
    if _IN_PYODIDE:
        return show_meshes_async(do_print)
    catalog = _get_catalog()
    if do_print:
        catalog.list()
    return catalog.entries


# ============================================================
# Mesh download
# ============================================================
async def download_mesh_async(mesh_name, folder="./", filetype="msh", size="medium"):
    """Async: download `mesh_name` of the given `size` and `filetype` into `folder`."""
    catalog = MeshCatalog(auto_fetch=False)
    await catalog.refresh_async()
    return await catalog.download_async(mesh_name, size=size, filetype=filetype, folder=folder)


def download_mesh(mesh_name, folder="./", filetype="msh", size="medium"):
    """Download `mesh_name` of the given `size` and `filetype` into `folder`; returns a
    coroutine under Pyodide.
    """
    if _IN_PYODIDE:
        return download_mesh_async(mesh_name, folder, filetype, size)
    catalog = _get_catalog()
    return catalog.download(mesh_name, size=size, filetype=filetype, folder=folder)


# ============================================================
# meshio physical boundaries
# ============================================================
def get_boundary_names(mesh_path, do_print=True):
    """Return the gmsh physical tags per cell type of a mesh file (needs meshio)."""
    if not _HAVE_MESHIO:
        raise RuntimeError(
            "get_boundary_names requires meshio, which is not available."
        )
    if do_print:
        print(f"Reading mesh: {mesh_path}")
    mesh = meshio.read(mesh_path)

    tags = {}
    if "gmsh:physical" in mesh.cell_data:
        for block, phys in zip(mesh.cells, mesh.cell_data["gmsh:physical"]):
            tags[block.type] = sorted(set(phys))

    if do_print:
        print("Found physical tags:")
        for c, names in tags.items():
            print(f" - {c}: {names}")

    return tags

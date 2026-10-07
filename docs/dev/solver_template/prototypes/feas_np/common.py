"""Shared helpers for the feasibility scripts (scratch only)."""
import os, numpy as np
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax
jax.config.update("jax_enable_x64", True)

from zoomy_core.model.models.swe import SWE
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.mesh import LSQMesh, ensure_lsq_mesh
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
import zoomy_core.model.boundary_conditions as BC


def build_nsm(model, order=2, limiter="venkatakrishnan"):
    sm = SystemModel.from_model(model)
    return NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order, limiter=limiter))


def runtimes(nsm):
    from zoomy_jax.transformation.jax_runtime import JaxRuntime
    rt_np = NumpyRuntimeModel.from_system_model(nsm.sm)
    rt_jx = JaxRuntime.from_nsm(nsm)
    params = np.array(list(nsm.sm.parameter_values.values()), dtype=float)
    return rt_np, rt_jx, params


def bf_geometry(mesh, dim):
    n_bf = mesh.n_boundary_faces
    bf_cells = np.asarray(mesh.boundary_face_cells)
    fidx = np.asarray(mesh.boundary_face_face_indices)
    fc = np.asarray(mesh.face_centers)[fidx, :]          # (n_bf, 3)
    cc = np.asarray(mesh.cell_centers)[:, np.asarray(mesh.face_cells)[0, fidx]].T  # (n_bf,3)
    nrm = np.asarray(mesh.face_normals)[:dim, fidx]       # (dim, n_bf)
    d = np.linalg.norm(fc[:, :dim] - cc[:, :dim], axis=1)
    names = list(mesh.boundary_conditions_sorted_names)
    tag = [names[k] for k in np.asarray(mesh.boundary_face_function_numbers)]
    return dict(n_bf=n_bf, cells=bf_cells, fidx=fidx, fc=fc, cc=cc, nrm=nrm,
                d=d, tag=tag)

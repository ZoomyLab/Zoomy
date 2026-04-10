from typing import TypedDict, Optional


class ModelSpec(TypedDict, total=False):
    class_path: str
    init: dict
    parameters: dict


class MeshSpec(TypedDict, total=False):
    type: str
    domain: list[float]
    n_cells: int
    nx: int
    ny: int


class SolverSpec(TypedDict, total=False):
    time_end: float
    cfl: float
    output_snapshots: int


class ZoomyCase(TypedDict, total=False):
    version: str
    model: ModelSpec
    mesh: MeshSpec
    solver: SolverSpec


class JobProgress(TypedDict, total=False):
    iteration: int
    time: float
    time_end: float
    dt: float


class JobStatus(TypedDict, total=False):
    job_id: str
    status: str
    progress: Optional[JobProgress]
    error: Optional[str]


class HealthInfo(TypedDict, total=False):
    status: str
    tag: str
    version: str
    backends: list[str]

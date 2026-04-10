from pydantic import BaseModel
from typing import Optional


class MeshSpec(BaseModel):
    type: str = "create_1d"
    domain: Optional[list[float]] = None
    n_cells: Optional[int] = None
    nx: Optional[int] = None
    ny: Optional[int] = None
    file: Optional[str] = None


class ModelSpec(BaseModel):
    class_path: str
    init: dict = {}
    parameters: dict = {}


class SolverSpec(BaseModel):
    time_end: float = 0.1
    cfl: float = 0.45
    output_snapshots: int = 10


class ZoomyCase(BaseModel):
    version: str = "1.0"
    model: ModelSpec
    mesh: MeshSpec
    solver: SolverSpec = SolverSpec()


class JobProgress(BaseModel):
    iteration: int = 0
    time: float = 0.0
    time_end: float = 0.0
    dt: float = 0.0


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: Optional[JobProgress] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0"
    tag: str = "numpy"
    backends: list[str] = ["numpy"]

from pydantic import BaseModel
from typing import Optional


class MeshSpec(BaseModel):
    type: str = "create_1d"
    domain: Optional[list[float]] = None
    n_cells: Optional[int] = None
    nx: Optional[int] = None
    ny: Optional[int] = None


class ModelSpec(BaseModel):
    class_path: str
    init: dict = {}
    parameters: dict = {}


class SolverSpec(BaseModel):
    time_end: float = 0.1
    cfl: float = 0.45
    output_snapshots: int = 10
    min_dt: float = 1e-6


class InitialConditionSpec(BaseModel):
    type: str = "constant"
    h_left: Optional[float] = None
    h_right: Optional[float] = None
    x_jump: Optional[float] = None


class ZoomyCase(BaseModel):
    version: str = "1.0"
    model: ModelSpec
    mesh: MeshSpec
    solver: SolverSpec = SolverSpec()
    initial_conditions: Optional[InitialConditionSpec] = None

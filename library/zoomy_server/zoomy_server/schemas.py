"""Case schema: model.py + numerics.py + mesh.py + settings.json"""

from pydantic import BaseModel
from typing import Optional


class SolverSettings(BaseModel):
    time_end: float = 0.1
    cfl: float = 0.45
    output_snapshots: int = 10
    min_dt: float = 1e-6
    reconstruction_order: int = 1
    limiter: str = "venkatakrishnan"
    mesh: str = "mesh.h5"
    initial_conditions: Optional[str] = None

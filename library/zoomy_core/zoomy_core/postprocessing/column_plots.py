"""Column-field postprocessing — RE-EXPORT shim.

The canonical home is now `zoomy_plotting.column_plots` (REQ-135: the
generalizable matplotlib layer lives in the published plotting package).
This module preserves the historical import path
`zoomy_core.postprocessing.column_plots` so every existing case keeps
working unchanged.
"""
from zoomy_plotting.column_plots import *  # noqa: F401,F403
# ``CANONICAL`` and the dataclass are consumed by mesh_plots / cases but are
# not in ``__all__``; re-export them explicitly.
from zoomy_plotting.column_plots import (  # noqa: F401
    CANONICAL, ColumnField,
)
from zoomy_plotting.column_plots import __all__  # noqa: F401

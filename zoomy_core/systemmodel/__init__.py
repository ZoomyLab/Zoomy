"""Runtime operator-form PDE system (`SystemModel`), built from a declarative `Model` via
`SystemModel.from_model`.
"""
from zoomy_core.systemmodel.system_model import (
    SystemModel,
    register_function_slot,
    face_normal_symbols,
)
from zoomy_core.systemmodel.operations import (
    OPERATOR_SLOTS,
    map_operator_slots,
    normalize_face_normal,
    register_aux,
    regularize_pow,
    regularize_depth_direct,
    regularize_depth_aux,
    kp_hinv,
)

__all__ = [
    "SystemModel",
    "register_function_slot",
    "face_normal_symbols",
    "OPERATOR_SLOTS",
    "map_operator_slots",
    "normalize_face_normal",
    "register_aux",
    "regularize_pow",
    "regularize_depth_direct",
    "regularize_depth_aux",
    "kp_hinv",
]

"""Module `zoomy_core.transformation.to_c`."""

from zoomy_core.transformation.generic_c import (
    GenericCppModel,
    GenericCppNumerics,
)


class CppModel(GenericCppModel):
    """
    Configuration wrapper for Model C++ generation.
    Does not override generation logic.
    """

    _output_subdir = ".c_interface"
    _is_template_class = True

    def __init__(self, model, *args, **kwargs):
        """Initialize the instance."""
        super().__init__(model, *args, **kwargs)
        self.real_type = "T"
        self.math_namespace = "std::"


class CppNumerics(GenericCppNumerics):
    """
    Configuration wrapper for Numerics C++ generation.
    """

    _output_subdir = ".c_interface"
    _is_template_class = True

    def __init__(self, numerics, *args, **kwargs):
        # We assume gpu_enabled might be passed in kwargs or we force it if needed
        # Defaults to False in base unless passed
        """Initialize the instance."""
        super().__init__(numerics, *args, **kwargs)
        self.real_type = "T"
        self.math_namespace = "std::"
        # Enable GPU macros by default for this specialized printer if desired:
        self.gpu_enabled = True

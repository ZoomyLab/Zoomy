# Make submodules available via import library.zoomy_*
from importlib import import_module

__all__ = [
    "zoomy_core",
    "zoomy_amrex",
    "zoomy_cli",
    "zoomy_client",
    "zoomy_dmplex",
    "zoomy_fenicsx",
    "zoomy_firedrake",
    "zoomy_foam",
    "zoomy_gui",
    "zoomy_jax",
    "zoomy_js",
    "zoomy_mesh",
    "zoomy_server",
    "zoomy_tests",
    "firedrake_animate",
]

for _name in __all__:
    try:
        globals()[_name] = import_module(f"library.{_name}")
    except ModuleNotFoundError:
        # Optional: helpful during partial checkouts
        print(f"⚠️ Warning: library submodule '{_name}' not found.")

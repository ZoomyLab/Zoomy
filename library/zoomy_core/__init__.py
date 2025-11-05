from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("zoomy-core")
except PackageNotFoundError:
    # Package not installed, e.g. running from source
    __version__ = "0.0.0"

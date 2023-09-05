from importlib.metadata import PackageNotFoundError, version

from .general_ga import run_ga_maximization
from .preconfigured_gas import default_ga

__all__ = ["run_ga_maximization", "default_ga"]

try:
    __version__ = version("mol_ga")
except PackageNotFoundError:
    # package is not installed
    pass

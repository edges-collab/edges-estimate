"""Top-level package for edges-estimate."""

from .calibration import *
from .eor_models import *
from .foregrounds import *
from .likelihoods import *

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

__author__ = """Steven Murray"""
__email__ = "steven.g.murray@asu.edu"
__version__ = version("edges_estimate")

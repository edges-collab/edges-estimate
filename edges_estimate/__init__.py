# -*- coding: utf-8 -*-

"""Top-level package for edges-estimate."""

from .calibration import *
from .eor_models import *
from .foregrounds import *
from .likelihoods import *

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

__author__ = """Steven Murray"""
__email__ = "steven.g.murray@asu.edu"
__version__ = version('edges_estimate')
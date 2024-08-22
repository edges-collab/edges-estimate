"""Top-level package for edges-estimate."""

from importlib.metadata import PackageNotFoundError, version

from . import foregrounds, likelihoods
from .calibration import AntennaQ, CalibratorQ
from .eor_models import AbsorptionProfile, phenom_model

__version__ = version("edges_estimate")

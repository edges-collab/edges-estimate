"""Top-level package for edges-estimate."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from . import foregrounds, likelihoods
from .calibration import AntennaQ, CalibratorQ
from .eor_models import AbsorptionProfile, phenom_model

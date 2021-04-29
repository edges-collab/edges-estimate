"""Functions for generating YAML configs."""
import numpy as np
import yaml
from edges_cal import CalibrationObservation
from pathlib import Path
from typing import Optional, Tuple, Union
from yabf import load_likelihood_from_yaml

from .calibration import CalibratorQ
from .likelihoods import CalibrationChi2


def create_calibration_config_from_calobs(
    calobs: CalibrationObservation,
    fname: Optional[str] = None,
    bounds: bool = True,
    direc: Optional[Union[str, Path]] = Path("."),
) -> Tuple[Path, CalibrationChi2]:
    direc = Path(direc)

    fname = (
        fname
        or f"{calobs.path.name}_l{calobs.freq.min:.2f}MHz_h{calobs.freq.max}MHz_c{calobs.cterms}_w{calobs.wterms}{'bounds' if bounds else '_no_bounds'}"
    )

    # Write out necessary data files
    np.savez(
        (direc / fname).with_suffix(".data.npz"),
        **{k: spec.spectrum.averaged_Q for k, spec in calobs._loads.items()},
    )
    np.savez(
        (direc / fname).with_suffix(".sigma.npz"),
        **{k: np.sqrt(spec.spectrum.variance_Q) for k, spec in calobs._loads.items()},
    )

    prms = {}
    for kind in ["C1", "C2", "Tunc", "Tcos", "Tsin"]:
        poly = getattr(calobs, f"{kind}_poly")
        prms[kind]["length"] = len(poly.coefficients)
        prms[kind]["fiducial"] = list(poly.coefficients[::-1])

        if bounds:
            prms[kind]["min"] = [
                coeff - 20 * np.abs(coeff) for coeff in poly.coefficients[::-1]
            ]
            prms[kind]["max"] = [
                coeff + 20 * np.abs(coeff) for coeff in poly.coefficients[::-1]
            ]

    config = {
        "name": fname,
        "external_modules": ["edges_estimate"],
        "likelihoods": {
            "calibration": {
                "class": "CalibrationChi2",
                "data": str((direc / fname).with_suffix(".data.npz")),
                "kwargs": {
                    "use_model_sigma": False,
                    "sigma": str((direc / fname).with_suffix(".sigma.npz")),
                },
                "components": {
                    "calibrator": {
                        "class": "CalibratorQ",
                        "params": prms,
                        "kwargs": {
                            "path": str(calobs.path),
                            "calobs_args": {
                                "f_low": calobs.freq.min,
                                "f_high": calobs.freq.max,
                                "cterms": calobs.cterms,
                                "wterms": calobs.wterms,
                                "load_kwargs": {
                                    "ignore_times_percent": calobs.open.spectrum.ignore_times_percent,
                                    "cache_dir": calobs.open.spectrum.cache_dir,
                                },
                            },
                        },
                    }
                },
            }
        },
    }

    with open((direc / fname).with_suffix(".config.yml"), "w") as fl:
        yaml.dump(config, fl)

    return (
        (direc / fname).with_suffix(".config.yml"),
        load_likelihood_from_yaml((direc / fname).with_suffix(".config.yml")),
    )

"""Functions for generating YAML configs."""
import numpy as np
import yaml
from edges_cal import CalibrationObservation
from pathlib import Path
from typing import Optional, Tuple, Union
from yabf import load_likelihood_from_yaml
from yabf.core.yaml import FileToLoad

from .calibration import CalibratorQ
from .likelihoods import CalibrationChi2


def write_yaml_dict(dct, indent=0):
    return ("\n" + " " * 2 * indent).join(
        yaml.dump(dct, default_flow_style=False).split("\n")
    )


def create_calibration_config_from_calobs(
    calobs: CalibrationObservation,
    fname: Optional[str] = None,
    bounds: bool = True,
    direc: Optional[Union[str, Path]] = Path("."),
) -> Tuple[Path, CalibrationChi2]:
    direc = Path(direc)

    fname = (
        fname
        or f"R{calobs.io.receiver_num}_{calobs.io.ambient_temp}C_{calobs.io.year}_{calobs.io.month}_{calobs.io.day}_{int(calobs.freq.min)}-{int(calobs.freq.max)}MHz_c{calobs.cterms}_w{calobs.wterms}{'_bounds' if bounds else '_no_bounds'}"
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
        prms[kind] = {}
        poly = getattr(calobs, f"{kind}_poly")
        prms[kind]["length"] = len(poly.coefficients)
        prms[kind]["fiducial"] = [float(p) for p in poly.coefficients[::-1]]

        if bounds:
            prms[kind]["min"] = [
                float(coeff - 20 * np.abs(coeff)) for coeff in poly.coefficients[::-1]
            ]
            prms[kind]["max"] = [
                float(coeff + 20 * np.abs(coeff)) for coeff in poly.coefficients[::-1]
            ]

    config = f"""
name: {fname}
external_modules:
  - edges_estimate
likelihoods:
  calibration:
    class: CalibrationChi2
    data: !npz {fname}.data.npz
    kwargs:
      use_model_sigma: false
      sigma: !npz {fname}.sigma.npz
    components:
      calibrator:
        class: CalibratorQ
        params:
          {write_yaml_dict(prms, indent=5)}
        kwargs:
          path: {calobs.io.original_path}
          calobs_args:
            f_low: {float(calobs.freq.min)}
            f_high: {float(calobs.freq.max)}
            cterms: {calobs.cterms}
            wterms: {calobs.wterms}
            load_kwargs:
              ignore_times_percent: {calobs.open.spectrum.ignore_times_percent}
              cache_dir: {calobs.open.spectrum.cache_dir}
            run_num:
              {write_yaml_dict(calobs.io.run_num, indent=7)}
            repeat_num:
              {write_yaml_dict(calobs.io.s11.repeat_num, indent=7)}

"""

    with open((direc / fname).with_suffix(".config.yml"), "w") as fl:
        fl.write(config)

    return (
        (direc / fname).with_suffix(".config.yml"),
        load_likelihood_from_yaml((direc / fname).with_suffix(".config.yml")),
    )

"""Functions for generating YAML configs."""
import numpy as np
import yaml
from edges_cal import CalibrationObservation
from pathlib import Path
from typing import Optional, Tuple, Union
from yabf import load_likelihood_from_yaml
from .eor_models import AbsorptionProfile
from .calibration import CalibratorQ
from .likelihoods import CalibrationChi2


def write_yaml_dict(dct, indent=0):
    return ("\n" + " " * 2 * indent).join(
        yaml.dump(dct, default_flow_style=False).split("\n")
    )

def write_with_indent(str, indent=0):
  return ("\n"+"  "*indent).join(str.split("\n"))


def create_calibration_config_from_calobs(
    calobs: CalibrationObservation,
    fname: Optional[str] = None,
    bounds: bool = True,
    direc: Optional[Union[str, Path]] = Path("."),
    save_lk_independently: bool=True,
    save_cmp_independently: bool=True
) -> Tuple[Path, CalibrationChi2]:
    direc = Path(direc)

    fname = (
        fname
        or f"R{calobs.io.receiver_num}_{calobs.io.ambient_temp}C_{calobs.io.year:04}-{calobs.io.month:02}-{calobs.io.day:02}_{int(calobs.freq.min)}-{int(calobs.freq.max)}MHz_c{calobs.cterms:02}_w{calobs.wterms:02}{'_bounds' if bounds else '_no_bounds'}"
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

    path = direc.absolute() / fname
    cmp_config = f"""
name: calibrator
class: CalibratorQ
params:
  {write_yaml_dict(prms, indent=1)}
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
    {write_yaml_dict(calobs.io.run_num, indent=2)}
  repeat_num:
    {write_yaml_dict(calobs.io.s11.repeat_num, indent=2)}
"""
    if save_cmp_independently:
        with open(path.with_suffix(".component.yml"), "w") as fl:
          fl.write(cmp_config)
          print(f'Wrote {path.with_suffix(".component.yml")}')

    lk_config = f"""
name: calibration
class: CalibrationChi2
data: !npz {path}.data.npz
use_model_sigma: false
sigma: !npz {path}.sigma.npz
components:
  - {write_with_indent(cmp_config, 2) if not save_cmp_independently else path.with_suffix(".component.yml")}
"""

    if save_lk_independently:
        with open(path.with_suffix(".likelihood.yml"), "w") as fl:
          fl.write(lk_config)
          print(f'Wrote {path.with_suffix(".likelihood.yml")}')

    config = f"""
name: {fname}
external_modules:
  - edges_estimate
likelihoods:
  - {write_with_indent(lk_config, 2) if not save_lk_independently else path.with_suffix(".likelihood.yml")}
"""

    with open(path.with_suffix(".yml"), "w") as fl:
        fl.write(config)
        print(f'Wrote {path.with_suffix(".yml")}')

    return (
        path.with_suffix(".yml"),
        load_likelihood_from_yaml(path.with_suffix(".yml")),
    )


def make_absorption(freq, fiducial = None, fix=tuple()):
    fiducial = fiducial or {}
    params={
        'A': {'max':2, 'min': 0.05, 'fiducial': fiducial.get('A', 0.5)}, 
        'nu0': {'min': 60, 'max': 90, 'fiducial': fiducial.get('nu0', 78.5)}, 
        'tau': {'min': 1, 'max': 20, 'fiducial': fiducial.get('tau', 7)}, 
        'w': {'min':1, 'max': 25, 'fiducial': fiducial.get('w', 18)}
    }

    fid = {p: params.pop(p)['fiducial'] for p in fix}

    return AbsorptionProfile(
        name='absorption',
        fiducial=fid,
        params=params,
        freqs=freq
    )


def create_linear_fg_config(
    data: np.ndarray,
    freq: np.ndarray,
    fname: Optional[str] = None,
    bounds: bool = True,
    direc: Optional[Union[str, Path]] = Path("."),
    f_min: float=50.0,
    f_max: float=100.0,   
    fix = None,
) -> Tuple[Path, CalibrationChi2]:
    direc = Path(direc)

    if fix:
        fixstr = "_fix-" + '-'.join(fix)
    else:
        fix = []
        fixstr = ''
    fname = (
        fname
        or f"linear_fg_{int(f_min)}-{int(f_max)}MHz{'_bounds' if bounds else '_no_bounds'}{fixstr}"
    )

    mask = ~np.isnan(data) & (freq >= f_min) & (freq <= f_max)
    dfname = direc/fname 

    # Write out necessary data files
    np.savez(
        (direc / fname).with_suffix(".data.npz"),
        spectrum=data[mask],
    )
    np.save((direc / fname).with_suffix(".freq.npy"), freq[mask])

    eor = make_absorption(freq[mask], fix=fix)

    path = direc.absolute() / fname
    cmp_config = f"""
name: eor
class: AbsorptionProfile
freqs: !npy {path}.freq.npy
params:
  {write_yaml_dict(eor.params.as_dict(), indent=1)}
"""
    with open(path.with_suffix(".eor.yml"), "w") as fl:
        fl.write(cmp_config)
        print(f'Wrote {path.with_suffix(".component.yml")}')

    lk_config = f"""
name: linear_fg_model
class: LinearFG
data: !npz {path}.data.npz
freq: !npy {path}.freq.npy
sigma: {np.sqrt(np.nanmean(np.square(data[1:] - data[:-1])))}
components:
  - {path.with_suffix(".eor.yml")}
"""

    with open(path.with_suffix(".likelihood.yml"), "w") as fl:
        fl.write(lk_config)
        print(f'Wrote {path.with_suffix(".likelihood.yml")}')

    config = f"""
name: {fname}
external_modules:
  - edges_estimate
likelihoods:
  - {path.with_suffix(".likelihood.yml")}
"""

    with open(path.with_suffix(".yml"), "w") as fl:
        fl.write(config)
        print(f'Wrote {path.with_suffix(".yml")}')

    return (
        path.with_suffix(".yml"),
        load_likelihood_from_yaml(path.with_suffix(".yml")),
    )


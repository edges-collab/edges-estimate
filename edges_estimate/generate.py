"""Functions for generating YAML configs."""
import numpy as np
from edges_cal import CalibrationObservation
from typing import Optional, Union
from .likelihoods import CalibrationChi2
from .calibration import CalibratorQ
from pathlib import Path
import yaml
from yabf import load_likelihood_from_yaml

def create_calibration_config_from_calobs(calobs: CalibrationObservation, fname: Optional[str] = None, bounds: bool=True, direc: Optional[Union[str, Path]]) -> Tuple[Path, CalibrationChi2]:
    direc = Path(direc)

    file_name = fname or f"{calobs.path.name}_l{calobs.freq.min:.2f}MHz_h{calobs.freq.max}MHz_c{calobs.cterms}_w{calobs.wterms}{'bounds' if bounds else '_no_bounds'}"
    
    # Write out necessary data files
    np.savez((direc / fname).with_suffix(".data.npz"), **{k: spec.spectrum.averaged_Q for k, spec in calobs._loads.items()})
    np.savez((direc / fname).with_suffix(".sigma.npz"), **{k: np.sqrt(spec.spectrum.variance_Q) for k, spec in calobs._loads.items()})

    prms = {}
    for kind in ['C1', "C2", "Tunc", "Tcos", "Tsin"]:
        poly = getattr(calobs, f"{kind}_poly")
        prms[kind]['length'] = len(poly.coefficients)
        prms[kind]['fiducial'] = list(poly.coefficients[::-1])

        if bounds:
            prms[kind]['min'] = [coeff - 20*np.abs(coeff) for coeff in poly.coefficients[::-1]]
            prms[kind]['max'] = [coeff + 20*np.abs(coeff) for coeff in poly.coefficients[::-1]]
            

    config = {
        'name': file_name,
        'external_modules': ['edges_estimate'],
        'likelihoods': {
            'calibration': {
                'class': 'CalibrationChi2',
                'data': str(direc/fname).with_suffix(".data.npz")
                'kwargs': {
                    'use_model_sigma': False,
                    'sigma': str(direc/fname).with_suffix(".sigma.npz")
                }
                'components': {
                    'calibrator': {
                        'class': 'CalibratorQ',
                        'params': prms,
                        'kwargs': {
                            'path': str(calobs.path),
                            'calobs_args': {
                                'f_low': calobs.freq.min,
                                'f_high': calobs.freq.max,
                                'cterms': calobs.cterms,
                                'wterms': calobs.wterms,
                                'load_kwargs': {
                                    'ignore_times_percent': calobs.open.spectrum.ignore_times_percent,
                                    'cache_dir': calobs.open.spectrum.cache_dir
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    with open((direc / file_name).with_suffix(".config.yml"), 'w') as fl:
        yaml.dump(config, fl)

    return (direc / file_name).with_suffix(".config.yml"), load_likelihood_from_yaml((direc / file_name).with_suffix(".config.yml"))
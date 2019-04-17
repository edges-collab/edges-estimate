import importlib

import numpy as np
import yaml
from os import path
from .likelihood import Param
from scipy import stats

def _absfile(yml, fname):
    if path.isabs(fname):
        return fname
    else:
        return path.join(path.dirname(path.abspath(yml)), fname)

def load_likelihood_from_yaml(fname, **kwargs):
    with open(fname) as f:
        config = yaml.load(f)

    # Load data.
    # prefer data handed to the function
    data = kwargs.get("data", None)

    if data is None:
        data = config.get("data", None)
        if isinstance(data, str):
            data = np.load(_absfile(fname, data))

    if data is None:
        raise ValueError("Require data either in YAML or passed to loader")

    # Load components
    comp = config.get("components", {})
    components = []

    for name, c in comp.items():
        module = importlib.import_module(c['module'])
        cls = getattr(module, name)

        cmp_data = {}
        for key, val in c.get("data", {}).items():
            if isinstance(val, str):
                if val.endswith(".npz"):
                    cmp_data.update(dict(np.load(_absfile(fname, val))))
                elif val.endswith(".npy"):
                    cmp_data[key] = np.load(_absfile(fname, val))
                else:
                    cmp_data[key] = val
            else:
                cmp_data[key] = val

        components.append(cls(**cmp_data))

    # Load parameters
    params = config.get("parameters", [])

    # Add any included parameters
    inc = params.pop("include", [])
    for incl in inc:
        with open(_absfile(fname, incl), 'rb') as f:
            params.update(yaml.load(f))

    parameters = []
    for pname, p in params.items():
        print(pname, p)
        # The ref value needs to be made into a scipy.stats object
        ref = p.pop("ref", None)
        if ref:
            ref = getattr(stats, ref.pop('dist'))(**ref)

        parameters.append(Param(pname, ref=ref, **p))

    # Load fiducial values
    fiducial = config.get("fiducial", {})

    # Load any other parameters
    other = config.get("kwargs", {})
    lk_data = {}
    for key, val in other.items():
        if isinstance(val, str):
            if val.endswith(".npz"):
                lk_data.update(dict(np.load(val)))
            elif val.endswith(".npy"):
                lk_data[key] = np.load(val)
            else:
                lk_data[key] = val
        else:
            lk_data[key] = val

    # Fiducial just gets munged in with other parameters
    lk_data.update(fiducial)

    # Get the likelihood cls to put it all together
    lk = config.get("likelihood")
    module = importlib.import_module(lk['module'])
    cls = getattr(module, lk['name'])

    return cls(data=data, components=components, params=parameters, **lk_data)

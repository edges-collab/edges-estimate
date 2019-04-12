import numpy as np
import pypolychord as ppc
from emcee import EnsembleSampler
from pypolychord.priors import UniformPrior
from pypolychord.settings import PolyChordSettings


def _flat_array(elements):
    lst = []

    if hasattr(elements, "__len__"):
        for e in elements:
            lst += _flat_array(e)
    else:
        lst.append(elements)

    return lst


def run_mcmc(likelihood, sampler="emcee", sampler_kw={},
             sampling_kw={}):
    ndim = len(likelihood.parameters)

    if sampler == "emcee":
        nwalkers = sampler_kw.pop("nwalkers", 2 * ndim)

        sampler = EnsembleSampler(
            log_prob_fn=likelihood, nwalkers=nwalkers,
            ndim=ndim, **sampler_kw
        )

        sampler.run_mcmc(
            np.array([[p.generate_ref() for i in range(nwalkers)] for p in likelihood.parameters.values()]).T,
            **sampling_kw
        )

        return sampler

    elif sampler == "polychord":
        # A really bad hack!
        nderived = len(_flat_array(likelihood(None)[1]))
        settings = PolyChordSettings(ndim, nderived,
                                     **sampler_kw)

        prior_volume = 0
        for p in likelihood.parameters.values():
            if np.isinf(p.min) or np.isinf(p.max):
                raise ValueError("Polychord requires bounded priors")
            prior_volume += np.log(p.max - p.min)

        # Determine proper prior.
        def prior(hypercube):
            ret = []
            for p, h in zip(likelihood.parameters.values(), hypercube):
                ret.append(UniformPrior(p.min, p.max)(h))
            return ret

        def posterior(p):
            lnl, derived = likelihood(p)
            return max(lnl + np.log(prior_volume), 0.99 * np.nan_to_num(-np.inf)), np.array(_flat_array(derived))

        output = ppc.run_polychord(
            posterior, ndim, nderived,
            settings=settings, prior=prior
        )

        paramnames = [(p.name, p.latex) for p in likelihood.parameters.values()]
        # also have to add derived...
        paramnames += [(f'der{i}', f'der{i}') for i in range(nderived)]
        output.make_paramnames_files(paramnames)

        return output

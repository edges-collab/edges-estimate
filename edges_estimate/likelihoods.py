import numpy as np
from cached_property import cached_property
from yabf import Likelihood, Parameter, Param
from scipy import stats


class Chi2:
    base_parameters = [
        Parameter("sigma", 0.013, min=0, latex=r"\sigma")
    ]

    def get_sigma(self, model, **params):
        return params['sigma']

    def _mock(self, model, **params):
        sigma = self.get_sigma(model, **params)
        return model + np.random.normal(loc=0, scale=sigma, size=len(model))

    def lnl(self, model, **params):
        sigma = self.get_sigma(model, **params)
        nm = stats.norm(loc=model, scale=sigma)

        lnl = np.sum(nm.logpdf(self.data))
        if np.isnan(lnl):
            lnl = -np.inf
        return lnl

    # ===== Potential Derived Quantities
    def residual(self, model, **params):
        return self.data - model

    def rms(self, model, **params):
        return np.sqrt(np.mean((model - self.data)**2))


class MultiComponentSpectrumChi2(Chi2, Likelihood):
    @cached_property
    def freqs(self):
        return self.components[0].freqs

    def _reduce(self, ctx, **dct):
        spectra = np.array([v for k, v in ctx.items() if k.endswith("spectrum")])
        return np.sum(spectra, axis=0)


class MultiComponentSpectrumChi2SigmaLin(MultiComponentSpectrumChi2):
    base_parameters = [
        Parameter("sigma_a", 0.013, min=0, latex=r"\sigma_a"),
        Parameter("sigma_b", 0.0, min=0, latex=r"\sigma_b"),
    ]

    def __init__(self, nuc=75.0, *args, **kwargs):
        self.nuc = nuc
        super().__init__(*args, **kwargs)

    def get_sigma(self, model, **params):
        return (self.freqs/self.nuc)*params['sigma_b'] + params['sigma_a']


class MultiComponentSpectrumChi2SigmaT(MultiComponentSpectrumChi2):
    base_parameters = [
        Parameter("sigma_a", 0.013, min=0, latex=r"\sigma_a"),
        Parameter("sigma_b", 0.0, min=-1, max=1, latex=r"\sigma_b"),
    ]

    def __init__(self, T0=1750, *args, **kwargs):
        self.T0 = T0
        super().__init__(*args, **kwargs)

    def get_sigma(self, model, **params):
        return (model/self.T0)**params['sigma_b'] * params['sigma_a']

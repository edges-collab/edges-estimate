import numpy as np
from cached_property import cached_property
from .mcmc_framework.likelihood import Likelihood, Parameter, Param
from scipy import stats


class Chi2:
    sigma = Parameter(0.013, min=0, latex=r"\sigma")

    def mock(self, fill_in_dct=True, **dct):
        if fill_in_dct:
            dct = self._fill_dct(dct)

        sig = self.model(fill_in_dict=False, **dct)
        return sig + np.random.normal(loc=0, scale=dct['sigma'], size=len(sig))

    def logp(self, model, **dct):
        nm = stats.norm(loc=model, scale=dct['sigma'])

        lnl = np.sum(nm.logpdf(self.data))
        if np.isnan(lnl):
            lnl = -np.inf
        return lnl

    # ===== Potential Derived Quantities
    def residual(self, model, dct):
        return self.data - model

    def rms(self, model, dct):
        return np.sqrt(np.mean((model - self.data)**2))


class MultiComponentSpectrumChi2(Chi2, Likelihood):
    @cached_property
    def freqs(self):
        return self.components[0].freqs

    def _model(self, ctx, **dct):
        spectra = np.array([v for k, v in ctx.items() if k.endswith("spectrum")])
        return np.sum(spectra, axis=0)

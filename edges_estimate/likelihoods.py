import attr
import numpy as np
from scipy import stats
from yabf import Likelihood, Parameter


@attr.s(frozen=True)
class Chi2:
    base_parameters = [
        Parameter("sigma", 0.013, min=0, latex=r"\sigma")
    ]

    sigma = attr.ib(None, kw_only=True)

    def get_sigma(self, model, **params):
        if self.sigma is not None:
            if "sigma" in self.active_params_dct:
                # Act as if sigma is important
                return params['sigma'] * self.sigma
            else:
                return self.sigma
        else:
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
    def residual(self, model, ctx, **params):
        return self.data - model

    def rms(self, model, ctx, **params):
        return np.sqrt(np.mean((model - self.data) ** 2))


@attr.s(frozen=True)
class MultiComponentChi2(Chi2, Likelihood):
    kind = attr.ib("spectrum", validator=attr.validators.instance_of(str), kw_only=True)

    def _reduce(self, ctx, **params):
        models = np.array([v for k, v in ctx.items() if k.endswith(self.kind)])
        return np.sum(models, axis=0)

    def lnl(self, model, **params):
        # return -inf if any bit of the spectrum is negative
        if np.any(model <= 0):
            return -np.inf

        return super().lnl(model, **params)


def _positive(x):
    assert x > 0


@attr.s(frozen=True)
class MultiComponentChi2SigmaLin(MultiComponentChi2):
    base_parameters = [
        Parameter("sigma_a", 0.013, min=0, latex=r"\sigma_a"),
        Parameter("sigma_b", 0.0, min=0, latex=r"\sigma_b"),
    ]

    nuc = attr.ib(75.0, converter=float, validator=_positive, kw_only=True)

    def get_sigma(self, model, **params):
        return (self.freqs / self.nuc) * params['sigma_b'] + params['sigma_a']


@attr.s(frozen=True)
class MultiComponentChi2SigmaT(MultiComponentChi2):
    base_parameters = [
        Parameter("sigma_a", 0.013, min=0, latex=r"\sigma_a"),
        Parameter("sigma_b", 0.0, min=-1, max=1, latex=r"\sigma_b"),
    ]

    T0 = attr.ib(1750, kw_only=True, converter=float)

    def get_sigma(self, model, **params):
        return (model / self.T0) ** params['sigma_b'] * params['sigma_a']

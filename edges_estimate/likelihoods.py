import attr
import numpy as np
from scipy import stats
from yabf import Likelihood, Parameter
from cached_property import cached_property


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

        if type(sigma) == float:
            sigma = sigma * np.ones_like(self.data)

        # Ensure we don't use flagged channels
        mask = np.logical_or(np.isnan(self.data), np.isinf(sigma))
        d = self.data[~mask]
        m = model[~mask]

        nm = stats.norm(loc=m, scale=sigma[~mask])

        lnl = np.sum(nm.logpdf(d))
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
    positive = attr.ib(True, converter=bool, kw_only=True)

    def _reduce(self, ctx, **params):
        models = np.array([v for k, v in ctx.items() if k.endswith(self.kind)])
        return np.sum(models, axis=0)

    def lnl(self, model, **params):
        # return -inf if any bit of the spectrum is negative
        if self.positive and np.any(model <= 0):
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


@attr.s(frozen=True)
class RadiometricAndWhiteNoise(MultiComponentChi2):
    """
    Likelihood with noise model based on Sims et al 2019 (1910.03165)

    This will only work if a single spectrum is used in the likelihood.

    Two tunable parameters exist: alpha_rn, the amplitude offset of the radiometric
    noise, and sigma_wn, the additive white-noise component.
    """
    base_parameters = [
        Parameter("alpha_rn", 1, min=0, max=100, latex=r"\alpha_{\rm rn}"),
        Parameter("sigma_wn", 0.0, min=0, latex=r"\sigma_{\rm wn}"),
    ]

    integration_time = attr.ib(convert=np.float, kw_only=True)  # in seconds!
    weights = attr.ib(1, kw_only=True)

    @weights.validator
    def _wght_validator(self, att, val):
        if type(val) == int and val == 1:
            return
        elif isinstance(val, np.ndarray) and val.shape == self.freqs.shape:
            return
        else:
            raise ValueError(f"weights must be an array with the same length as freqs."
                             f"Got weight.shape == {val.shape} and freqs.shape == {self.freqs.shape}")

    @cached_property
    def freqs(self):
        for cmp in self.components:
            if hasattr(cmp, "freqs"):
                return cmp.freqs

    @cached_property
    def channel_width(self):
        assert np.allclose(np.diff(self.freqs, 2), 0), "the frequencies given are not regular!"
        return (self.freqs[1] - self.freqs[0])*1e6  # convert to Hz

    @cached_property
    def radiometer_norm(self):
        return self.channel_width * self.integration_time

    def get_sigma(self, model, **params):
        return np.sqrt((1/self.weights) * (params['alpha_rn'] * model**2/self.radiometer_norm + params['sigma_wn']**2))


@attr.s(frozen=True)
class CalibrationChi2(Likelihood):
    """
    data should be passed as a dict of {source: qp}
    """
    base_parameters = [
        Parameter("sigma_scale", 1, min=0, latex=r"f_\sigma")
    ]

    white_noise_sigma = attr.ib(False, convert=bool, kw_only=True)

    def _reduce(self, ctx, **params):
        for k in ctx:
            if k.endswith("calibration_q"):
                key = k
                break

        for k in ctx:
            if k.endswith("calibration_qsigma"):
                sigma_key = k
                break

        return {"Qp": ctx[key], "curlyQ": ctx[sigma_key]}

    def get_sigma(self, curlyQ, Qp, **params):
        if self.white_noise_sigma:
            return params['sigma_scale']
        else:
            return params['sigma_scale'] * Qp**2 * (1 + curlyQ)

    def _mock(self, model, **params):
        sigma = self.get_sigma(model, **params)
        return model + np.random.normal(loc=0, scale=sigma, size=len(model))

    def lnl(self, model, **params):
        lnl = 0
        for source, data in self.data.items():
            sigma = self.get_sigma(model['curlyQ'][source], model['Qp'][source], **params)

            lnl += np.sum(-0.5*(np.log(2)+np.log(np.pi) + 2*np.log(sigma) + (model['Qp'][source] - data)**2 / (2 * sigma**2)))
            # nm = stats.norm(loc=model['Qp'][source], scale=sigma)
            # lnl += np.sum(nm.logpdf(data))
            if np.isnan(lnl):
                lnl = -np.inf
                break
        return lnl

    # ===== Potential Derived Quantities
    def residual_open(self, model, ctx, **params):
        return self.data['open'] - model['Qp']['open']

    def residual_short(self, model, ctx, **params):
        return self.data['short'] - model['Qp']['short']

    def residual_hot_load(self, model, ctx, **params):
        return self.data['hot_load'] - model['Qp']['hot_load']

    def residual_ambient(self, model, ctx, **params):
        return self.data['ambient'] - model['Qp']['ambient']

    def rms_open(self, model, ctx, **params):
        return np.sqrt(np.mean((model['Qp']['open'] - self.data['open']) ** 2))

    def rms_short(self, model, ctx, **params):
        return np.sqrt(np.mean((model['Qp']['short'] - self.data['short']) ** 2))

    def rms_hot_load(self, model, ctx, **params):
        return np.sqrt(np.mean((model['Qp']['hot_load'] - self.data['hot_load']) ** 2))

    def rms_ambient(self, model, ctx, **params):
        return np.sqrt(np.mean((model['Qp']['ambient'] - self.data['ambient']) ** 2))

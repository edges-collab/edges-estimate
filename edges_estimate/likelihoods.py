import attr
import numpy as np
from scipy import stats
from yabf import Likelihood, Parameter
from cached_property import cached_property
from yabf.chi2 import Chi2, MultiComponentChi2
from edges_cal import receiver_calibration_func as rcf
from getdist import loadMCSamples
import matplotlib.pyplot as plt
from pathlib import Path

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

    integration_time = attr.ib(converter=np.float, kw_only=True)  # in seconds!
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
    """Data should be passed as a dict of {source: qp}.
    """
    base_parameters = [
        Parameter("sigma_scale", 1, min=0, latex=r"f_\sigma")
    ]

    sigma = attr.ib(None, kw_only=True)
    use_model_sigma = attr.ib(default=False, converter=bool, kw_only=True)

    def _reduce(self, ctx, **params):
        out = {}
        for k in ctx:
            if k.endswith("calibration_q"):
                out['Qp'] = ctx[k]
                break

        for k in ctx:
            if k.endswith("calibration_qsigma"):
                out['curlyQ'] = ctx[k]
                break

        out['data_mask'] = ctx['data_mask']

        return out

    def get_sigma(self, model, source=None, **params):
        if self.sigma is not None:
            if isinstance(self.sigma, dict):
                return self.sigma[source][model['data_mask']]
            else:
                return self.sigma
        elif not self.use_model_sigma:
            return params['sigma_scale']
        else:
            return params['sigma_scale'] * model['Qp'][source]**2 * (1 + model['curlyQ'][source])

    def _mock(self, model, **params):
        sigma = self.get_sigma(model, **params)
        return model + np.random.normal(loc=0, scale=sigma, size=len(model))

    def lnl(self, model, **params):
        lnl = 0
        for source, data in self.data.items():
            sigma = self.get_sigma(model, source=source, **params)
            lnl += -np.nansum(
                np.log(sigma) + (model['Qp'][source] - data[model['data_mask']])**2 / (2 * sigma**2)
            )
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

    def get_polys(self, samples, indices=None):
        """Get the polynomial curves from an MCSamples posterior."""
        names = list(self.child_active_param_dct.keys())
        
        if isinstance(samples, (Path, str)):
            samples = loadMCSamples(samples).samples

        if indices is None:
            indices = list(range(len(samples)))
        if isinstance(indices, int):
            indices = list(range(indices))
            
        c1 = np.zeros((len(indices), len(self['calibrator'].freq)))
        c2 = np.zeros((len(indices), len(self['calibrator'].freq)))
        tunc = np.zeros((len(indices), len(self['calibrator'].freq)))
        tcos = np.zeros((len(indices), len(self['calibrator'].freq)))
        tsin = np.zeros((len(indices), len(self['calibrator'].freq)))
        
        for ix in indices:
            params = {name: v for name, v in zip(names,samples[ix])}
            c1[ix], c2[ix], tunc[ix], tcos[ix], tsin[ix] = self['calibrator'].get_calibration_curves(params)
            
        return c1, c2, tunc, tcos, tsin

    def plot_mc_curves(self, samples, indices=None):
        fig, ax = plt.subplots(5,1, sharex=True, gridspec_kw={"hspace": 0}, figsize=(10, 8))
        freq = self['calibrator'].freq
        calibrator = self['calibrator']
        names = list(calibrator.child_active_param_dct.keys())

        ml_params = np.concatenate(
            (calibrator.calobs.C1_poly.coefficients[::-1], 
            calibrator.calobs.C2_poly.coefficients[::-1], 
            calibrator.calobs.Tunc_poly.coefficients[::-1],
            calibrator.calobs.Tcos_poly.coefficients[::-1], 
            calibrator.calobs.Tsin_poly.coefficients[::-1]
            )
        )

        c1, c2, tunc, tcos, tsin = self.get_polys(samples, indices=indices)


        for i, (name, thing, ml_thing, fid) in enumerate(zip(
            (r'$C_1$', r'$C_2$', r'$T_{\rm unc}$', r'$T_{\rm cos}$', r'$T_{\rm sin}$'), 
            (c1, c2, tunc, tcos, tsin),
            calibrator.get_calibration_curves({name: val for name, val in zip(names, ml_params)}),
            calibrator.get_calibration_curves(
                {apar.name:apar.fiducial for apar in cal_lk.child_active_params}
            )
        )):
            perc = np.percentile(thing, [16, 50, 84], axis=0)
            ax[i].fill_between(freq, perc[0], perc[2], alpha=0.5)
            ax[i].plot(freq, perc[1], label='Median MCMC')
            ax[i].plot(freq, ml_thing, label='MAP')
            #ax[i].plot(freq, fid, label='Iterative')
            
            ax[i].set_ylabel(name)
            
        ax[-1].set_xlabel("Frequency [MHz]")
        ax[0].legend()

@attr.s(frozen=True)
class CalibrationPlus(Likelihood):
    """Data should be passed as a dict of {source: qp}.
    """
    base_parameters = [
        Parameter("sigma_scale", 1, min=0, latex=r"f_\sigma")
    ]
    Q = attr.ib(kw_only=True)
    sigma = attr.ib(None, kw_only=True)
    spec_sigma = attr.ib(None, kw_only=True)
    freq = attr.ib(None, kw_only=True)
    K = attr.ib(None, kw_only=True)
    
    def _reduce(self, ctx, **params):
        out = {}
        for k in ctx:
            if k.endswith("calibration_q"):
                out['Qp'] = ctx[k]
                break

        for k in ctx:
            if k.endswith("calibration_qsigma"):
                out['curlyQ'] = ctx[k]
                break

        out['spectrum'] = np.sum([val for key, val in ctx.items() if key.endswith('spectrum')], axis=0)
        out['cal_curves'] = ctx['cal_curves']
        out['data_mask'] = ctx['data_mask']

        out['recal_spec'] = self.recalibrate(
            self.freq, self.data['uncal_spectrum'], self.K, 
            ctx['cal_curves']['c1'], ctx['cal_curves']['c2'], ctx['cal_curves']['tu'], ctx['cal_curves']['tc'], 
            ctx['cal_curves']['ts']
        )
        return out


    def recalibrate(self, freq, uncal, K, scale, offset, tu, tc, ts):
        a, b = rcf.get_linear_coefficients_from_K(
            K, scale(freq), offset(freq), tu(freq), tc(freq), ts(freq), t_load=300,
        )

        return uncal*a + b

    def get_sigma(self, model, source=None, **params):
        if self.sigma is not None:
            if isinstance(self.sigma, dict):
                return self.sigma[source][model['data_mask']]
            else:
                return self.sigma
        else:
            return params['sigma_scale']

    def _mock(self, model, **params):
        sigma = self.get_sigma(model, **params)
        return model + np.random.normal(loc=0, scale=sigma, size=len(model))

    def lnl(self, model, **params):
        lnl = 0
        for source, data in self.Q.items():
            sigma = self.get_sigma(model, source=source, **params)
            lnl += -np.nansum(
                np.log(sigma) + (model['Qp'][source] - data[model['data_mask']])**2 / (2 * sigma**2)
            )
            if np.isnan(lnl):
                lnl = -np.inf
                break
        
        # Ensure we don't use flagged channels
        mask = ~np.isnan(model['recal_spec'])
        d = model['recal_spec'][mask]
        m = model['spectrum'][mask]

        sigma = self.spec_sigma

        if isinstance(sigma, (float, int)):
            sigma = sigma * np.ones_like(d)

        s = sigma[mask]

        nm = stats.norm(loc=m, scale=s)

        lnl += np.sum(nm.logpdf(d))
        if np.isnan(lnl):
            lnl = -np.inf
        return lnl
from __future__ import annotations
import attr
import numpy as np
from cached_property import cached_property
from edges_cal import receiver_calibration_func as rcf
from getdist import loadMCSamples
from matplotlib import pyplot as plt
from pathlib import Path
from scipy import stats
from yabf import Likelihood, Parameter
from yabf.chi2 import Chi2, MultiComponentChi2
from typing import Union, Tuple, Type, Callable, Literal
from edges_cal.modelling import Model, NoiseWaves, Polynomial, LinLog, CompositeModel
from yabf import ParamVec, Component, ParameterVector
from .eor_models import AbsorptionProfile

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
        return (self.freqs / self.nuc) * params["sigma_b"] + params["sigma_a"]


@attr.s(frozen=True)
class MultiComponentChi2SigmaT(MultiComponentChi2):
    base_parameters = [
        Parameter("sigma_a", 0.013, min=0, latex=r"\sigma_a"),
        Parameter("sigma_b", 0.0, min=-1, max=1, latex=r"\sigma_b"),
    ]

    T0 = attr.ib(1750, kw_only=True, converter=float)

    def get_sigma(self, model, **params):
        return (model / self.T0) ** params["sigma_b"] * params["sigma_a"]


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
            raise ValueError(
                f"weights must be an array with the same length as freqs."
                f"Got weight.shape == {val.shape} and freqs.shape == {self.freqs.shape}"
            )

    @cached_property
    def freqs(self):
        for cmp in self.components:
            if hasattr(cmp, "freqs"):
                return cmp.freqs

    @cached_property
    def channel_width(self):
        assert np.allclose(
            np.diff(self.freqs, 2), 0
        ), "the frequencies given are not regular!"
        return (self.freqs[1] - self.freqs[0]) * 1e6  # convert to Hz

    @cached_property
    def radiometer_norm(self):
        return self.channel_width * self.integration_time

    def get_sigma(self, model, **params):
        return np.sqrt(
            (1 / self.weights)
            * (
                params["alpha_rn"] * model ** 2 / self.radiometer_norm
                + params["sigma_wn"] ** 2
            )
        )


@attr.s(frozen=True)
class CalibrationChi2(Likelihood):
    """Data should be passed as a dict of {source: qp}.
    """

    base_parameters = [Parameter("sigma_scale", 1, min=0, latex=r"f_\sigma")]

    sigma = attr.ib(None, kw_only=True)
    use_model_sigma = attr.ib(default=False, converter=bool, kw_only=True)

    def _reduce(self, ctx, **params):
        out = {}
        for k in ctx:
            if k.endswith("calibration_q"):
                out["Qp"] = ctx[k]
                break

        for k in ctx:
            if k.endswith("calibration_qsigma"):
                out["curlyQ"] = ctx[k]
                break

        out["data_mask"] = ctx["data_mask"]

        return out

    def get_sigma(self, model, source=None, **params):
        if self.sigma is not None:
            if isinstance(self.sigma, dict):
                return self.sigma[source][model["data_mask"]]
            else:
                return self.sigma
        elif not self.use_model_sigma:
            return params["sigma_scale"]
        else:
            return (
                params["sigma_scale"]
                * model["Qp"][source] ** 2
                * (1 + model["curlyQ"][source])
            )

    def _mock(self, model, **params):
        sigma = self.get_sigma(model, **params)
        return model + np.random.normal(loc=0, scale=sigma, size=len(model))

    def lnl(self, model, **params):
        lnl = 0
        for source, data in self.data.items():
            sigma = self.get_sigma(model, source=source, **params)
            lnl += -np.nansum(
                np.log(sigma)
                + (model["Qp"][source] - data[model["data_mask"]]) ** 2
                / (2 * sigma ** 2)
            )
            if np.isnan(lnl):
                lnl = -np.inf
                break
        return lnl

    # ===== Potential Derived Quantities
    def residual_open(self, model, ctx, **params):
        return self.data["open"] - model["Qp"]["open"]

    def residual_short(self, model, ctx, **params):
        return self.data["short"] - model["Qp"]["short"]

    def residual_hot_load(self, model, ctx, **params):
        return self.data["hot_load"] - model["Qp"]["hot_load"]

    def residual_ambient(self, model, ctx, **params):
        return self.data["ambient"] - model["Qp"]["ambient"]

    def rms_open(self, model, ctx, **params):
        return np.sqrt(np.mean((model["Qp"]["open"] - self.data["open"]) ** 2))

    def rms_short(self, model, ctx, **params):
        return np.sqrt(np.mean((model["Qp"]["short"] - self.data["short"]) ** 2))

    def rms_hot_load(self, model, ctx, **params):
        return np.sqrt(np.mean((model["Qp"]["hot_load"] - self.data["hot_load"]) ** 2))

    def rms_ambient(self, model, ctx, **params):
        return np.sqrt(np.mean((model["Qp"]["ambient"] - self.data["ambient"]) ** 2))

    def get_polys(self, samples, indices=None):
        """Get the polynomial curves from an MCSamples posterior."""
        names = list(self.child_active_param_dct.keys())

        if isinstance(samples, (Path, str)):
            samples = loadMCSamples(samples).samples

        if indices is None:
            indices = list(range(len(samples)))
        if isinstance(indices, int):
            indices = list(range(indices))

        c1 = np.zeros((len(indices), len(self["calibrator"].freq)))
        c2 = np.zeros((len(indices), len(self["calibrator"].freq)))
        tunc = np.zeros((len(indices), len(self["calibrator"].freq)))
        tcos = np.zeros((len(indices), len(self["calibrator"].freq)))
        tsin = np.zeros((len(indices), len(self["calibrator"].freq)))

        for ix in indices:
            params = {name: v for name, v in zip(names, samples[ix])}
            c1[ix], c2[ix], tunc[ix], tcos[ix], tsin[ix] = self[
                "calibrator"
            ].get_calibration_curves(params)

        return c1, c2, tunc, tcos, tsin

    def plot_mc_curves(self, samples, indices=None):
        fig, ax = plt.subplots(
            5, 1, sharex=True, gridspec_kw={"hspace": 0}, figsize=(10, 8)
        )
        freq = self["calibrator"].freq
        calibrator = self["calibrator"]
        names = list(calibrator.child_active_param_dct.keys())

        ml_params = np.concatenate(
            (
                calibrator.calobs.C1_poly.coefficients[::-1],
                calibrator.calobs.C2_poly.coefficients[::-1],
                calibrator.calobs.Tunc_poly.coefficients[::-1],
                calibrator.calobs.Tcos_poly.coefficients[::-1],
                calibrator.calobs.Tsin_poly.coefficients[::-1],
            )
        )

        c1, c2, tunc, tcos, tsin = self.get_polys(samples, indices=indices)

        for i, (name, thing, ml_thing, fid) in enumerate(
            zip(
                (
                    r"$C_1$",
                    r"$C_2$",
                    r"$T_{\rm unc}$",
                    r"$T_{\rm cos}$",
                    r"$T_{\rm sin}$",
                ),
                (c1, c2, tunc, tcos, tsin),
                calibrator.get_calibration_curves(
                    {name: val for name, val in zip(names, ml_params)}
                ),
                calibrator.get_calibration_curves(
                    {apar.name: apar.fiducial for apar in self.child_active_params}
                ),
            )
        ):
            perc = np.percentile(thing, [16, 50, 84], axis=0)
            ax[i].fill_between(freq, perc[0], perc[2], alpha=0.5)
            ax[i].plot(freq, perc[1], label="Median MCMC")
            ax[i].plot(freq, ml_thing, label="MAP")

            ax[i].set_ylabel(name)

        ax[-1].set_xlabel("Frequency [MHz]")
        ax[0].legend()

@attr.s(frozen=True, kw_only=True)
class PartialLinearModel(Chi2, Likelihood):
    r"""
    A likelihood where some of the parameters are linear and pre-marginalized.

    Parameters
    ----------
    linear_model
        A linear model containing all the terms that are linear.
    version
        Choice of a version of the likelihood. The default, 'keith', corresponds to 
        the derivation given in the notes, which is faster than using the full matrices.
    variance_func
        A callable function that takes two arguments: ``ctx`` and ``data``, and returns
        an array of model variance. If not provided, the input data must have a key
        called `"data_variance"` that provides static variance (i.e. the :math`\Sigma` in the 
        derivation in the Notes).
    data_func
        A function that has the same signature as ``variance_func``, but returns data
        (i.e. the :math:`d` in the derivation). This might be dependent on non-linear
        parameters (not not the linear ones!). If not provided, the input data must have
        a key called ``"data"``. 
    basis_func 
        It is not recommended to provide this, but if provided it should be a function
        that takes the linear basis, context and data, and returns a new linear model,
        effectively altering the linear basis functions based on the nonlinear parameters.
        
    Notes
    -----
    The general idea is laid out in Monsalve et al. 2018 
    (or https://arxiv.org/pdf/0809.0791.pdf). In this class, the variables are typically
    named the same as in Monsalve et al. 2018 (eg. Q, C, V, Sigma). 
    However, note that M2018 misses some fairly significant simplifications. 

    Eq. 15 of M18 is

    .. math:: d_\star^T \Sigma^{-1} d_\start - d_\star^T \Sigma^{-1} A (A^T \Sigma^{-1} A)^{-1} A^T \Sigma^{-1} d_\star

    where :math:`d_\star`  is the residual of the linear model: :math:`d_star = d - A\hat{\theta}`.
    (note that we're omitting the nonlinear 21cm model from that paper here because
    it's just absorbed into :math:`d`.) Note that part of the second term is just the 
    "hat" matrix from weighted least-squares, i.e.

    .. math:: H = A (A^T \Sigma^{-1} A)^{-1} A^T \Sigma^{-1}

    which when applied to a data vector, returns the maximum-likelihood model of the data.

    Thus we can re-write

    .. math:: d_\star^T \Sigma^{-1} d_\start - d_\star^T \Sigma^{-1} H (d - A \hat{\theta}).

    But :math:`A\hat{\theta}` is equivalent to `H d` (i.e. both produce the maximum
    likelihood model for the data), so we have

    .. math:: d_\star^T \Sigma^{-1} d_\start - d_\star^T \Sigma^{-1} H (d - Hd).

    But the `H` matrix is idempotent, so :math:`Hd - HHd = Hd - Hd = 0`. So we are left
    with the first term only. 
    """
    linear_model: Model = attr.ib()
    version: Literal['mine', 'raul', 'keith', 'raul-full'] = attr.ib(default='keith')
    variance_func: Callable | None = attr.ib(default=None)
    data_func: Callable | None = attr.ib(default=None)
    basis_func: Callable | None = attr.ib(default=None)

    def _reduce(self, ctx, **params):
        if self.variance_func is None:
            var = self.data['data_variance']
        else:
            var = self.variance_func(ctx, self.data)

        if self.data_func is None:
            data = self.data['data']
        else:
            data = self.data_func(ctx, self.data)

        if self.basis_func is None:
            linear_model = self.linear_model
        else:
            linear_model = self.basis_func(self.linear_model, ctx, self.data)

        linear_fit = linear_model.fit(ydata=data, weights=1/var)
        return linear_fit, data, var

    @cached_property
    def Q(self):
        if self.basis_func is not None or self.variance_func is not None:
            raise AttributeError("Q is not static in this instance!")
        return (self.linear_model.basis / self.data['data_variance']).dot(self.linear_model.basis.T)

    @cached_property
    def logdetCinv(self) -> float:
        Cinv = self.Q
        return np.log(np.linalg.det(Cinv))

    @cached_property
    def sigma_plus_v_inverse(self):
        if self.basis_func is not None or self.variance_func is not None:
            raise AttributeError("V is not static in this instance!")
        A = self.linear_model.basis        
        var = self.data['data_variance']
        Sig = np.diag(var)
        SigInv = np.diag(1 / var)
        C = np.linalg.inv(self.Q)
        SigFG = A.T.dot(C.dot(A))
        V = np.linalg.inv(np.linalg.inv(SigFG) - SigInv)
        return np.linalg.inv(Sig + V)

    def lnl(self, model, **params):
        # Ensure we don't use flagged channels
        fit, data, var = model

        if not hasattr(var, '__len__'):
            var = var * np.ones(len(data))

        logdetSig = np.sum(var) if self.variance_func is not None else 0
        
        try:
            logdetCinv = self.logdetCinv
        except AttributeError:
            logdetCinv = np.log(np.linalg.det((fit.model.basis / var).dot(fit.model.basis.T)))
            
        if self.version == 'mine':
            lnl = - 0.5 * (logdetSig + logdetCinv + np.sum(data*fit.residual/var))
        elif self.version == 'keith': 
            lnl =  - 0.5 * (logdetSig + logdetCinv + np.sum(fit.residual**2/var))
        elif self.version == 'raul':
            A = fit.model.basis
            B = A.dot(fit.residual/var)
            try:
                Q = self.Q
            except AttributeError:
                Q = (A / var).dot(A.T)
            lnl = -0.5*(logdetCinv +  logdetSig + B.T.dot(np.linalg.inv(Q).dot(B)) + np.sum(fit.residual**2/var))
        elif self.version == 'raul-full':
            try:
                newsig = self.sigma_plus_v_inverse
            except AttributeError:
                A = fit.model.basis
                C = np.linalg.inv(A.T.dot((A/var)))
                newsig = np.linalg.inv(A.T.dot(C.dot(A)))
                newsig[np.diag_indices_from(newsig)] -= 1/var
                newsig = np.linalg.inv(newsig)
                newsig[np.diag_indices_from(newsig)] += var
                newsig = np.linalg.inv(newsig)
            
            lnl = -0.5 * (logdetSig + logdetCinv + fit.residual.dot(newsig.dot(fit.residual)))
        
        if np.isnan(lnl):
            lnl = -np.inf

        return lnl

@attr.s(frozen=True, kw_only=True)
class TNS(Component):
    x: np.ndarray = attr.ib()
    c_terms: int = attr.ib(default=5)
    
    @cached_property
    def provides(self):
        return ['tns']
    
    @cached_property
    def base_parameters(self):
        return ParameterVector(
            "t_lns",
            fiducial=[1500] + [0] * (self.c_terms - 1),
            length=self.c_terms,
            latex=r"T^{\rm L+NS}_{%s}",
        ).get_params()

    @cached_property
    def model(self):
        return Polynomial(n_terms=self.c_terms, f_center=1).at(x=self.x)
    
    def calculate(self, ctx, **params):
        return self.model(parameters=list(params.values()))

@attr.s(frozen=True, kw_only=True)
class NoiseWaveLikelihood:
    nw_model: NoiseWaves = attr.ib()
    data: dict = attr.ib()
    sig_by_tns: bool = attr.ib(default=True)
    version: Literal['mine', 'raul', 'keith', 'raul-full'] = attr.ib(default='keith')
    
    @cached_property
    def t_ns_model(self):
        return TNS(
            x=np.linspace(-1, 1, len(self.nw_model.freq)), 
            c_terms=self.nw_model.c_terms, 
            params=ParamVec('t_lns', length=self.nw_model.c_terms).get_params(),
        )

    @cached_property
    def partial_linear_model(self):
        return PartialLinearModel(
            linear_model = self.nw_model.linear_model,
            data = self.data,
            components = (self.t_ns_model,),
            data_func = self.transform_data,
            variance_func=self.transform_variance if self.sig_by_tns else None,
            version=self.version
        )

    @classmethod
    def transform_data(cls, ctx: dict, data: dict):
        tns = np.concatenate((ctx['tns'],)*4)
        return data['q'] * tns - data['k0']*data['T']

    @classmethod
    def transform_variance(cls, ctx: dict, data: dict):
        tns = np.concatenate((ctx['tns'],)*4)
        return data['data_variance'] * tns**2
        
    @classmethod
    def from_calobs(cls, calobs, sig_by_sigq=True, **kwargs):
        nw_model = NoiseWaves.from_calobs(calobs)
        k0 = np.concatenate(tuple(calobs.get_K()[src][0] for src in calobs._loads))

        data = {
            'q': np.concatenate(tuple(load.spectrum.averaged_Q for load in calobs._loads.values())),
            'T': np.concatenate(tuple(load.temp_ave * np.ones_like(calobs.freq.freq) for load in calobs._loads.values())),
            'k0': k0
        }

        if sig_by_sigq:
            data['data_variance'] = np.concatenate(tuple(load.spectrum.variance_Q / load.spectrum.n_integrations for load in calobs._loads.values()))
        else:
            data['data_variance'] = 1.0

        return cls(
            nw_model = nw_model,
            data=data,
            **kwargs
        )

@attr.s(frozen=True, kw_only=True)
class NoiseWavesPlusFG:
    freq: np.ndarray = attr.ib()
    _gamma_src: Dict[str, np.ndarray] = attr.ib()
    gamma_ant: np.ndarray = attr.ib()
    gamma_rec: np.ndarray = attr.ib()
    c_terms: int = attr.ib(default=5)
    w_terms: int = attr.ib(default=6)
    fg_model: Model = attr.ib(default=LinLog(n_terms=5))
    parameters: Sequence | None = attr.ib(default=None)

    @cached_property
    def gamma_src(self):
        return {**self._gamma_src, **{'ant': self.gamma_ant}}

    @cached_property
    def src_names(self) -> List[str]:
        """List of names of inputs sources (eg. ambient, hot_load, open, short)."""
        return list(self.gamma_src.keys())

    @cached_property
    def linear_model(self) -> CompositeModel:
        """The actual composite linear model object associated with the noise waves."""
        # K should be a an array of shape (Nsrc Nnu x Nnoisewaveterms)
        K = np.hstack(
            tuple(
                rcf.get_K(gamma_rec=self.gamma_rec, gamma_ant=s11src)
                for s11src in self.gamma_src.values()
            )
        )

        # x is the frequencies repeated for every input source
        x = np.repeat(np.linspace(-1, 1, len(self.freq)), len(self.gamma_src) - 1)

        # Except for the antenna: it should get pure frequencies.
        x = np.concatenate((x, self.freq))

        # K[0] multiples the fg, but not the other models.
        K[0][:len(self.freq)*(len(self.gamma_src) -1)] = 0.0

        return CompositeModel(
            models={
                "tunc": Polynomial(
                    n_terms=self.w_terms,
                    parameters=self.parameters[: self.w_terms]
                    if self.parameters is not None
                    else None,
                    f_center=1,
                ),
                "tcos": Polynomial(
                    n_terms=self.w_terms,
                    parameters=self.parameters[self.w_terms : 2 * self.w_terms]
                    if self.parameters is not None
                    else None,
                    f_center=1,
                ),
                "tsin": Polynomial(
                    n_terms=self.w_terms,
                    parameters=self.parameters[2 * self.w_terms : 3 * self.w_terms]
                    if self.parameters is not None
                    else None,
                    f_center=1,
                ),
                "tload": Polynomial(
                    n_terms=self.c_terms,
                    parameters=self.parameters[3 * self.w_terms :]
                    if self.parameters is not None
                    else None,
                    f_center=1,
                ),
                "fg": self.fg_model
            },
            extra_basis={"tunc": K[1], "tcos": K[2], "tsin": K[3], "tload": -1, 'fg': K[0]},
        ).at(x=x)

    def get_temperature_term(
        self,
        noise_wave: str,
        parameters: Sequence | None = None,
        src: str | None = None,
    ) -> np.ndarray:
        """Get the model for a particular temperature term."""
        out = self.linear_model.model.get_model(
            noise_wave,
            parameters=parameters,
            x=self.linear_model.x,
            with_extra=bool(src),
        )
        if src:
            indx = self.src_names.index(src)
            return out[indx * len(self.freq) : (indx + 1) * len(self.freq)]
        else:
            return out[: len(self.freq)]

    def get_full_model(
        self, src: str, parameters: Sequence | None = None
    ) -> np.ndarray:
        """Get the full model (all noise-waves) for a particular input source."""
        out = self.linear_model(parameters=parameters)
        indx = self.src_names.index(src)
        return out[indx * len(self.freq) : (indx + 1) * len(self.freq)]

    def get_fitted(
        self, data: np.ndarray, weights: np.ndarray | None = None
    ) -> NoiseWaves:
        """Get a new noise wave model with fitted parameters."""
        fit = self.linear_model.fit(ydata=data, weights=weights)
        return attr.evolve(self, parameters=fit.model_parameters)

    @classmethod
    def from_labcal(cls, labcal, fg_model=LinLog(n_terms=5)) -> NoiseWavesPlusFG:
        """Initialize a noise wave model from a calibration observation."""
        return cls(
            freq=labcal.calobs.freq.freq,
            gamma_src=labcal.calobs.s11_correction_models,
            gamma_rec=labcal.calobs.lna.s11_model(labcal.calobs.freq.freq),
            gamma_ant=labcal.antenna_s11,
            c_terms=labcal.calobs.cterms,
            w_terms=labcal.calobs.wterms,
            fg_model=fg_model,
        )

    def __call__(self, **kwargs) -> np.ndarray:
        """Call the underlying linear model."""
        return self.linear_model(**kwargs)

@attr.s(frozen=True, kw_only=True)
class DataCalibrationLikelihood:
    nwfg_model = attr.ib()
    data: dict = attr.ib()
    eor_components = attr.ib()

    @eor_components.default
    def _eorcmp(self):
        return (AbsorptionProfile(freqs=calobs.freq.freq, params=('A', 'w', 'tau', 'nu0')), )

    @cached_property
    def t_ns_model(self):
        return TNS(
            x=np.linspace(-1, 1, len(self.nwfg_model.freq)), 
            c_terms=self.nwfg_model.c_terms, 
            params=ParamVec('t_lns', length=self.nwfg_model.c_terms).get_params(),
        )

    @cached_property
    def partial_linear_model(self):
        return PartialLinearModel(
            linear_model = self.nwfg_model.linear_model,
            data = self.data,
            components = (self.t_ns_model,) + self.eor_components, 
            data_func = self.transform_data,
            variance_func=self.transform_variance,
            version='keith'
        )

    @classmethod
    def transform_data(cls, ctx: dict, data: dict):
        #nsrc = len(data['q']) // len(ctx['tns'])
        tns = ctx['tns']
        Tant = ctx['eor_spectrum']

        out = np.zeros((len(data['q']), len(tns)))
        for i, (src, q) in enumerate(data['q'].items()):
            Tsrc = data['T'][src] if src != 'ant' else Tant
            out[i] = tns*q - data['k0'][src] * Tsrc

        return out.flatten()

    @classmethod
    def transform_variance(cls, ctx: dict, data: dict):
        tns = ctx['tns']
        return np.array([v*tns**2 for v in data['data_variance'].values()]).flatten()
        
    @classmethod
    def from_labcal(cls, labcal, q_ant, qvar_ant, fg_model=LinLog(n_terms=5), **kwargs):
        nwfg_model = NoiseWavesPlusFG.from_labcal(labcal, fg_model=fg_model)
        
        k0 = {src: rcf.get_K(gamma_ant=gamma_src, gamma_rec=nwfg_model.gamma_rec)[0] for src, gamma_src in nwfg_model.gamma_src.items()}

        q = {name: load.spectrum.averaged_Q for name, load in labcal.calobs._loads.items()}
        q['ant'] = q_ant

        T = {name: load.temp_ave * np.ones(labcal.calobs.freq.n) for name, load in labcal.calobs._loads.items()}

        qvar = {name: load.spectrum.averaged_Q / load.spectrum.n_integrations for name, load in labcal.calobs._loads.items()}
        qvar['ant'] = qvar_ant
        
        data = {'q': q, 'T': T, 'k0': k0, 'data_variance': qvar}
        
        return cls(
            nwfg_model = nwfg_model,
            data=data,
            **kwargs
        )

@attr.s(frozen=True)
class RecalibratedSpectrum(Chi2, Likelihood):
    """Likelihood that calibrates input Q before fitting a model to it.

    Requires a component to provide `cal_curves`.
    """

    freq = attr.ib(None, kw_only=True)
    K = attr.ib(None, kw_only=True)

    def _reduce(self, ctx, **params):
        return {
            "spectrum": np.sum(
                [val for key, val in ctx.items() if key.endswith("spectrum")], axis=0,
            ),
            "cal_curves": ctx["cal_curves"],
            "data_mask": ctx["data_mask"],
            "recal_spec": self.recalibrate(
                ctx['freq_obj'].normalize(self.freq),
                self.data["uncal_spectrum"],
                self.K,
                ctx["cal_curves"]["c1"],
                ctx["cal_curves"]["c2"],
                ctx["cal_curves"]["tu"],
                ctx["cal_curves"]["tc"],
                ctx["cal_curves"]["ts"],
            ),
        }

    def recalibrate(self, freq, uncal, K, scale, offset, tu, tc, ts):
        a, b = rcf.get_linear_coefficients_from_K(
            K, scale(freq), offset(freq), tu(freq), tc(freq), ts(freq), t_load=300,
        )

        return uncal * a + b

    def lnl(self, model, **params):
        # Ensure we don't use flagged channels
        mask = ~np.isnan(model["recal_spec"])
        d = model["recal_spec"][mask]
        m = model["spectrum"][mask]

        sigma = self.get_sigma(model)

        if isinstance(sigma, (float, int)):
            sigma = sigma * np.ones_like(d)

        s = sigma[mask]
        nm = stats.norm(loc=m, scale=s)

        lnl = np.sum(nm.logpdf(d))
        if np.isnan(lnl):
            lnl = -np.inf
        return lnl


# @attr.s(kw_only=True)
# class LinearCalibrationPlus(Likelihood):
#     """This is a basic likelihood that implements the AMLP method of Tauscher+2021: https://arxiv.org/2105.01672.
#     """
#     sources = ('ambient', 'hot_load', 'open', 'short')
    
#     freq: np.ndarray = attr.ib()
#     gamma_coeffs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] = attr.ib()
#     lab_qstd: Tuple[np.ndarray] = attr.ib()
    
#     field_sigma: np.ndarray = attr.ib(default=None)
    
#     c_terms: int = attr.ib(default=6)
#     w_terms: int = attr.ib(default=6)
#     fg_terms: int = attr.ib(default=5)
#     cal_model: Union[Type[Model], str] = attr.ib(default="polynomial")
#     fg_model: Union[Type[Model], str] = attr.ib(default="linlog")

#     @cached_property
#     def base_parameters(self):
#         t_lns_terms = ParameterVector(
#             "t_lns", fiducial=[1500] + [0]*(self.c_terms - 1), length=self.c_terms, latex=r"T^{\rm L+NS}_{%s}"
#         ).get_params()

#         return tuple(t_lns_terms + (Parameter('field_sigma', fiducial=0.05, latex=r'\sigma_{\rm field}'),))


#     @cached_property
#     def noise_wave_model(self):
#         """The Calibration model."""
#         return NoiseWaves(
#             freq=self.freq, 
#             gamma_coeffs=self.gamma_coeffs[1:], 
#             c_terms=self.c_terms, 
#             w_terms=self.w_terms, 
#             model=self.cal_model,
#             fg_model=self.fg_model,
#             fg_terms=self.fg_terms,
#             gamma_coeff_fg=self.gamma_coeffs[0][-1],
#         )

#     @cached_property
#     def t_lns_model(self):
#         return Polynomial(n_terms=self.c_terms, default_x=self.freq)

#     def _reduce(self, ctx, **params):
#         return {
#             "spectrum": np.sum(
#                 [val for key, val in ctx.items() if key.endswith("spectrum")], axis=0,
#             ),
#             "t_lns": self.t_lns_model(parameters=[params[f't_lns_{i}'] for i in range(self.c_terms)])
#         }

#     @cached_property
#     def linear_hessian(self) -> np.ndarray:
#         """The Hessian of the linear calibration parameters."""
#         hessian =np.zeros((len(self.noise_wave_model.default_basis),)*2)

#         for i, basis in enumerate(self.noise_wave_model.default_basis):
#             for j, basis2 in enumerate(self.noise_wave_model.default_basis[i:]):
#                 hessian[i, j] = hessian[j,i] = np.sum(basis*basis2 / self.lab_qstd.flatten()**2)

#         return hessian

#     @cached_property
#     def linear_covariance(self) -> np.ndarray:
#         return np.linalg.inv(self.linear_hessian)

#     def noise_wave_fit(self, model=None, params=None):
#         if model is None:
#             params = self._fill_params(params)
#             model = self.reduce_model(params=params)

#         t_src = np.vstack([self.data[f'Tant_{source}']*np.ones(len(self.freq)) for source in self.sources])
#         t_src = np.vstack((t_src, model['spectrum']))

#         data = (model['t_lns']*self.Q - self.gamma_coeffs[0]*t_src).flatten()
        
#         return self.noise_wave_model.fit(data, weights=1/self.sigma.flatten())  # don't need mask here because it masks inside the modelling.
        
#     @cached_property
#     def Q(self):
#         return np.vstack([self.data[f'Q_{source}'] for source in self.sources + ("field", )])
#     @cached_property
#     def sigma(self):
#         return np.vstack((self.lab_qstd, self.field_sigma*np.ones(len(self.freq))))

#     @cached_property
#     def mask(self):
#         return ~np.isnan(self.Q)

#     def field_residual(self, model=None, params=None):
#         fit = self.noise_wave_fit(model=model, params=params)
#         return fit.residual[4*len(self.freq):]

#     def lnl(self, model, **params):
#         # Ensure we don't use flagged channels
#         fit = self.noise_wave_fit(model=model)
#         nm = stats.norm(loc=0, scale=self.sigma.flatten())

#         lnl = np.sum(nm.logpdf(fit.residual[self.mask.flatten()]))
#         lnl += 8e9
#         if np.isnan(lnl):
#             lnl = -np.inf
#         return lnl


@attr.s(kw_only=True)
class LinearFG(Likelihood):
    """This is a basic likelihood that implements the AMLP method of Tauscher+2021: https://arxiv.org/2105.01672.
    """
    freq: np.ndarray = attr.ib()
    sigma: float = attr.ib()
    fg_terms: int = attr.ib(default=5)
    fg_model: Union[Type[Model], str] = attr.ib(default="linlog")

    @cached_property
    def data_mask(self):
        return ~np.isnan(self.data['spectrum'])

    @cached_property
    def base_parameters(self):
        return (Parameter('sigma', fiducial=0.05, latex=r'\sigma'),)

    @cached_property
    def foreground_model(self):
        """The linear foreground model."""
        return Model.get_mdl(self.fg_model)(
            default_x=self.freq,
            n_terms=self.fg_terms
        )

    def _reduce(self, ctx, **params):
        signal = np.sum(
            [val for key, val in ctx.items() if key.endswith("spectrum")], axis=0,
        )

        fit = self.foreground_model.fit(ydata=self.data['spectrum'] - signal, weights=np.where(self.data_mask, 1/self.sigma, 0))

        return {
            "signal": signal,
            "spectrum":  signal + fit.evaluate(),
            "residual": fit.residual,
        }

    @cached_property
    def linear_hessian(self) -> np.ndarray:
        """The Hessian of the linear calibration parameters."""
        hessian =np.zeros((len(self.foreground_model.default_basis),)*2)

        for i, basis in enumerate(self.foreground_model.default_basis):
            for j, basis2 in enumerate(self.foreground_model.default_basis[i:], start=i):
                hessian[i, j] = hessian[j,i] = np.sum(basis*basis2 / self.sigma**2)

        return hessian

    @cached_property
    def linear_covariance(self) -> np.ndarray:
        return np.linalg.inv(self.linear_hessian)

    def lnl(self, model, **params):
        nm = stats.norm(0, scale=self.sigma)
        lnl = np.nansum(nm.logpdf(model['residual']))

        if np.isnan(lnl):
            lnl = -np.inf

        return lnl
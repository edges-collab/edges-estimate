"""Likelihoods for analysing EDGES data."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Callable

import attr
import numpy as np
from astropy import units as u
from cached_property import cached_property
from edges_cal import CalibrationObservation
from edges_cal import noise_waves as nw
from edges_cal import types as tp
from edges_cal.modelling import (
    CompositeModel,
    LinLog,
    Model,
    Polynomial,
    UnitTransform,
)
from edges_cal.simulate import simulate_q_from_calobs
from matplotlib import pyplot as plt
from scipy import stats
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from yabf import Component, Likelihood, Parameter, ParameterVector, ParamVec
from yabf.chi2 import Chi2, MultiComponentChi2

from .eor_models import AbsorptionProfile

logger = logging.getLogger(__name__)

try:
    profile  # noqa: B018
except NameError:

    def profile(fnc):
        return fnc


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
    Likelihood with noise model based on Sims et al 2019 (1910.03165).

    This will only work if a single spectrum is used in the likelihood.

    Two tunable parameters exist: alpha_rn, the amplitude offset of the radiometric
    noise, and sigma_wn, the additive white-noise component.
    """

    base_parameters = [
        Parameter("alpha_rn", 1, min=0, max=100, latex=r"\alpha_{\rm rn}"),
        Parameter("sigma_wn", 0.0, min=0, latex=r"\sigma_{\rm wn}"),
    ]

    integration_time = attr.ib(converter=float, kw_only=True)  # in seconds!
    weights = attr.ib(1, kw_only=True)

    @weights.validator
    def _wght_validator(self, att, val):
        if (isinstance(val, int) and val == 1) or (
            isinstance(val, np.ndarray) and val.shape == self.freqs.shape
        ):
            return

        raise ValueError(
            f"weights must be an array with the same length as freqs."
            f"Got weight.shape == {val.shape} and freqs.shape == {self.freqs.shape}"
        )

    @cached_property
    def freqs(self):
        for cmp in self.components:
            if hasattr(cmp, "freqs"):
                return cmp.freqs
        raise ValueError("No freqs found in any component")

    @cached_property
    def channel_width(self):
        assert np.allclose(np.diff(self.freqs, 2), 0), "the frequencies given are not regular!"
        return (self.freqs[1] - self.freqs[0]) * 1e6  # convert to Hz

    @cached_property
    def radiometer_norm(self):
        return self.channel_width * self.integration_time

    def get_sigma(self, model, **params):
        return np.sqrt(
            (1 / self.weights)
            * (params["alpha_rn"] * model**2 / self.radiometer_norm + params["sigma_wn"] ** 2)
        )


@attr.s(frozen=True, kw_only=True)
class PartialLinearModel(Chi2, Likelihood):
    r"""
    A likelihood where some of the parameters are linear and pre-marginalized.

    Parameters
    ----------
    linear_model
        A linear model containing all the terms that are linear.
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

    .. math:: d_\star^T \Sigma^{-1} d_\start - d_\star^T \Sigma^{-1} A
              (A^T \Sigma^{-1} A)^{-1} A^T \Sigma^{-1} d_\star

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
    variance_func: Callable | None = attr.ib(default=None)
    data_func: Callable | None = attr.ib(default=None)
    basis_func: Callable | None = attr.ib(default=None)
    subtract_fiducial: bool = attr.ib(default=False)
    verbose: bool = attr.ib(False)

    @profile
    def _reduce(self, ctx, **params):
        if self.variance_func is None:
            var = self.data["data_variance"]
        else:
            var = self.variance_func(ctx, self.data)

        data = self.data["data"] if self.data_func is None else self.data_func(ctx, self.data)

        linear_model = (
            self.linear_model
            if self.basis_func is None
            else self.basis_func(self.linear_model, ctx, self.data)
        )

        wght = 1.0 if np.all(var == 0) else 1 / var

        linear_fit = linear_model.fit(ydata=data, weights=wght)
        return linear_fit, data, var

    @cached_property
    def Q(self):
        if self.basis_func is not None or self.variance_func is not None:
            raise AttributeError("Q is not static in this instance!")
        return (self.linear_model.basis / self.data["data_variance"]).dot(self.linear_model.basis.T)

    @cached_property
    def logdetCinv(self) -> float | None:
        if np.all(self.data["data_variance"] == 0):
            return 0.0

        try:
            Cinv = self.Q
            return np.log(np.linalg.det(Cinv))
        except AttributeError:
            return None

    @cached_property
    def sigma_plus_v_inverse(self):
        if self.basis_func is not None or self.variance_func is not None:
            raise AttributeError("V is not static in this instance!")
        A = self.linear_model.basis
        var = self.data["data_variance"]
        Sig = np.diag(var)
        SigInv = np.diag(1 / var)
        C = np.linalg.inv(self.Q)
        SigFG = A.T.dot(C.dot(A))
        V = np.linalg.inv(np.linalg.inv(SigFG) - SigInv)
        return np.linalg.inv(Sig + V)

    @cached_property
    def fiducial_lnl(self):
        return attr.evolve(self, subtract_fiducial=False, verbose=False)()[0]

    @profile
    def logdet_cinv(self, model, ctx, **params) -> float:
        """A derived quantity, the log-determinant of the inverse of C."""
        if self.logdetCinv is None:
            fit = model[0]
            h = fit.hessian
            return np.log(np.linalg.det(h))
        return self.logdetCinv

    def logdet_sig(self, model, ctx, **params):
        """A derived quantity, the log-determinant of the covariance matrix."""
        var = model[-1]

        if not hasattr(var, "__len__"):
            var = var * np.ones(len(model[1]))
        elif np.all(var == 0):
            var = np.ones_like(var)

        var = var[~np.isinf(var)]

        return np.sum(np.log(var)) if self.variance_func is not None else 0

    def rms(self, model, ctx, **params):
        fit, data, var = model

        if not hasattr(var, "__len__"):
            var = var * np.ones(len(data))
        elif np.all(var == 0):
            var = np.ones_like(var)

        mask = ~np.isinf(var)
        data = data[mask]
        resid = fit.residual[mask]
        var = var[mask]

        return np.nansum(resid**2 / var)

    @profile
    def lnl(self, model, **params):
        # Ensure we don't use flagged channels
        logdetSig = self.logdet_sig(model, None, **params)
        logdetCinv = self.logdet_cinv(model, None, **params)
        rms = self.rms(model, None, **params)

        lnl = -0.5 * (logdetSig + logdetCinv + rms)

        if np.isnan(lnl):
            lnl = -np.inf

        if self.subtract_fiducial:
            lnl -= self.fiducial_lnl

        if self.verbose:
            print(params, lnl)

        if np.isnan(lnl) or np.isinf(lnl):
            logger.warn(f"Got bad log-likelihood: {lnl} for params: {params}")

        return lnl

    def get_unmarginalized_lnl(self, linear_params, nonlinear_params):
        ctx = self.get_ctx(params=nonlinear_params)

        var = (
            self.data["data_variance"]
            if self.variance_func is None
            else self.variance_func(ctx, self.data)
        )

        data = self.data["data"] if self.data_func is None else self.data_func(ctx, self.data)

        linear_model = (
            self.linear_model
            if self.basis_func is None
            else self.basis_func(self.linear_model, ctx, self.data)
        )

        linear = linear_model(parameters=linear_params)

        resid = data - linear
        nm = stats.norm(loc=0, scale=np.sqrt(var))
        return np.sum(nm.logpdf(resid))

    def _extracted_from_lnl_45(self, basis, var):
        A = basis
        C = np.linalg.inv(A.T.dot(A / var))
        result = np.linalg.inv(A.T.dot(C.dot(A)))
        result[np.diag_indices_from(result)] -= 1 / var
        result = np.linalg.inv(result)
        result[np.diag_indices_from(result)] += var
        return np.linalg.inv(result)


@attr.s(frozen=True, kw_only=True)
class TNS(Component):
    x: np.ndarray = attr.ib()
    c_terms: int = attr.ib(default=5)
    field_freq: np.ndarray | None = attr.ib(None)
    freq_range: tuple[float, float] = attr.ib()
    fiducial: np.typing.ArrayLike = attr.ib()

    @freq_range.default
    def _fr_default(self):
        return (self.x.min(), self.x.max())

    @freq_range.validator
    def _fr_validator(self, att, val):
        if len(val) != 2:
            raise ValueError("freq_range must be a 2-tuple of floats!")
        if val[1] <= val[0]:
            raise ValueError(f"freq_range must in the form (min, max), got: {val}")

    @fiducial.default
    def _fid_default(self):
        return [1500] + [0] * (self.c_terms - 1)

    @fiducial.validator
    def _fid_validator(self, att, val):
        if len(val) != self.c_terms:
            raise ValueError(
                f"fiducial must be a sequence of length c_terms ({self.c_terms}), got {len(val)}."
            )

    @cached_property
    def provides(self):
        if self.field_freq is None:
            return ["tns"]
        return ["tns", "tns_field"]

    @cached_property
    def base_parameters(self):
        return ParameterVector(
            "t_lns",
            fiducial=self.fiducial,
            length=self.c_terms,
            latex=r"T^{\rm L+NS}_{%s}",
        ).get_params()

    @cached_property
    def model(self):
        return Polynomial(
            n_terms=self.c_terms,
            transform=UnitTransform(range=self.freq_range),
            parameters=[p.fiducial for p in self.active_params],
        ).at(x=self.x)

    @cached_property
    def field_model(self):
        return Polynomial(
            n_terms=self.c_terms,
            transform=UnitTransform(range=self.freq_range),
            parameters=[p.fiducial for p in self.active_params],
        ).at(x=self.field_freq)

    def calculate(self, ctx, **params):
        tns = self.model(parameters=list(params.values()))

        if self.field_freq is not None:
            tns_field = self.field_model(parameters=list(params.values()))
            return tns, tns_field
        return tns


@attr.s(frozen=True, kw_only=True)
class S11Systematic(Component):
    freq: np.ndarray = attr.ib()
    measured: np.ndarray = attr.ib()
    _scale_model: Model = attr.ib()
    _delay_model: Model = attr.ib()

    @cached_property
    def scale_model(self):
        return self._scale_model.at(x=self.freq)

    @cached_property
    def delay_model(self):
        return self._delay_model.at(x=self.freq)

    @cached_property
    def base_parameters(self):
        return tuple(
            Parameter(f"logscale_{i}", fiducial=0, latex=rf"A^\Gamma_{i}")
            for i in range(self.scale_model.n_terms)
        ) + tuple(
            Parameter(f"delay_{i}", fiducial=0, latex=rf"tau^\Gamma_{i}")
            for i in range(self.delay_model.n_terms)
        )

    @cached_property
    def provides(self):
        return {f"{self.name}_gamma"}

    def calculate(self, ctx, **params):
        s = [params[f"logscale_{i}"] for i in range(self.scale_model.n_terms)]
        d = [params[f"delay_{i}"] for i in range(self.delay_model.n_terms)]

        return (self.measured * 10 ** self.scale_model(parameters=s)) * np.exp(
            -1j * self.delay_model(parameters=d) / 1000 * self.freq
        )


@attr.s(frozen=True, kw_only=True)
class NoiseWaveCoefficients(Component):
    # Requires two sub-components: a source s11 and an LNA s11
    source_names = attr.ib()
    as_dict = attr.ib(False)

    @cached_property
    def provides(self):
        return {"K"}

    @profile
    def calculate(self, ctx, **params):
        if not self.as_dict:
            return np.hstack(
                tuple(
                    nw.get_K(
                        gamma_rec=ctx["rcv_gamma" if src != "ant" else "rcvant_gamma"],
                        gamma_ant=ctx[f"{src}_gamma"],
                    )
                    for src in self.source_names
                )
            )
        return {
            src: nw.get_K(
                gamma_rec=ctx["rcv_gamma" if src != "ant" else "rcvant_gamma"],
                gamma_ant=ctx[f"{src}_gamma"],
            )
            for src in self.source_names
        }


@attr.s(frozen=True, kw_only=True)
class NoiseWaveLikelihood:
    nw_model: nw.NoiseWaves = attr.ib()
    data: dict = attr.ib()
    sig_by_tns: bool = attr.ib(default=True)
    t_ns_params: ParamVec = attr.ib()
    t_ns_freq_range: tuple[float, float] = attr.ib()
    derived: tuple[str] = attr.ib(factory=tuple)
    s11_systematics: Sequence[S11Systematic] = attr.ib(())

    @t_ns_freq_range.default
    def _tns_fr_default(self):
        return (self.nw_model.freq.min(), self.nw_model.freq.max())

    @t_ns_params.default
    def _tns_default(self) -> ParamVec:
        return ParamVec(
            "t_lns",
            length=self.nw_model.c_terms,
            fiducial=[1000] + [0] * (self.nw_model.c_terms - 1),
        )

    @cached_property
    def t_ns_model(self):
        if hasattr(self.nw_model.freq, "unit"):
            f = self.nw_model.freq.to_value("MHz")
        else:
            f = self.nw_model.freq
        return TNS(
            x=f,
            c_terms=self.nw_model.c_terms,
            params=self.t_ns_params.get_params(),
            fiducial=list(self.t_ns_params.fiducial),
            freq_range=self.t_ns_freq_range,
        )

    @cached_property
    def partial_linear_model(self):
        return PartialLinearModel(
            linear_model=self.nw_model.linear_model,
            data=self.data,
            components=(self.t_ns_model, *self.s11_systematics),
            data_func=self.transform_data,
            variance_func=self.transform_variance if self.sig_by_tns else None,
            derived=tuple(
                d if hasattr(PartialLinearModel, d) else getattr(self, d) for d in self.derived
            ),
            basis_func=self.apply_s11_systematics if self.s11_systematics else None,
        )

    @classmethod
    def transform_data(cls, ctx: dict, data: dict):
        n = len(data["q"]) // len(ctx["tns"])
        tns = np.concatenate((ctx["tns"],) * n)
        k0 = ctx["K"][0] if "K" in ctx else data["k0"]
        return data["q"] * tns - k0 * data["T"]

    @classmethod
    def transform_variance(cls, ctx: dict, data: dict):
        n = len(data["q"]) // len(ctx["tns"])
        tns = np.concatenate((ctx["tns"],) * n)
        return data["data_variance"] * tns**2

    @classmethod
    @profile
    def apply_s11_systematics(cls, linear_model, ctx, data):
        K = ctx["K"]

        extra_basis = {"tunc": K[1], "tcos": K[2], "tsin": K[3]}
        m = linear_model.model
        repeater = (m.tunc.n_terms, m.tcos.n_terms, m.tsin.n_terms)
        if "tload" in m.extra_basis:
            extra_basis["tload"] = m.extra_basis["tload"]
            repeater = (*repeater, m.tload.n_terms)

        # We do the following (updating the basis functions manually) for speed
        # If we don't pass "init_basis" explicitly, the whole basis function has to
        # be recomputed, whereas if we just multiply the new/old K, we get a decent
        # speedup.
        K = np.vstack((K[1:], np.atleast_2d(m.extra_basis["tload"])))
        new_extra_basis = np.repeat(K, repeater, axis=0)

        return attr.evolve(
            linear_model,
            init_basis=data["raw_basis"] * new_extra_basis,
            model=attr.evolve(linear_model.model, extra_basis=extra_basis),
        )

    @classmethod
    def from_calobs(
        cls,
        calobs,
        cterms=None,
        wterms=None,
        sig_by_sigq=True,
        sources=None,
        as_sim=None,
        reweight_sources: bool = False,
        reweight_frequencies: bool = False,
        s11_systematic_params=None,
        cable_noise_factor=1,
        source_factors=None,
        include_antsim: bool = False,
        seed: int = 1234,
        add_noise: bool = True,
        **kwargs,
    ):
        if sources is None:
            sources = tuple(calobs.loads.keys())

        s11_systematic_params = s11_systematic_params or {}

        as_sim = as_sim or []
        if as_sim == "all":
            as_sim = list(calobs.loads.keys())

        loads = {src: load for src, load in calobs.loads.items() if src in sources}
        if include_antsim:
            for name in calobs.metadata["io"].s11.simulators:
                loads[name] = calobs.new_load(name, io_obj=calobs.metadata["io"])

        sources = tuple(loads.keys())

        # Source Factors define how much "extra" weight to place on each source.
        if source_factors is None:
            source_factors = (1,) * len(loads)

        nw_model = nw.NoiseWaves.from_calobs(calobs, loads=loads, cterms=cterms, wterms=wterms)

        raw_bases = nw_model.get_linear_model(with_k=False).basis

        freq = calobs.freq.freq.to_value("MHz")
        if s11_systematic_params:
            lna_delay_model = s11_systematic_params.get("rcv", {}).pop(
                "delay_model", Polynomial(n_terms=1)
            )
            lna_scale_model = s11_systematic_params.get("rcv", {}).pop(
                "scale_model", Polynomial(n_terms=1)
            )

            lna = S11Systematic(
                freq=freq,
                measured=calobs.receiver.s11_model(freq),
                params=s11_systematic_params.get("rcv", {}),
                delay_model=lna_delay_model,
                scale_model=lna_scale_model,
                name="rcv",
            )

            s11_systematics = tuple(
                S11Systematic(
                    freq=freq,
                    measured=load.reflections.s11_model(freq),
                    params=s11_systematic_params.get(src, {}),
                    delay_model=s11_systematic_params.get(src, {}).pop(
                        "delay_model", Polynomial(n_terms=1)
                    ),
                    scale_model=s11_systematic_params.get(src, {}).pop(
                        "scale_model", Polynomial(n_terms=1)
                    ),
                    name=src,
                )
                for src, load in loads.items()
            )

            noise_wave_coeffs = (
                NoiseWaveCoefficients(source_names=sources, components=s11_systematics + (lna,)),
            )
        else:
            noise_wave_coeffs = ()

        ks = {
            name: load.reflections.get_k_matrix(calobs.receiver, freq=nw_model.freq)
            for name, load in loads.items()
        }
        k0 = np.concatenate(tuple(ks[src][0] for src in loads))

        if as_sim:
            np.random.seed(seed)

        data = {
            "q": np.concatenate(
                tuple(
                    simulate_q_from_calobs(calobs, name)
                    + (
                        np.random.normal(
                            scale=np.sqrt(load.spectrum.variance_Q / load.spectrum.n_integrations)
                        )
                        if add_noise
                        else 0
                    )
                    if name in as_sim
                    else load.spectrum.averaged_Q
                    for name, load in loads.items()
                )
            ),
            "T": np.concatenate(
                tuple(load.temp_ave * np.ones(calobs.freq.n) for load in loads.values())
            ),
            "k0": k0,
            "gamma_src": {name: source.s11_model(freq) for name, source in loads.items()},
            "gamma_rcv": calobs.receiver.s11_model(freq),
            "raw_basis": raw_bases,
        }

        overall_mean = np.mean([np.mean(load.spectrum.variance_Q) for load in loads.values()])

        if sig_by_sigq:
            if reweight_sources:
                data["data_variance"] = np.concatenate(
                    tuple(
                        overall_mean * load.spectrum.variance_Q / np.mean(load.spectrum.variance_Q)
                        for load in loads.values()
                    )
                )
            elif reweight_frequencies:
                data["data_variance"] = np.concatenate(
                    tuple(
                        np.mean(load.spectrum.variance_Q) * np.ones_like(load.spectrum.variance_Q)
                        for load in loads.values()
                    )
                )
            else:
                data["data_variance"] = np.concatenate(
                    tuple(
                        load.spectrum.variance_Q / load.spectrum.n_integrations
                        for load in loads.values()
                    )
                )
        else:
            data["data_variance"] = np.concatenate(
                tuple(
                    overall_mean * np.ones_like(load.spectrum.variance_Q) for load in loads.values()
                )
            )

        noise_factor = np.concatenate(
            tuple(
                np.ones_like(load.spectrum.averaged_Q)
                * (cable_noise_factor if name in ["open", "short"] else 1)
                * source_factors[i]
                for i, (name, load) in enumerate(loads.items())
            )
        )

        data["data_variance"] *= noise_factor / np.mean(noise_factor)

        return cls(nw_model=nw_model, data=data, s11_systematics=noise_wave_coeffs, **kwargs)

    def get_cal_curves(
        self, params: Sequence | None = None, freq=None, sample=True
    ) -> dict[str, np.ndarray]:
        fit = self.partial_linear_model.reduce_model(params=params)[0]

        if freq is None:
            freq = self.nw_model.freq

        model = fit.fit.model

        linear_params = fit.get_sample()[0] if sample else fit.model_parameters

        out = {
            name: model.get_model(name, x=freq, parameters=linear_params) for name in model.models
        }

        pp = [
            params[i]
            for i, p in enumerate(self.partial_linear_model.child_active_params)
            if p.name.startswith("t_lns")
        ]
        out["tns"] = self.t_ns_model.model.model(x=freq, parameters=pp)
        out["params"] = linear_params
        return out

    def get_linear_coefficients(
        self,
        freq,
        labcal,
        params=None,
        ctx=None,
        fit=None,
        linear_params=None,
        load=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the linear coefficients that convert uncalibrated to calibrated temp.

        The equation is T_cal = a*T_uncal + b
        """
        pp = []
        for i, p in enumerate(self.partial_linear_model.child_active_params):
            if p.name.startswith("t_lns"):
                pp.append(params[i])
        tns = self.t_ns_model.model.model(x=freq, parameters=pp)

        model = self.nw_model.linear_model.model
        if linear_params is not None:
            params = linear_params
        else:
            ctx = self.partial_linear_model.get_ctx(params=params)
            if fit is None:
                fit = self.partial_linear_model.reduce_model(ctx=ctx)[0]
            params = fit.model_parameters

        if load is None:
            gamma_ant = labcal.antenna_s11_model(freq)
        elif load.load_name + "_gamma" in ctx:
            gamma_ant = ctx[load.load_name + "_gamma"]
        else:
            gamma_ant = load.reflections.s11_model(freq)

        if "rcv_gamma" in ctx:
            rcv = ctx["rcv_gamma"]

            if len(freq) != len(rcv):
                re = Spline(self.nw_model.freq, np.real(rcv))(freq)
                im = Spline(self.nw_model.freq, np.imag(rcv))(freq)
                rcv = re + 1j * im
        else:
            rcv = labcal.calobs.receiver_s11(freq * u.MHz)

        return nw.get_linear_coefficients(
            gamma_ant=gamma_ant,
            gamma_rec=rcv,
            sca=tns / labcal.calobs.t_load_ns,
            off=labcal.calobs.t_load - model.get_model("tload", x=freq, parameters=params),
            t_unc=model.get_model("tunc", x=freq, parameters=params),
            t_cos=model.get_model("tcos", x=freq, parameters=params),
            t_sin=model.get_model("tsin", x=freq, parameters=params),
            t_load=labcal.calobs.t_load,
        )

    # Start derived parameters
    @staticmethod
    def tunchat(model, ctx, **params):
        return model[0].fit.model.tunc.parameters

    @staticmethod
    def tcoshat(model, ctx, **params):
        return model[0].fit.model.tcos.parameters

    @staticmethod
    def tsinhat(model, ctx, **params):
        return model[0].fit.model.tsin.parameters

    @staticmethod
    def tloadhat(model, ctx, **params):
        return model[0].fit.model.tload.parameters

    @staticmethod
    def linear(model, ctx, **params):
        fit = model[0]
        return fit.get_sample()

    def rms_parts(self, model, ctx, **params):
        fit, _, var = model

        rms_parts = []
        n = 0
        for d in self.data.values():
            v = var[n : n + len(d)]
            r = fit.residual[n : n + len(d)]

            mask = (~np.isinf(v)) & (~np.isnan(v))

            rms_parts.append(np.nansum(r[mask] ** 2 / v[mask]))
            n += len(d)

        return rms_parts

    def lnl_at_calobs(self, calobs: CalibrationObservation):
        """Compute the (unmarginalized) likelihood at the same parameters as a calobs."""
        p = np.concatenate(
            (
                calobs.Tunc_poly.coeffs[::-1],
                calobs.Tcos_poly.coeffs[::-1],
                calobs.Tsin_poly.coeffs[::-1],
                calobs.t_load - calobs.C2_poly.coeffs[::-1],
            )
        )

        nl = calobs.C1_poly.coeffs[::-1] * calobs.t_load_ns

        return self.partial_linear_model.get_unmarginalized_lnl(p, nl)


@attr.s(frozen=True, kw_only=True)
class NoiseWavesPlusFG:
    freq: tp.FreqType = attr.ib()
    _gamma_src: dict[str, np.ndarray] = attr.ib()
    gamma_ant: np.ndarray = attr.ib()
    gamma_rec: Callable = attr.ib()
    field_freq: tp.FreqType = attr.ib()
    c_terms: int = attr.ib(default=5)
    w_terms: int = attr.ib(default=6)
    fg_model: Model = attr.ib(default=LinLog(n_terms=5))
    parameters: Sequence[float] | None = attr.ib(default=None)
    loss: float | np.ndarray = attr.ib(default=1.0)
    bm_corr: float | np.ndarray = attr.ib(default=1.0)

    @field_freq.default
    def _ff_default(self) -> np.ndarray:
        return self.freq

    @cached_property
    def gamma_src(self):
        return {**self._gamma_src, "ant": self.gamma_ant}

    def _freq(self, src: str):
        if src == "ant":
            return self.field_freq
        return self.freq

    @cached_property
    def src_names(self) -> list[str]:
        """List of names of inputs sources (eg. ambient, hot_load, open, short)."""
        return list(self.gamma_src.keys())

    @cached_property
    def linear_model(self):
        return self.get_linear_model()

    def get_linear_model(self, with_k: bool = True) -> CompositeModel:
        """The actual composite linear model object associated with the noise waves."""
        # K should be a an array of shape (Nsrc Nnu x Nnoisewaveterms)
        if with_k:
            K = np.hstack(
                tuple(
                    nw.get_K(
                        gamma_rec=self.gamma_rec(self._freq(name)),
                        gamma_ant=self.gamma_src[name](self._freq(name)),
                    )
                    for name in self.src_names
                )
            )
            # K[0] multiples the fg, but not the other models.
            K[0][: len(self.freq) * (len(self.gamma_src) - 1)] = 0.0
            K[0][-len(self.field_freq) :] *= self.loss * self.bm_corr

        x = np.concatenate(
            (
                np.tile(self.freq.to_value("MHz"), len(self.src_names) - 1),
                self.field_freq.to_value("MHz"),
            )
        )

        transform = UnitTransform(
            range=[self.freq.to_value("MHz").min(), self.freq.to_value("MHz").max()]
        )

        models = {
            "tunc": Polynomial(
                n_terms=self.w_terms,
                parameters=self.parameters[: self.w_terms] if self.parameters is not None else None,
                transform=transform,
            ),
            "tcos": Polynomial(
                n_terms=self.w_terms,
                parameters=self.parameters[self.w_terms : 2 * self.w_terms]
                if self.parameters is not None
                else None,
                transform=transform,
            ),
            "tsin": Polynomial(
                n_terms=self.w_terms,
                parameters=self.parameters[2 * self.w_terms : 3 * self.w_terms]
                if self.parameters is not None
                else None,
                transform=transform,
            ),
            "tload": Polynomial(
                n_terms=self.c_terms,
                parameters=(
                    self.parameters[3 * self.w_terms : 3 * self.w_terms + self.c_terms]
                    if self.parameters is not None
                    else None
                ),
                transform=transform,
            ),
            "fg": self.fg_model,
        }

        if with_k:
            extra_basis = {
                "tunc": K[1],
                "tcos": K[2],
                "tsin": K[3],
                "tload": -1 * np.ones(len(x)),
                "fg": K[0],
            }
        else:
            extra_basis = {}

        return CompositeModel(models=models, extra_basis=extra_basis).at(x=x)

    def _get_idx(self, src: str):
        if src == "ant":
            return slice(-len(self.field_freq), None, None)
        return slice(
            self.src_names.index(src) * len(self.freq),
            (1 + self.src_names.index(src)) * len(self.freq),
            None,
        )

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
            return out[self._get_idx(src)]
        return out[: len(self.linear_model.x)]

    def get_full_model(self, src: str, parameters: Sequence | None = None) -> np.ndarray:
        """Get the full model (all noise-waves) for a particular input source."""
        out = self.linear_model(parameters=parameters)
        return out[self._get_idx(src)]

    def get_fitted(self, data: np.ndarray, weights: np.ndarray | None = None) -> nw.NoiseWaves:
        """Get a new noise wave model with fitted parameters."""
        fit = self.linear_model.fit(ydata=data, weights=weights)
        return attr.evolve(self, parameters=fit.model_parameters)

    def with_params_from_calobs(self, calobs, cterms=None, wterms=None, fg_terms=None):
        cterms = cterms or calobs.cterms
        wterms = wterms or calobs.wterms

        def modify(thing, n):
            if len(thing) < wterms:
                return thing + [0] * (n - len(thing))
            if len(thing) > wterms:
                return thing[:n]
            return thing

        tu = modify(calobs.Tunc_poly.coefficients[::-1].tolist(), wterms)
        tc = modify(calobs.Tcos_poly.coefficients[::-1].tolist(), wterms)
        ts = modify(calobs.Tsin_poly.coefficients[::-1].tolist(), wterms)

        c2 = (-calobs.C2_poly.coefficients[::-1]).tolist()
        c2[0] += calobs.t_load
        c2 = modify(c2, cterms)

        if fg_terms is None:
            fg_terms = [0] * self.fg_model.n_terms

        return attr.evolve(self, parameters=tu + tc + ts + c2 + list(fg_terms))

    @classmethod
    def from_labcal(
        cls,
        labcal,
        calobs,
        fg_model=LinLog(n_terms=5),
        loads: dict | None = None,
        cterms=None,
        wterms=None,
        **kwargs,
    ) -> NoiseWavesPlusFG:
        """Initialize a noise wave model from a calibration observation."""
        cterms = cterms or calobs.cterms
        wterms = wterms or calobs.wterms

        if loads is None:
            loads = calobs.loads

        gamma_src = {name: load.reflections.s11_model for name, load in loads.items()}

        out = cls(
            freq=calobs.freq.freq,
            gamma_src=gamma_src,
            gamma_rec=calobs.receiver.s11_model,
            gamma_ant=labcal.antenna_s11_model,
            c_terms=cterms,
            w_terms=wterms,
            fg_model=fg_model,
            **kwargs,
        )
        return out.with_params_from_calobs(calobs, cterms, wterms, fg_model.parameters)

    def __call__(self, **kwargs) -> np.ndarray:
        """Call the underlying linear model."""
        return self.linear_model(**kwargs)


@attr.s(frozen=True, kw_only=True)
class DataCalibrationLikelihood:
    nwfg_model = attr.ib()
    data: dict = attr.ib()
    eor_components = attr.ib()
    t_ns_params: ParamVec | None = attr.ib(None)
    verbose: bool = attr.ib(False)
    subtract_fiducial: bool = attr.ib(False)
    save_linear_params: bool = attr.ib(True)
    save_sampled_linear_params: bool = attr.ib(True)
    derived: list[str | Callable] = attr.ib(factory=list)
    s11_systematics: Sequence[S11Systematic] = attr.ib(())

    @eor_components.default
    def _eorcmp(self):
        return (AbsorptionProfile(freqs=self.nwfg_model.freq, params=("A", "w", "tau", "nu0")),)

    @cached_property
    def src_names(self) -> tuple[str]:
        return tuple(self.data["q"].keys())

    @cached_property
    def t_ns_model(self):
        if self.t_ns_params is None:
            t_ns_params = ParamVec("t_lns", length=self.nwfg_model.c_terms)
        else:
            t_ns_params = self.t_ns_params
        return TNS(
            x=self.nwfg_model.freq.to_value("MHz"),
            field_freq=self.nwfg_model.field_freq,
            c_terms=self.nwfg_model.c_terms,
            params=t_ns_params.get_params(),
        )

    # Start derived parameters
    @staticmethod
    def tunchat(model, ctx, **params):
        return model[0].fit.model.tunc.parameters

    @staticmethod
    def tcoshat(model, ctx, **params):
        return model[0].fit.model.tcos.parameters

    @staticmethod
    def tsinhat(model, ctx, **params):
        return model[0].fit.model.tsin.parameters

    @staticmethod
    def tloadhat(model, ctx, **params):
        return model[0].fit.model.tload.parameters

    @staticmethod
    def tfghat(model, ctx, **params):
        return model[0].fit.model.fg.parameters

    @staticmethod
    def linear(model, ctx, **params):
        fit = model[0]
        return fit.get_sample()

    @cached_property
    def partial_linear_model(self):
        derived = []
        if self.save_linear_params:
            derived.extend(["tunchat", "tcoshat", "tsinhat", "tloadhat", "tfghat"])
        if self.save_sampled_linear_params:
            derived.append("linear")

        derived.extend(self.derived)

        return PartialLinearModel(
            linear_model=self.nwfg_model.linear_model,
            data=self.data,
            components=(self.t_ns_model, *self.eor_components, *self.s11_systematics),
            data_func=self.transform_data,
            variance_func=self.transform_variance,
            verbose=self.verbose,
            subtract_fiducial=self.subtract_fiducial,
            derived=tuple(
                d if hasattr(PartialLinearModel, d) else getattr(self, d) for d in derived
            ),
            basis_func=self.apply_s11_systematics if self.s11_systematics else None,
        )

    def transform_data(self, ctx: dict, data: dict):
        tns = ctx["tns"]
        Tant = ctx["eor_spectrum"]

        k0 = {name: kk[0] for name, kk in ctx["K"].items()} if "K" in ctx else data["k0"]

        out = []
        for src in self.src_names:
            if src == "ant":
                out.append(
                    ctx["tns_field"] * data["q"]["ant"]
                    - k0["ant"] * (Tant * data["loss"] * data["bm_corr"] + data["loss_temp"])
                )
            else:
                Tsrc = data["T"][src]
                out.append(tns * data["q"][src] - k0[src] * Tsrc)

        return np.concatenate(out)

    def transform_variance(self, ctx: dict, data: dict):
        tns = ctx["tns"]
        field_tns = ctx["tns_field"]

        return np.concatenate(
            [
                data["data_variance"][src] * (field_tns**2 if src == "ant" else tns**2)
                for src in self.src_names
            ]
        )

    @classmethod
    @profile
    def apply_s11_systematics(cls, linear_model, ctx, data):
        K = ctx["K"]
        m = linear_model.model

        extra_basis = {
            "tunc": np.concatenate([kk[1] for kk in K.values()]),
            "tcos": np.concatenate([kk[2] for kk in K.values()]),
            "tsin": np.concatenate([kk[3] for kk in K.values()]),
            "tload": m.extra_basis["tload"],
            "fg": np.concatenate([kk[0] for kk in K.values()]),
        }
        repeater = (
            m.tunc.n_terms,
            m.tcos.n_terms,
            m.tsin.n_terms,
            m.tload.n_terms,
            m.fg.n_terms,
        )

        # We do the following (updating the basis functions manually) for speed
        # If we don't pass "init_basis" explicitly, the whole basis function has to
        # be recomputed, whereas if we just multiply the new/old K, we get a decent
        # speedup.
        K = np.vstack(extra_basis.values())
        new_extra_basis = np.repeat(K, repeater, axis=0)

        return attr.evolve(
            linear_model,
            init_basis=data["raw_basis"] * new_extra_basis,
            model=attr.evolve(linear_model.model, extra_basis=extra_basis),
        )

    @classmethod
    def from_labcal(
        cls,
        labcal,
        calobs,
        q_ant,
        qvar_ant,
        loads: dict | None = None,
        loss: float | np.ndarray = 1.0,
        loss_temp: float | np.ndarray = 0.0,
        fg_model=LinLog(n_terms=5),
        as_sim: tuple[str] = (),
        cal_noise="data",
        field_freq: np.ndarray = attr.NOTHING,
        bm_corr: float | np.ndarray = 1.0,
        s11_systematic_params=None,
        cterms=None,
        wterms=None,
        add_noise=True,
        seed=1234,
        **kwargs,
    ):
        if loads is None:
            loads = calobs.loads

        if as_sim == "all":
            as_sim = tuple(calobs.loads.keys())

        nwfg_model = NoiseWavesPlusFG.from_labcal(
            labcal,
            calobs,
            loads=loads,
            fg_model=fg_model,
            field_freq=field_freq,
            loss=loss,
            bm_corr=bm_corr,
            cterms=cterms,
            wterms=wterms,
        )

        freq = calobs.freq.freq.to_value("MHz")
        if s11_systematic_params:
            lna = S11Systematic(
                freq=freq,
                measured=calobs.receiver.s11_model(freq),
                params=s11_systematic_params.get("rcv", {}),
                name="rcv",
            )
            lna_ant = S11Systematic(
                freq=field_freq.to_value("MHz"),
                measured=calobs.receiver.s11_model(field_freq.to_value("MHz")),
                params=s11_systematic_params.get("rcv", {}),
                name="rcvant",
            )

            s11_systematics = tuple(
                S11Systematic(
                    freq=freq,
                    measured=load.reflections.s11_model(freq),
                    params=s11_systematic_params.get(src, {}),
                    name=src,
                )
                for src, load in loads.items()
            )

            ant_s11 = S11Systematic(
                freq=field_freq.to_value("MHz"),
                measured=labcal.antenna_s11_model(field_freq),
                params=s11_systematic_params.get("ant", {}),
                name="ant",
            )

            noise_wave_coeffs = (
                NoiseWaveCoefficients(
                    source_names=tuple(loads.keys()) + ("ant",),
                    components=s11_systematics + (lna, lna_ant, ant_s11),
                    as_dict=True,
                ),
            )
        else:
            noise_wave_coeffs = ()

        k0 = {
            src: nw.get_K(
                gamma_ant=gamma_src(nwfg_model._freq(src)),
                gamma_rec=nwfg_model.gamma_rec(nwfg_model._freq(src)),
            )[0]
            for src, gamma_src in nwfg_model.gamma_src.items()
        }

        q = {
            name: simulate_q_from_calobs(calobs, name)
            if name in as_sim
            else load.spectrum.averaged_Q
            for name, load in loads.items()
        }

        q["ant"] = q_ant

        T = {name: load.temp_ave * np.ones(labcal.calobs.freq.n) for name, load in loads.items()}
        qvar = {"ant": qvar_ant}

        if cal_noise == "data" or isinstance(cal_noise, dict):
            qvar.update(
                {
                    name: load.spectrum.variance_Q / load.spectrum.n_integrations
                    for name, load in loads.items()
                }
            )
        else:
            qvar.update({name: cal_noise * np.ones(calobs.freq.n) for name in loads})

        if add_noise:
            np.random.seed(seed)
            for k in q:
                if k in as_sim:
                    if isinstance(cal_noise, dict):
                        q[k] += cal_noise[k]
                    else:
                        q[k] += np.random.normal(scale=np.sqrt(qvar[k]))

        raw_bases = nwfg_model.get_linear_model(with_k=False).basis

        data = {
            "q": q,
            "T": T,
            "k0": k0,
            "data_variance": qvar,
            "loss": loss,
            "bm_corr": bm_corr,
            "loss_temp": loss_temp,
            "gamma_src": {name: source.s11_model(freq) for name, source in loads.items()},
            "gamma_rcv": calobs.receiver.s11_model(freq),
            "raw_basis": raw_bases,
        }

        if not len(nwfg_model.field_freq) == len(q["ant"]) == len(qvar["ant"]):
            raise ValueError("field_freq, q_ant and qvar_ant must be of the same shape.")

        return cls(
            nwfg_model=nwfg_model,
            data=data,
            s11_systematics=noise_wave_coeffs,
            **kwargs,
        )

    def get_linear_coefficients(
        self, params=None, ctx=None, fit=None, linear_params=None, src="ant"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the linear coefficients that convert Q into a calibrated temperature.

        The equation is T_cal = a*Q + b

        This takes the input parameters (for TNS and T21), and computes the best-fit
        for the linear parameters.
        """
        if ctx is None:
            ctx = self.partial_linear_model.get_ctx(params=params)

        if fit is None and linear_params is None:
            fit = self.partial_linear_model.reduce_model(params=params)[0]
            params = fit.model_parameters
            model = fit.fit.model
            x = fit.fit.x
        elif fit is None:
            model = self.nwfg_model.linear_model.model
            x = self.nwfg_model.linear_model.x
            params = linear_params
        elif linear_params is None:
            model = fit.fit.model
            x = fit.fit.x
            params = fit.model_parameters
        else:
            x = fit.fit.x
            model = fit.fit.model
            params = linear_params

        idx = self.nwfg_model._get_idx(src)

        if src == "ant":
            a = ctx["tns_field"] / self.data["k0"]["ant"]
        else:
            a = ctx["tns"] / self.data["k0"][src]

        b = (
            -(
                model.get_model("tunc", x=x, with_extra=True, parameters=params)
                + model.get_model("tcos", x=x, with_extra=True, parameters=params)
                + model.get_model("tsin", x=x, with_extra=True, parameters=params)
                + model.get_model("tload", x=x, with_extra=True, parameters=params)
            )[idx]
            / self.data["k0"][src]
        )
        return a, b

    def recalibrated_sky_temp(
        self, params=None, ctx=None, fit=None, linear_params=None, a=None, b=None
    ) -> np.ndarray:
        """Get calibrated temperature of the data at a certain set of parameters.

        This takes the input parameters (for TNS and T21), computes the best-fit
        for the linear parameters, and applies the resulting calibration to the
        input field data.

        i.e. it gets (Tns*Q - K*Tnw)/K0
        """
        if a is None or b is None:
            a, b = self.get_linear_coefficients(
                params=params, ctx=ctx, fit=fit, linear_params=linear_params, src="ant"
            )

        return (a * self.data["q"]["ant"] + b - self.data["loss_temp"]) / (
            self.data["bm_corr"] * self.data["loss"]
        )

    def recalibrated_source_temp(self, src: str, params=None) -> np.ndarray:
        a, b = self.get_linear_coefficients(params=params, src=src)
        return a * self.data["q"][src] + b

    def sky_temp_model(self, params=None) -> tuple[np.ndarray, np.ndarray]:
        fit, *_ = self.partial_linear_model.reduce_model(params=params)
        freq = self.nwfg_model._freq("ant")

        fg = fit.fit.model.fg(x=freq)  # there's no K[0] or loss in this
        ctx = self.partial_linear_model.get_ctx(params=params)

        eor = ctx["eor_spectrum"]

        return fg, eor

    def cal_temp_model(self, src):
        return self.data["T"][src]

    def plot_calibrated_temps(self, params=None, fig=None, ax=None, labcal=None, fid_resid=None):
        if labcal is not None:
            calobs = labcal.calobs

        if fig is None:
            fig, ax = plt.subplots(
                5,
                2,
                sharex=True,
                gridspec_kw={"wspace": 0.05, "hspace": 0.05},
                figsize=(5, 10),
            )

        for i, src in enumerate(self.src_names):
            if src == "ant":
                recal_temp = self.recalibrated_sky_temp(params=params)
                eor, fg = self.sky_temp_model(params=params)
                recal_temp_model = eor + fg
            else:
                recal_temp = self.recalibrated_source_temp(src=src, params=params)
                recal_temp_model = self.cal_temp_model(src)
            freq = self.nwfg_model._freq(src)

            ax[i, 0].plot(freq, recal_temp, label="Calibrated Data")
            ax[i, 0].plot(freq, recal_temp_model, label="Model", ls="--")
            ax[i, 0].set_ylabel(src)
            ax[i, 1].plot(freq, recal_temp - recal_temp_model)

            if labcal is not None and src != "ant":
                fid_data = calobs.calibrate(src, q=self.data["q"][src])
                ax[i, 0].plot(freq, fid_data, label="Fiducial Cal Data")
                ax[i, 1].plot(
                    freq,
                    fid_data - recal_temp_model,
                    label="Fid Resid",
                )

            elif labcal is not None and src == "ant":
                fid_data = (
                    labcal.calibrate_q(self.data["q"]["ant"], freq=freq) - self.data["loss_temp"]
                ) / (self.data["bm_corr"] / self.data["loss"])
                ax[i, 0].plot(freq, fid_data)
                ax[i, 1].plot(freq, fid_data - recal_temp, label="Recal vs Labcal")
                if fid_resid is not None:
                    ax[i, 1].plot(freq, fid_resid)

        ax[0, 0].legend()
        ax[0, 1].legend()
        ax[-1, 1].legend()
        return fig, ax

    def plot_models(self, params=None, calobs=None, fig=None, ax=None, plot_zscore=True):
        fit, data, var = self.partial_linear_model.reduce_model(params=params)

        if not plot_zscore:
            var = np.ones_like(var)

        if fig is None:
            fig, ax = plt.subplots(
                5,
                2,
                sharex=True,
                gridspec_kw={"wspace": 0.05, "hspace": 0.05},
                figsize=(5, 10),
            )

        n = 0
        mdata = fit.evaluate()
        for i, src in enumerate(self.src_names):
            freq = self.nwfg_model._freq(src)
            ax[i, 0].plot(freq, data[n : n + len(freq)], label="Recalibrated Data")
            ax[i, 0].plot(freq, mdata[n : n + len(freq)], label="best-Fit Linear Model", ls="--")
            ax[i, 0].set_ylabel(src)
            ax[i, 1].plot(
                freq,
                (data[n : n + len(freq)] - mdata[n : n + len(freq)])
                / np.sqrt(var[n : n + len(freq)]),
                label="Recal Resid",
            )

            if calobs is not None and src != "ant":
                fid_data = (
                    calobs.C1() * calobs.t_load_ns * self.data["q"][src]
                    - self.data["k0"][src] * self.data["T"][src]
                )
                ax[i, 0].plot(calobs.freq.freq, fid_data, label="Fiducial Cal Data")
                fid_model_q = (
                    calobs.decalibrate(temp=self.data["T"][src], load=src) - calobs.t_load
                ) / calobs.t_load_ns
                fid_model = (
                    calobs.C1() * calobs.t_load_ns * fid_model_q
                    - self.data["k0"][src] * self.data["T"][src]
                )
                if plot_zscore:
                    fid_var = (calobs.C1() * calobs.t_load_ns) ** 2 * self.data["data_variance"][
                        src
                    ]
                else:
                    fid_var = 1

                ax[i, 0].plot(calobs.freq.freq, fid_model, label="Fiducial Cal Model")

                ax[i, 1].plot(
                    calobs.freq.freq,
                    (fid_data - fid_model) / np.sqrt(fid_var),
                    label="Fid Resid",
                )

            n += len(freq)

        ax[0, 0].legend()
        ax[0, 1].legend()
        return fig, ax


@attr.s(frozen=True, kw_only=True)
class LinearFG:
    """Classic traditional EoR fit to a calibrated sky spectrum.

    This defines a ``partial_linear_model`` that simply fits the EoR + FG, where the FG
    is assumed to be a linear model. While this is not the ultimate in flexibility,
    since the FG aren't always linear, it is fast.
    """

    freq: np.ndarray = attr.ib()
    t_sky: np.ndarray = attr.ib()
    var: np.ndarray = attr.ib()
    fg: Model = attr.ib()
    eor: AbsorptionProfile = attr.ib()

    @eor.default
    def _eorcmp(self):
        return (AbsorptionProfile(freqs=self.freq, params=("A", "w", "tau", "nu0")),)

    @cached_property
    def partial_linear_model(self):
        return PartialLinearModel(
            linear_model=self.fg.at(x=self.freq),
            data={"t_sky": self.t_sky, "data_variance": self.var},
            components=(self.eor,),
            data_func=self.transform_data,
            variance_func=None,
        )

    def transform_data(self, ctx: dict, data: dict):
        return data["t_sky"] - ctx["eor_spectrum"]

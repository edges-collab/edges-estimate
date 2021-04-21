"""
Components for performing calibration on raw data.
"""
import attr
from attr import validators as vld
import numpy as np
from cached_property import cached_property
from edges_cal import receiver_calibration_func as rcf
from edges_cal.cal_coefficients import SwitchCorrection, LNA, CalibrationObservation
from edges_cal.receiver_calibration_func import power_ratio
from yabf import Component, Parameter
import logging
from edges_io.logging import logger

def _log_level_converter(val):
    if isinstance(val, int):
        return val
    elif isinstance(val, str):
        try:
            return getattr(logging, val.upper())
        except AttributeError:
            raise ValueError(f"{val} is not an available logging level")
    else:
        raise TypeError("log_level must be int or str")

@attr.s(frozen=True, cache_hash=True)
class _CalibrationQ(Component):
    """Base Component providing calibration Q_P.
    """
    path = attr.ib(kw_only=True, default='', validator=vld.instance_of(str))
    calobs_args = attr.ib(kw_only=True, default={}, converter=dict, validator=vld.instance_of(dict))
    _log_level = attr.ib(kw_only=True, default=logging.WARNING, converter=_log_level_converter)
    _calobs = attr.ib(kw_only=True, default=None,
                      validator=vld.optional(vld.instance_of(CalibrationObservation)))

    @cached_property
    def calobs(self):
        if self._calobs is not None:
            return self._calobs
        else:
            if not self.path:
                raise ValueError("if calobs is not given, path must be")

            logger.setLevel(self._log_level)
            return CalibrationObservation(path=self.path, **self.calobs_args)

    @cached_property
    def base_parameters(self):
        c1_terms = [Parameter(f"C1_{i}", 1 if not i else 0, latex=rf"C^1_{i}") for i
                    in range(self.calobs.cterms)]
        c2_terms = [Parameter(f"C2_{i}", 0, latex=rf"C^2_{i}") for i in range(self.calobs.cterms)]
        tunc_terms = [Parameter(f"Tunc_{i}", 0, latex=r"T^{\rm unc}_{%s}" % i) for i in
                      range(self.calobs.wterms)]
        tcos_terms = [Parameter(f"Tcos_{i}", 0, latex=r"T^{\rm cos}_{%s}" % i) for i in
                      range(self.calobs.wterms)]
        tsin_terms = [Parameter(f"Tsin_{i}", 0, latex=r"T^{\rm sin}_{%s}" % i) for i in
                      range(self.calobs.wterms)]

        return tuple(c1_terms + c2_terms + tunc_terms + tcos_terms + tsin_terms)

    @cached_property
    def freq(self):
        return self.calobs.freq.freq

    @cached_property
    def data_mask(self):
        """The data itself is averaged_Q from the LoadSpectrum, which may involve different
        frequencies than the calibration itself. Here we get which elements to actually use."""
        mask = []
        for i, flag in enumerate(self.calobs.open.spectrum.freq.mask):
            if not flag:
                continue
            else:
                mask.append(self.calobs.freq.mask[i])
        return np.array(mask, dtype=bool)

    @cached_property
    def freq_recentred(self):
        return np.linspace(-1, 1, len(self.freq))

    @cached_property
    def provides(self):
        return [f"{self.name}_calibration_q", 'data_mask', 'cal_curves']

    def get_calibration_curves(self, params):
        # Put coefficients in backwards, because that's how the polynomial works.
        c1, c2, tu, tc, ts = self.get_cal_funcs(params)

        return (
            c1(self.freq_recentred),
            c2(self.freq_recentred),
            tu(self.freq_recentred),
            tc(self.freq_recentred),
            ts(self.freq_recentred)
        )

    def get_cal_funcs(self, params=None):
        params = self._fill_params(params)

        # Put coefficients in backwards, because that's how the polynomial works.
        c1_poly = np.poly1d([params[f'C1_{i}'] for i in range(self.calobs.cterms)[::-1]])
        c2_poly = np.poly1d([params[f'C2_{i}'] for i in range(self.calobs.cterms)[::-1]])
        tunc_poly = np.poly1d([params[f'Tunc_{i}'] for i in range(self.calobs.wterms)[::-1]])
        tcos_poly = np.poly1d([params[f'Tcos_{i}'] for i in range(self.calobs.wterms)[::-1]])
        tsin_poly = np.poly1d([params[f'Tsin_{i}'] for i in range(self.calobs.wterms)[::-1]])

        return c1_poly, c2_poly, tunc_poly, tcos_poly, tsin_poly


@attr.s(frozen=True)
class CalibratorQ(_CalibrationQ):
    """Component providing calibration Q_P for calibrator sources ambient, hot_load,
    open, short.

    Parameters
    ----------
    """
    @cached_property
    def s11_models(self):
        return {
            "open": self.calobs.open.s11_model(self.freq),
            "short": self.calobs.short.s11_model(self.freq),
            "hot_load": self.calobs.hot_load.s11_model(self.freq),
            "ambient": self.calobs.ambient.s11_model(self.freq),
            "lna": self.calobs.lna.s11_model(self.freq)
        }

    @cached_property
    def Ks(self):
        return {
            name: rcf.get_K(self.s11_models['lna'], self.s11_models[name]) for name in self.s11_models if name != 'lna'
        }

    def calculate(self, ctx=None, **params):
        scale, offset, tu, tc, ts = self.get_cal_funcs(params)

        Qp = {}
        for name, source in self.calobs._loads.items():
            temp_ant = source.spectrum.temp_ave

            a, b = rcf.get_linear_coefficients_from_K(
                self.Ks[name],
                scale(self.freq_recentred), offset(self.freq_recentred), tu(self.freq_recentred), tc(self.freq_recentred), ts(self.freq_recentred),
                t_load=300,
            )

            Qp[name] = ((temp_ant - b) / a - 300) / 400

        return Qp, self.data_mask, {'c1': scale, 'c2': offset, 'tu': tu, 'tc': tc, 'ts': ts}


@attr.s(frozen=True)
class AntennaQ(_CalibrationQ):
    """Component providing calibration Q_P for calibrator sources ambient, hot_load,
    open, short.

    Parameters
    ----------
    antenna : :class:`~edges_cal.cal_coefficients.SwitchCorrection` or
        :class:`~edges_cal.cal_coefficients.LoadSpectrum`
        The properties of the antenna. If a `LoadSpectrum`, assumes that the true temperature
        is known. If a `SwitchCorrection`, assumes that the true temperature is forward-modelled
        by subcomponents.
    receiver : :class:`~edges_cal.cal_coefficients.LNA`
        The S11 of the reciever/LNA.
    """
    antenna = attr.ib(kw_only=True, validator=attr.validators.instance_of(SwitchCorrection))
    receiver = attr.ib(kw_only=True, validator=attr.validators.instance_of(LNA))

    @cached_property
    def freq(self):
        return self.antenna.freq.freq

    def calculate(self, ctx=None, **params):
        scale, offset, tu, tc, ts = self.get_calibration_curves(params)

        temp_ant = sum([v for k, v in ctx.items() if k.endswith('spectrum')])
        gamma_ant = self.antenna.get_s11_correction_model()(self.freq)

        return power_ratio(
            scale=scale,
            offset=offset,
            temp_cos=tc,
            temp_sin=ts,
            temp_unc=tu,
            temp_ant=temp_ant,
            gamma_ant=gamma_ant,
            gamma_rec=self.receiver.get_s11_correction_model()(self.freq),
            temp_noise_source=400,
            temp_load=300
        )

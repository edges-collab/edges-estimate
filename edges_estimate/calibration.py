"""
Components for performing calibration on raw data.
"""
import attr
import numpy as np
from cached_property import cached_property
from edges_cal.receiver_calibration_func import power_ratio
from edges_cal.cal_coefficients import SwitchCorrection, LNA, CalibrationObservation
from edges_cal import receiver_calibration_func as rcf
from yabf import Component, Parameter


@attr.s(frozen=True)
class _CalibrationQ(Component):
    """Base Component providing calibration Q_P.
    """

    @cached_property
    def base_parameters(self):
        c1_terms = [Parameter(f"C1_{i}", 1 if not i else 0, latex=rf"C^1_{i}") for i
                    in range(self.nterms_c1)]
        c2_terms = [Parameter(f"C2_{i}", 0, latex=rf"C^2_{i}") for i in range(self.nterms_c2)]
        tunc_terms = [Parameter(f"Tunc_{i}", 0, latex=r"T^{\rm unc}_{%s}" % i) for i in
                      range(self.nterms_tunc)]
        tcos_terms = [Parameter(f"Tcos_{i}", 0, latex=r"T^{\rm cos}_{%s}" % i) for i in
                      range(self.nterms_tcos)]
        tsin_terms = [Parameter(f"Tsin_{i}", 0, latex=r"T^{\rm sin}_{%s}" % i) for i in
                      range(self.nterms_tsin)]

        return tuple(c1_terms + c2_terms + tunc_terms + tcos_terms + tsin_terms)

    @cached_property
    def freq(self):
        pass

    @cached_property
    def freq_recentred(self):
        return np.linspace(-1, 1, len(self.freq))

    @cached_property
    def provides(self):
        return [f"{self.name}_calibration_q", f"{self.name}_calibration_qsigma"]

    def get_calibration_curves(self, params):
        # Put coefficients in backwards, because that's how the polynomial works.
        c1_poly = np.poly1d([params[f'C1_{i}'] for i in range(self.calobs.cterms)[::-1]])
        c2_poly = np.poly1d([params[f'C2_{i}'] for i in range(self.calobs.cterms)[::-1]])
        tunc_poly = np.poly1d([params[f'Tunc_{i}'] for i in range(self.calobs.wterms)[::-1]])
        tcos_poly = np.poly1d([params[f'Tcos_{i}'] for i in range(self.calobs.wterms)[::-1]])
        tsin_poly = np.poly1d([params[f'Tsin_{i}'] for i in range(self.calobs.wterms)[::-1]])

        return (
            c1_poly(self.freq_recentred),
            c2_poly(self.freq_recentred),
            tunc_poly(self.freq_recentred),
            tcos_poly(self.freq_recentred),
            tsin_poly(self.freq_recentred)
        )


@attr.s(frozen=True)
class CalibratorQ(_CalibrationQ):
    """Component providing calibration Q_P for calibrator sources ambient, hot_load,
    open, short.

    Parameters
    ----------
    antenna : :class:`~edges_cal.cal_coefficients.SwitchCorrection` or :class:`~edges_cal.cal_coefficients.LoadSpectrum`
        The properties of the antenna. If a `LoadSpectrum`, assumes that the true temperature
        is known. If a `SwitchCorrection`, assumes that the true temperature is forward-modelled
        by subcomponents.
    receiver : :class:`~edges_cal.cal_coefficients.LNA`
        The S11 of the reciever/LNA.
    """
    calobs = attr.ib(kw_only=True, validator=attr.validators.instance_of(CalibrationObservation))

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

    def calculate(self, ctx=None, **params):
        scale, offset, tu, tc, ts = self.get_calibration_curves(params)

        Qp = {}
        curlyQ = {}
        for source in self.calobs._sources:
            if source == 'hot_load':
                temp_ant = self.calobs.hot_load_corrected_ave_temp
            else:
                temp_ant = getattr(self.calobs, source).temp_ave

            terms = power_ratio(
                scale=scale,
                offset=offset,
                temp_cos=tc,
                temp_sin=ts,
                temp_unc=tu,
                temp_ant=temp_ant,
                gamma_ant=self.calobs.s11_correction_models[source],
                gamma_rec=self.calobs.lna_s11.get_s11_correction_model()(self.freq),
                temp_noise_source=400,
                temp_load=300,
                return_terms=True
            )

            Qp[source] = sum(terms[:5])/terms[5]
            curlyQ[source] = sum([t**2 for t in terms[:5]]) / sum(terms[:5])**2

        return Qp, curlyQ


@attr.s(frozen=True)
class AntennaQ(_CalibrationQ):
    """Component providing calibration Q_P for calibrator sources ambient, hot_load,
    open, short.

    Parameters
    ----------
    antenna : :class:`~edges_cal.cal_coefficients.SwitchCorrection` or :class:`~edges_cal.cal_coefficients.LoadSpectrum`
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

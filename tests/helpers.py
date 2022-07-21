import numpy as np
from astropy import units as u
from edges_cal.modelling import Polynomial, UnitTransform
from edges_cal.simulate import simulate_qant_from_calobs
from scipy import stats
from yabf import ParamVec


def get_tns_model(calobs, ideal=False):
    if ideal:
        p = np.array([1575, -175, 70.0, -17.5, 7.0, -3.5])
    else:
        p = calobs.C1_poly.coeffs[::-1] * calobs.t_load_ns

    t_ns_model = Polynomial(
        parameters=p,
        transform=UnitTransform(range=(calobs.freq.min.value, calobs.freq.max.value)),
    )

    t_ns_params = ParamVec(
        "t_lns",
        length=len(p),
        min=p - 100,
        max=p + 100,
        ref=[stats.norm(v, scale=1.0) for v in p],
        fiducial=p,
    )
    return t_ns_model, t_ns_params


def sim_antenna_q(labcal, calobs, fg, eor, ideal_tns=True, loss=1, bm_corr=1):
    spec = fg(x=eor.freqs) + eor()["eor_spectrum"]

    tns_model, _ = get_tns_model(calobs, ideal=ideal_tns)
    _scale_model = tns_model.with_params(
        np.array(tns_model.parameters) / calobs.t_load_ns
    )

    return simulate_qant_from_calobs(
        calobs,
        ant_s11=labcal.antenna_s11_model(eor.freqs),
        ant_temp=spec,
        scale_model=lambda f: _scale_model(f.to_value("MHz")),
        loss=loss,
        freq=eor.freqs * u.MHz,
        bm_corr=bm_corr,
    )

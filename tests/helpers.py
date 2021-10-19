import numpy as np
from edges_cal.modelling import Polynomial, UnitTransform
from edges_cal.simulate import simulate_qant_from_calobs
from scipy import stats
from yabf import ParamVec


def get_tns_model(calobs, ideal=True):
    if ideal:
        p = np.array([1575, -175, 70.0, -17.5, 7.0, -3.5])
    else:
        p = calobs.C1_poly.coeffs[::-1] * calobs.t_load_ns

    t_ns_model = Polynomial(
        parameters=p, transform=UnitTransform(range=(calobs.freq.min, calobs.freq.max))
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


def sim_antenna_q(labcal, fg, eor, ideal_tns=True, loss=1, bm_corr=1):
    calobs = labcal.calobs

    spec = fg(x=eor.freqs) + eor()["eor_spectrum"]

    tns_model, _ = get_tns_model(calobs, ideal=ideal_tns)
    scale_model = tns_model.with_params(tns_model.parameters / calobs.t_load_ns)

    return simulate_qant_from_calobs(
        calobs,
        ant_s11=labcal.antenna_s11_model(eor.freqs),
        ant_temp=spec,
        scale_model=scale_model,
        loss=loss,
        freq=eor.freqs,
        bm_corr=bm_corr,
    )

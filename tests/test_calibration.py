import pytest

import numpy as np
from edges_cal.modelling import Polynomial, UnitTransform
from edges_cal.simulate import simulate_qant_from_calobs
from scipy import stats
from yabf import ParamVec, run_map

from edges_estimate.eor_models import AbsorptionProfile
from edges_estimate.likelihoods import DataCalibrationLikelihood


def get_tns_model(calobs, ideal=True):
    if ideal:
        p = np.array([1575, -175, 70.0, -17.5, 7.0, -3.5])
    else:
        p = calobs.C1_poly.coeffs[::-1] * calobs.t_load_ns

    t_ns_model = Polynomial(parameters=p, transform=UnitTransform())

    t_ns_params = ParamVec(
        "t_lns",
        length=len(p),
        min=p - 100,
        max=p + 100,
        ref=[stats.norm(v, scale=1.0) for v in p],
        fiducial=p,
    )
    return t_ns_model, t_ns_params


def sim_antenna_q(labcal, fg, eor, ideal_tns=True):
    calobs = labcal.calobs

    spec = fg(x=eor.freqs) + eor()["eor_spectrum"]

    tns_model, _ = get_tns_model(calobs, ideal=ideal_tns)
    scale_model = tns_model.with_params(tns_model.parameters / calobs.t_load_ns)

    return simulate_qant_from_calobs(
        calobs, ant_s11=labcal.antenna_s11, ant_temp=spec, scale_model=scale_model
    )


def get_likelihood(
    labcal, qvar_ant, fg, eor, cal_noise, simulate=True, ideal_tns=True, smooth=1
):
    calobs = labcal.calobs
    fid_eor = get_eor(calobs)
    q = sim_antenna_q(labcal, fg, fid_eor, ideal_tns=ideal_tns)

    if isinstance(qvar_ant, (int, float)):
        qvar_ant = qvar_ant * np.ones_like(labcal.calobs.freq.freq)

    q = q + np.random.normal(scale=qvar_ant)

    tns_model, tns_params = get_tns_model(calobs, ideal=ideal_tns)

    if ideal_tns:
        scale_model = Polynomial(
            parameters=np.array(tns_params.fiducial) / labcal.calobs.t_load_ns,
            transform=UnitTransform(),
        )
    else:
        scale_model = None

    return DataCalibrationLikelihood.from_labcal(
        labcal,
        q_ant=q[::smooth],
        qvar_ant=qvar_ant[::smooth] / smooth,
        fg_model=fg,
        eor_components=(eor,),
        sim=simulate,
        scale_model=scale_model,
        t_ns_params=tns_params,
        cal_noise=cal_noise,
        field_freq=calobs.freq.freq[::smooth],
    )


def get_eor(calobs, smooth=1):
    return AbsorptionProfile(
        freqs=calobs.freq.freq[::smooth],
        params={
            "A": {
                "fiducial": 0.5,
                "min": 0,
                "max": 1.5,
                "ref": stats.norm(0.5, scale=0.01),
            },
            "w": {
                "fiducial": 15,
                "min": 5,
                "max": 25,
                "ref": stats.norm(15, scale=0.1),
            },
            "tau": {
                "fiducial": 5,
                "min": 0,
                "max": 20,
                "ref": stats.norm(5, scale=0.1),
            },
            "nu0": {
                "fiducial": 78,
                "min": 60,
                "max": 90,
                "ref": stats.norm(78, scale=0.1),
            },
        },
    )


@pytest.mark.parametrize(
    "lc,qvar_ant,cal_noise,simulate,ideal_tns,atol,smooth",
    [
        ("labcal", 0.0, 0.0, True, True, 0.01, 1),  # No noise
        ("labcal", 1e-10, 1e-10, True, True, 0.01, 1),  # Small constant noise
        (
            "labcal",
            1e-10,
            "data",
            True,
            True,
            0.01,
            1,
        ),  # Realistic non-constant noise on smooth cal solutions
        ("labcal12", 1e-10, "data", False, False, 0.05, 1),  # Actual cal data
        (
            "labcal12",
            1e-10,
            "data",
            False,
            False,
            0.05,
            10,
        ),  # Actual cal data, with fewer data freqs
    ],
)
def test_cal_data_likelihood(
    lc, fiducial_fg, qvar_ant, cal_noise, simulate, ideal_tns, atol, smooth, request
):
    labcal = request.getfixturevalue(lc)
    eor = get_eor(labcal.calobs, smooth=smooth)
    lk = get_likelihood(
        labcal,
        qvar_ant=qvar_ant,
        fg=fiducial_fg,
        eor=eor,
        cal_noise=cal_noise,
        simulate=simulate,
        ideal_tns=ideal_tns,
        smooth=smooth,
    )

    res = run_map(lk.partial_linear_model)
    eorspec = lk.partial_linear_model.get_ctx(params=res.x)

    tns_model, _ = get_tns_model(labcal.calobs, ideal=ideal_tns)
    tns_model = tns_model(labcal.calobs.freq.freq)
    np.testing.assert_allclose(tns_model, eorspec["tns"], atol=0, rtol=1e-2)
    np.testing.assert_allclose(
        eor()["eor_spectrum"], eorspec["eor_spectrum"], atol=atol, rtol=0
    )

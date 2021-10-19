import pytest

import matplotlib as mpl
import numpy as np
from edges_cal.modelling import Polynomial, UnitTransform
from edges_cal.simulate import simulate_qant_from_calobs
from helpers import get_tns_model, sim_antenna_q
from scipy import stats
from yabf import ParamVec, run_map

from edges_estimate.eor_models import AbsorptionProfile
from edges_estimate.likelihoods import DataCalibrationLikelihood


@pytest.fixture(scope="function")
def calobs_freq(calobs):
    return calobs.freq.freq


@pytest.fixture(scope="function")
def calobs_freq_smoothed8(calobs):
    return calobs.freq.freq[::8]


@pytest.fixture(scope="function")
def sky_freq():
    return np.linspace(58, 93, 120)


def get_likelihood(
    labcal,
    qvar_ant,
    fg,
    eor,
    cal_noise,
    simulate=True,
    ideal_tns=True,
    loss=1,
    bm_corr=1,
    seed=1234,
):
    calobs = labcal.calobs
    q = sim_antenna_q(labcal, fg, eor, ideal_tns=ideal_tns, loss=loss, bm_corr=bm_corr)

    if isinstance(qvar_ant, (int, float)):
        qvar_ant = qvar_ant * np.ones_like(eor.freqs)

    if seed:
        np.random.seed(seed)

    q = q + np.random.normal(scale=qvar_ant)

    tns_model, tns_params = get_tns_model(calobs, ideal=ideal_tns)

    if ideal_tns:
        scale_model = Polynomial(
            parameters=np.array(tns_params.fiducial) / labcal.calobs.t_load_ns,
            transform=UnitTransform(range=(calobs.freq.min, calobs.freq.max)),
        )
    else:
        scale_model = None

    return DataCalibrationLikelihood.from_labcal(
        labcal,
        q_ant=q,
        qvar_ant=qvar_ant,
        fg_model=fg,
        eor_components=(eor,),
        sim=simulate,
        scale_model=scale_model,
        t_ns_params=tns_params,
        cal_noise=cal_noise,
        field_freq=eor.freqs,
        loss=loss,
        bm_corr=bm_corr,
    )


def get_eor(freqs):
    return AbsorptionProfile(
        freqs=freqs,
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


def view_results(
    lk, res_data, calobs, eor, plt, sim_tns=True, label=None, fig=None, ax=None, c=0
):
    """Simple function to create a plot of input vs expected TNS and T21."""
    eorspec = lk.partial_linear_model.get_ctx(params=res_data.x)

    if fig is None:
        plot_input = True
        fig, ax = plt.subplots(2, 2, figsize=(15, 7), sharex=True)
    else:
        plot_input = False

    color = f"C{c}"
    nu = calobs.freq.freq

    tns_model, _ = get_tns_model(calobs, ideal=sim_tns)
    tns_model = tns_model(nu)

    if plot_input:
        ax[0, 0].plot(nu, tns_model, label="Input", color="k")

    ax[0, 0].plot(
        nu,
        eorspec["tns"],
        label="Estimated" + (" " + label if label else ""),
        color=color,
    )

    ax[1, 0].plot(
        nu,
        eorspec["tns"] - tns_model,
        label=r"$\Delta T_{\rm NS}$" if plot_input else None,
        color=color,
    )

    ax[0, 0].set_title(r"$T_{\rm NS}$")
    ax[0, 0].set_ylabel("Temperature [K]")

    nu = eor.freqs

    if plot_input:
        ax[0, 1].plot(nu, eor()["eor_spectrum"], color="k")

    ax[0, 1].plot(nu, eorspec["eor_spectrum"])
    ax[0, 1].set_title(r"$T_{21}$")
    delta = eorspec["eor_spectrum"] - eor()["eor_spectrum"]
    ax[1, 1].plot(
        nu,
        delta,
        color=color,
        label=fr"Max $\Delta = {np.max(np.abs(delta))*1000:1.2e}$mK",
    )
    ax[1, 0].set_ylabel("Difference [K]")

    ax[1, 0].set_xlabel("Frequency")
    ax[1, 1].set_xlabel("Frequency")

    ax[0, 0].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()

    return fig, ax


# Define a loss function gotten from fitting to Alan's loss model (not necessary that
# its accurate, just realistic to make the test useful)
data_like_loss = Polynomial(
    parameters=[
        9.94009502e-01,
        7.17401803e-04,
        4.50954613e-03,
        -5.83728183e-03,
        -2.10009238e-03,
        3.81924327e-03,
        -1.58099823e-03,
        4.37297391e-04,
    ],
    n_terms=8,
    transform=UnitTransform(range=[50, 100]),
)

data_like_bmcorr = Polynomial(
    parameters=[
        1.00000433,
        -0.00790527,
        -0.00566472,
        -0.00278198,
        0.0016465,
        -0.00294487,
        -0.00508308,
        0.00379791,
        0.00293712,
        -0.00173159,
    ],
    n_terms=10,
    transform=UnitTransform(range=[50, 100]),
)


def unity_loss(x):
    return 1


@pytest.mark.parametrize(
    "lc,qvar_ant,cal_noise,simulate,ideal_tns,atol,fsky,loss,bm_corr",
    [
        (
            "labcal",
            0.0,
            0.0,
            True,
            True,
            0.01,
            "calobs_freq",
            unity_loss,
            unity_loss,
        ),  # No noise
        (
            "labcal",
            1e-10,
            1e-10,
            True,
            True,
            0.01,
            "calobs_freq",
            unity_loss,
            unity_loss,
        ),  # Small constant noise
        (
            "labcal",
            1e-10,
            "data",
            True,
            True,
            0.01,
            "calobs_freq",
            unity_loss,
            unity_loss,
        ),  # Realistic non-constant noise on smooth cal solutions
        (
            "labcal12",
            1e-10,
            "data",
            False,
            False,
            0.05,
            "calobs_freq",
            unity_loss,
            unity_loss,
        ),  # Actual cal data
        (
            "labcal",
            1e-10,
            "data",
            True,
            True,
            0.01,
            "calobs_freq_smoothed8",
            unity_loss,
            unity_loss,
        ),  # Realistic non-constant noise on smooth cal solutions with fewer sky freqs
        (
            "labcal",
            1e-12,
            "data",
            True,
            True,
            0.01,
            "sky_freq",
            unity_loss,
            unity_loss,
        ),  # Realistic non-constant noise on smooth cal solutions with different freq range
        (
            "labcal",
            1e-10,
            "data",
            True,
            True,
            0.01,
            "calobs_freq",
            data_like_loss,
            unity_loss,
        ),  # Realistic non-constant noise on smooth cal solutions WITH LOSS
        (
            "labcal",
            1e-10,
            "data",
            True,
            True,
            0.01,
            "calobs_freq",
            data_like_loss,
            data_like_bmcorr,
        ),  # Realistic non-constant noise on smooth cal solutions WITH LOSS AND BMCORR
    ],
)
def test_cal_data_likelihood(
    lc,
    fiducial_fg,
    qvar_ant,
    cal_noise,
    simulate,
    ideal_tns,
    atol,
    fsky,
    loss,
    bm_corr,
    request,
    plt,
):
    fsky = request.getfixturevalue(fsky)
    labcal = request.getfixturevalue(lc)
    eor = get_eor(fsky)

    lk = get_likelihood(
        labcal,
        qvar_ant=qvar_ant,
        fg=fiducial_fg,
        eor=eor,
        cal_noise=cal_noise,
        simulate=simulate,
        ideal_tns=ideal_tns,
        loss=loss(fsky),
        bm_corr=bm_corr(fsky),
    )

    res = run_map(lk.partial_linear_model)
    eorspec = lk.partial_linear_model.get_ctx(params=res.x)

    tns_model, _ = get_tns_model(labcal.calobs, ideal=ideal_tns)
    tns_model = tns_model(labcal.calobs.freq.freq)

    if plt == mpl.pyplot:
        view_results(lk, res, labcal.calobs, eor, plt, sim_tns=simulate)

    np.testing.assert_allclose(tns_model, eorspec["tns"], atol=0, rtol=1e-2)
    np.testing.assert_allclose(
        eor()["eor_spectrum"], eorspec["eor_spectrum"], atol=atol, rtol=0
    )

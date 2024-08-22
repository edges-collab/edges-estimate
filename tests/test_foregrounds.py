import numpy as np
from yabf import chi2, run_map


def create_mock_data(fiducial_fg_logpoly):
    spec = fiducial_fg_logpoly()
    assert len(spec["LogPoly_spectrum"]) == 100
    return spec["LogPoly_spectrum"]


def test_retrieve_params(fiducial_fg_logpoly):
    spec = create_mock_data(fiducial_fg_logpoly)
    lk = chi2.MultiComponentChi2(
        kind="spectrum", components=[fiducial_fg_logpoly], data=spec
    )
    a = run_map(lk)
    assert a.success
    assert np.allclose(a.x, [2, -2.5, 50])
    assert len(a.x) == 3


def test_damped_oscillations(fiducial_dampedoscillations):
    spec = fiducial_dampedoscillations()
    print(spec)
    assert np.allclose(spec["DampedOscillations_spectrum"], 0)
    assert len(spec["DampedOscillations_spectrum"]) == 100

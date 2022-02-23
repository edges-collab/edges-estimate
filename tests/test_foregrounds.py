import pytest
import numpy as np
from yabf import chi2, run_map

#@pytest.fixture(scope="session")
def create_mock_data(fiducial_fg_logpoly):
    spec = fiducial_fg_logpoly()
    assert len(spec['LogPoly_spectrum'])==100
    return spec['LogPoly_spectrum']


def test_retrieve_params(fiducial_fg_logpoly):
    spec = create_mock_data(fiducial_fg_logpoly)
    lk = chi2.MultiComponentChi2(kind='spectrum',components=[fiducial_fg_logpoly],data=spec)
    a = run_map(lk)
    print(a)
    assert a.success
    assert len(a.x) == 3

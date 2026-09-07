"""
This test module seeks to systematically test the NoiseWaveLikelihood likelihood.

It does this by first ensuring it gets reasonable results when the data is purely
simulated. It then goes on to test that if the input model is larger than the true
model, the extra parameters are estimated at zero. Then it goes on to compare the BIC
of each of these models, ensuring that the smaller model has a better BIC.
"""

import numpy as np
import pytest
from numdifftools import Hessian
from yabf import run_map

from edges_estimate.likelihoods import NoiseWaveLikelihood


@pytest.fixture(scope="module")
def fid_lk(calobs):
    return NoiseWaveLikelihood.from_calobs(calobs, seed=1234, as_sim="all")


@pytest.fixture(scope="module")
def larger_lk(calobs):
    return NoiseWaveLikelihood.from_calobs(calobs, as_sim="all", cterms=7, seed=1234)


@pytest.fixture(scope="module")
def true_tns(calobs):
    return calobs.C1_poly.coeffs[::-1] * calobs.t_load_ns


@pytest.fixture(scope="module")
def larger_tns(true_tns):
    return np.concatenate((true_tns, [0]))


@pytest.fixture(scope="module")
def fid_res(fid_lk):
    return run_map(fid_lk.partial_linear_model)


@pytest.fixture(scope="module")
def larger_res(larger_lk):
    return run_map(larger_lk.partial_linear_model)


def get_std(lk, res):
    h = Hessian(lambda x: lk.partial_linear_model(params=x)[0])
    return np.sqrt(np.diag(np.linalg.inv(-h(res.x))))


def test_pure_sim(fid_lk, fid_res, true_tns):
    assert fid_res.success
    std = get_std(fid_lk, fid_res)
    assert np.allclose(fid_res.x, true_tns, rtol=0, atol=3 * std)


def test_pure_sim_larger_model(larger_lk, larger_res, larger_tns):
    assert larger_res.success
    std = get_std(larger_lk, larger_res)
    assert np.allclose(larger_res.x, larger_tns, rtol=0, atol=3 * std)


def test_pure_sim_bic(fid_lk, fid_res, larger_lk, larger_res):
    fit, var, data = fid_lk.partial_linear_model.reduce_model(params=fid_res.x)
    nparams = fid_lk.partial_linear_model.linear_model.n_terms + len(
        fid_lk.partial_linear_model.child_active_params
    )
    fid_bic = nparams * len(data) - 2 * fid_lk.partial_linear_model(params=fid_res.x)[0]

    fit, var, data = larger_lk.partial_linear_model.reduce_model(params=larger_res.x)
    nparams = larger_lk.partial_linear_model.linear_model.n_terms + len(
        larger_lk.partial_linear_model.child_active_params
    )
    larger_bic = nparams * len(data) - 2 * larger_lk.partial_linear_model(params=larger_res.x)[0]

    assert fid_bic < larger_bic

import numpy as np
from yabf import run_map

from edges_estimate.likelihoods import NoiseWaveLikelihood


def test_pure_sim(calobs):
    lk = NoiseWaveLikelihood.from_sim_calobs(calobs, variance="data")

    out = run_map(lk.partial_linear_model)
    assert out.success
    np.testing.assert_allclose(out.x, calobs.C1_poly.coeffs[::-1] * calobs.t_load_ns)

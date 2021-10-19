
from edges_estimate import plots as p
from getdist import loadMCSamples
import pytest

@pytest.fixture(scope="function")
def test_mcsamples(mcsamples):
    samples= loadMCSamples('sample')
    return samples

def test_model_from_mc_samples(samples):
    models= p.get_models_from_mcsamples(samples, lk_names = ["lowband"])
    assert len(models) ==1
    return models

def test_def_make_residual_plot_shaded(models):
    p.make_residual_plot_shaded()
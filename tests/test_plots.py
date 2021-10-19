import pytest

<<<<<<< HEAD
import pytest

import numpy as np

from edges_estimate import plots as p


@pytest.fixture(scope="session")
def models():
    models = {key: {'model': np.random.normal(size=(20, 50))} for key in ['one', 'two']}
    return models

@pytest.fixture(scope="session")
def freqs():
    return  {key:  np.random.normal(size=50) for key in ['one', 'two']}

@pytest.fixture(scope="session")
def temp():
    return  {key: np.random.normal(size=50) for key in ['one', 'two']}

def test_def_make_residual_plot_shaded(models,freqs,temp):
    p.make_residual_plot_shaded(models,freqs,temp)

=======
from getdist import loadMCSamples

from edges_estimate import plots as p


@pytest.fixture(scope="function")
def test_mcsamples(mcsamples):
    samples = loadMCSamples("sample")
    return samples


def test_model_from_mc_samples(samples):
    models = p.get_models_from_mcsamples(samples, lk_names=["lowband"])
    assert len(models) == 1
    return models


def test_def_make_residual_plot_shaded(models):
    p.make_residual_plot_shaded()
>>>>>>> 4eaede6b496a24ccdb722f52fa9931e3477acd55

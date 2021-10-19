
from edges_estimate import plots as p
import numpy as np
import pytest

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


"""Test loss functions in dmg/models/criterion/."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

from dmg.models.criterion.base import BaseCriterion
from tests import get_available_classes

# Path to loss functions
PATH = Path(__file__).parent.parent / 'src' / 'dmg' / 'models' / 'criterion'
PKG_PATH = 'dmg.models.criterion'


@pytest.fixture(params=get_available_classes(PATH, PKG_PATH, BaseCriterion))
def loss_class(request):
    """Fixture to provide each loss function class dynamically."""
    return request.param


@pytest.fixture
def config():
    """Fixture for default configuration."""
    return {
        'eps': 0.1,
        'beta': 4e2,
        'device': 'cpu',
        'other_param': 42,
    }


@pytest.fixture
def prediction_data():
    """Fixture for prediction tensor data."""
    return torch.tensor(
        [[[2.0], [7.0], [1.0], [4.0]], [[9.0], [7.0], [2.0], [6.0]]],
    )


@pytest.fixture
def target_data():
    """Fixture for target tensor data."""
    return torch.tensor(
        [[[3.0], [1.0], [4.0], [np.nan]], [[1.0], [5.0], [9.0], [2.0]]],
    )

@pytest.fixture
def sample_id_data():
    """Fixture for sample ID tensor data."""
    return np.array([0, 1, 2, 3])


def test_init(loss_class, config, target_data):
    """Test initialization of loss function classes."""
    try:
        loss_fn = loss_class(config=config, y_obs=target_data)
        assert hasattr(loss_fn, 'name')
        assert hasattr(loss_fn, 'config')
        assert hasattr(loss_fn, 'device')
        assert loss_fn.device == 'cpu'
    except (AssertionError, RuntimeError, TypeError) as e:
        pytest.fail(f"Initialization failed for {loss_class.__name__}: {e}")


def test_forward(
    loss_class,
    config,
    prediction_data,
    target_data,
    sample_id_data,
):
    """Test forward method with valid input."""
    try:
        loss_fn = loss_class(config=config, y_obs=target_data)

        # Forward pass
        loss = loss_fn(prediction_data, target_data, sample_ids=sample_id_data)

        # Check if loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar tensor
        assert not torch.isnan(loss)  # Loss should not be NaN
    except (AssertionError, RuntimeError, TypeError) as e:
        pytest.fail(f"Forward pass failed for {loss_class.__name__}: {e}")

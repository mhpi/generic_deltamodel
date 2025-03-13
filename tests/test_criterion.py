"""Test loss functions in dMG/models/criterion/."""
import importlib
import pkgutil
from pathlib import Path

import numpy as np
import pytest
import torch
from dMG.models.criterion.base import BaseCriterion

# Path to loss functions
LOSS_DIR = Path(__file__).parent.parent / "src" / "dMG" / "models" / "criterion"


def get_loss_classes():
    """Dynamically import all loss function classes from the specified directory."""
    loss_classes = []
    for _, module_name, _ in pkgutil.iter_modules([str(LOSS_DIR)]):
        module = importlib.import_module(f"dMG.models.criterion.{module_name}")
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            # Check if the attribute is a subclass of torch.nn.Module and not abstract
            if (
                isinstance(attr, type)
                and attr is not BaseCriterion
                and issubclass(attr, torch.nn.Module)
                and attr.__name__ != "Module"
            ):
                loss_classes.append(attr)
    # Remove BaseCriterion class

    return loss_classes


@pytest.fixture(params=get_loss_classes())
def loss_class(request):
    """Fixture to provide each loss function class dynamically."""
    return request.param


@pytest.fixture
def config():
    """Fixture for default configuration."""
    return {"eps": 0.1}


@pytest.fixture
def target_data():
    """Fixture for target tensor data."""
    return torch.tensor(
        [[[3.0], [1.0], [4.0], [np.nan]], [[1.0], [5.0], [9.0], [2.0]]]
    )


@pytest.fixture
def prediction_data():
    """Fixture for prediction tensor data."""
    return torch.tensor([1.1, 2.2, 3.3, 4.4])


def test_init(loss_class, config, target_data):
    """Test initialization of loss function classes."""
    try:
        loss_fn = loss_class(target=target_data, config=config)
        assert hasattr(loss_fn, "name")
        assert hasattr(loss_fn, "config")
        assert hasattr(loss_fn, "device")
        assert loss_fn.device == "cpu"
    except Exception as e:
        pytest.fail(f"Initialization failed for {loss_class.__name__}: {e}")


# def test_forward_valid_input(loss_class, config, target_data, prediction_data):
#     """Test forward method with valid input."""
#     try:
#         loss_fn = loss_class(target=target_data, config=config)

#         # Forward pass
#         y_pred = prediction_data.unsqueeze(1)  # Shape adjustment for compatibility
#         loss = loss_fn(y_pred)

#         # Check if loss is a scalar tensor
#         assert isinstance(loss, torch.Tensor)
#         assert loss.dim() == 0  # Scalar tensor
#         assert not torch.isnan(loss)  # Loss should not be NaN
#     except Exception as e:
#         pytest.fail(f"Forward pass failed for {loss_class.__name__}: {e}")


# def test_forward_with_nan_in_target(loss_class, config, target_data, prediction_data):
#     """Test forward method with NaN values in the target."""
#     try:
#         loss_fn = loss_class(target=target_data, config=config)

#         # Forward pass
#         y_pred = prediction_data.unsqueeze(1)  # Shape adjustment for compatibility
#         loss = loss_fn(y_pred)

#         # Ensure NaN values are ignored in computation
#         assert isinstance(loss, torch.Tensor)
#         assert loss.dim() == 0  # Scalar tensor
#         assert not torch.isnan(loss)  # Loss should not be NaN
#     except Exception as e:
#         pytest.fail(f"NaN handling failed for {loss_class.__name__}: {e}")


# def test_forward_with_all_nan_in_target(loss_class, config):
#     """Test forward method when all target values are NaN."""
#     target = torch.tensor([[float('nan')], [float('nan')], [float('nan')]])
#     try:
#         loss_fn = loss_class(target=target, config=config)

#         # Forward pass
#         y_pred = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(1)
#         with pytest.raises(ValueError, match="No valid target values available"):
#             loss_fn(y_pred)
#     except Exception as e:
#         pytest.fail(f"All-NaN test failed for {loss_class.__name__}: {e}")


# def test_forward_with_zero_std(loss_class, config):
#     """Test forward method when standard deviation of target or prediction is zero."""
#     target = torch.tensor([[1.0], [1.0], [1.0]])  # Constant target
#     try:
#         loss_fn = loss_class(target=target, config=config)

#         # Forward pass
#         y_pred = torch.tensor([1.0, 1.0, 1.0]).unsqueeze(1)
#         loss = loss_fn(y_pred)

#         # Loss should handle zero std without crashing
#         assert not torch.isnan(loss)
#     except Exception as e:
#         pytest.fail(f"Zero std test failed for {loss_class.__name__}: {e}")


# def test_forward_with_different_shapes(loss_class, config, target_data, prediction_data):
#     """Test forward method with mismatched shapes."""
#     try:
#         loss_fn = loss_class(target=target_data, config=config)

#         # Prediction tensor with incompatible shape
#         y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

#         with pytest.raises(ValueError, match="Prediction and target shapes do not match"):
#             loss_fn(y_pred)
#     except Exception as e:
#         pytest.fail(f"Shape mismatch test failed for {loss_class.__name__}: {e}")


# def test_forward_with_empty_tensors(loss_class, config):
#     """Test forward method with empty tensors."""
#     target = torch.empty((0, 1))  # Empty target tensor
#     try:
#         loss_fn = loss_class(target=target, config=config)

#         # Forward pass
#         y_pred = torch.empty((0, 1))  # Empty prediction tensor
#         with pytest.raises(ValueError, match="No valid data available for computation"):
#             loss_fn(y_pred)
#     except Exception as e:
#         pytest.fail(f"Empty tensor test failed for {loss_class.__name__}: {e}")


# def test_forward_with_invalid_config(loss_class):
#     """Test initialization with invalid configuration."""
#     target = torch.tensor([[1.0], [2.0]])
#     config = {"eps": -0.1}  # Invalid epsilon value

#     try:
#         with pytest.raises(ValueError, match="Epsilon must be non-negative"):
#             loss_class(target=target, config=config)
#     except Exception as e:
#         pytest.fail(f"Invalid config test failed for {loss_class.__name__}: {e}")


# def test_forward_with_custom_device(loss_class, config):
#     """Test forward method with custom device (e.g., GPU)."""
#     if torch.cuda.is_available():
#         device = "cuda"
#         target = torch.tensor([[1.0], [2.0]], device=device)
#         try:
#             loss_fn = loss_class(target=target, config=config, device=device)

#             # Forward pass
#             y_pred = torch.tensor([1.1, 2.2], device=device).unsqueeze(1)
#             loss = loss_fn(y_pred)

#             # Ensure loss is computed on the same device
#             assert loss.device.type == device
#         except Exception as e:
#             pytest.fail(f"Custom device test failed for {loss_class.__name__}: {e}")
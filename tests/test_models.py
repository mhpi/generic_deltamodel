"""Test models in dmg/models/."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest
import torch

from dmg.core.utils import set_randomseed
from dmg.models.delta_models.dpl_model import DplModel
from dmg.models.model_handler import ModelHandler


def test_dpl_model_initialization_with_config(config):
    """Test DplModel initialization using config."""
    model = DplModel(config=config['model'], device='cpu')

    assert model.initialized
    assert model.nn_model is not None
    assert model.phy_model is not None
    assert model.device.type == 'cpu'


def test_dpl_model_forward_pass(config, mock_dataset):
    """Test the DplModel forward pass."""
    set_randomseed(config['seed'])

    model = DplModel(config=config['model'], device='cpu')

    output = model(mock_dataset)
    assert isinstance(output, torch.Tensor)
    assert output.shape == mock_dataset['target'].shape
    assert not torch.isnan(output).any()


def test_dpl_model_with_different_nn_architectures(config, mock_dataset):
    """Test DplModel with different NN architecture types."""
    set_randomseed(config['seed'])

    # Test with LSTM (default in config)
    model_lstm = DplModel(config=config['model'], device='cpu')
    output_lstm = model_lstm(mock_dataset)
    assert output_lstm.shape == mock_dataset['target'].shape

    # Test with MLP
    config_mlp = config.copy()
    config_mlp['model']['nn']['name'] = 'MlpModel'
    model_mlp = DplModel(config=config_mlp['model'], device='cpu')
    output_mlp = model_mlp(mock_dataset)
    assert output_mlp.shape == mock_dataset['target'].shape


def test_model_handler(config, mock_dataset):
    """Test the ModelHandler."""
    set_randomseed(config['seed'])

    handler = ModelHandler(config)
    assert 'Hbv' in handler.model_dict
    assert handler.model_type == 'dm'

    # Test forward pass in train mode
    handler.train()
    output = handler(mock_dataset)
    assert 'Hbv' in output
    assert isinstance(output['Hbv'], torch.Tensor)

    # Test forward pass in eval mode
    output_eval = handler(mock_dataset, eval=True)
    assert 'Hbv' in output_eval
    assert isinstance(output_eval['Hbv'], torch.Tensor)


def test_model_handler_multiple_models(config, mock_dataset):
    """Test ModelHandler with multiple physics models."""
    # This would require modifying config to have multiple models
    # Skip if not in multimodel mode
    if config['multimodel_type'] == 'none':
        pytest.skip("Test requires multimodel configuration")


def test_model_handler_loss_calculation(config, mock_dataset):
    """Test ModelHandler loss calculation functionality."""
    from dmg.models.criterion.rmse_loss import RmseLoss

    set_randomseed(config['seed'])

    handler = ModelHandler(config)
    loss_func = RmseLoss(config)

    # Forward pass
    _ = handler(mock_dataset)

    # Calculate loss
    loss = handler.calc_loss(mock_dataset, loss_func)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert loss >= 0
    assert not torch.isnan(loss)


def test_model_handler_get_parameters(config):
    """Test that ModelHandler can retrieve all trainable parameters."""
    handler = ModelHandler(config)

    params = list(handler.get_parameters())

    assert len(params) > 0
    for param in params:
        assert isinstance(param, torch.Tensor)
        assert param.requires_grad


def test_model_train_eval_mode_switching(config, mock_dataset):
    """Test switching between train and eval modes."""
    handler = ModelHandler(config)

    # Train mode
    handler.train()
    for model in handler.model_dict.values():
        assert model.training

    # Eval mode
    handler.eval()
    for model in handler.model_dict.values():
        assert not model.training


def test_dpl_model_parameter_shapes(config, mock_dataset):
    """Test that NN outputs correct parameter shapes for physics model."""
    set_randomseed(config['seed'])

    model = DplModel(config=config['model'], device='cpu')

    with torch.no_grad():
        if type(model.nn_model).__name__ == 'LstmMlpModel':
            params = model.nn_model(
                mock_dataset['xc_nn_norm'], mock_dataset['c_nn_norm']
            )
        else:
            params = model.nn_model(mock_dataset['xc_nn_norm'])

    expected_param_count = model.phy_model.learnable_param_count
    assert params.shape[-1] == expected_param_count, (
        f"NN outputs {params.shape[-1]} params, physics model expects {expected_param_count}"
    )


def test_model_device_placement(config, mock_dataset):
    """Test model placement on specified device."""
    model = DplModel(config=config['model'], device='cpu')

    # Check all parameters are on correct device
    for param in model.parameters():
        assert param.device.type == 'cpu'

    # Forward pass should work
    output = model(mock_dataset)
    assert output.device.type == 'cpu'

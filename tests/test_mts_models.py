"""
Tests for multi-timescale (MTS) differentiable models.

Coverage:
- MtsDplModel initialization and forward pass
- Multi-scale data handling
- Low-frequency and high-frequency parameter paths
- MtsModelHandler functionality
- Gradient flow in multi-timescale models
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from dmg.core.utils import initialize_config, set_randomseed
from dmg.models.delta_models.mts_dpl_model import MtsDplModel
from dmg.models.mts_model_handler import MtsModelHandler

_MTS_SKIP_REASON = "Requires hydrodl2 with Hbv_2_mts model"


# ---------------------------------------------------------------------------
#  Model config and mock dataset.
# ---------------------------------------------------------------------------


@pytest.fixture
def mts_config():
    """Configuration for multi-timescale model testing."""
    config_dict = {
        'mode': 'train',
        'multimodel_type': 'none',
        'seed': 111111,
        'logging': 'tensorboard',
        'cache_states': False,
        'device': 'cpu',
        'gpu_id': 0,
        'data_loader': 'MtsHydroLoader',
        'data_sampler': 'MtsHydroSampler',
        'trainer': 'MsTrainer',
        'train': {
            'start_time': '2000/01/01',
            'end_time': '2000/01/31',
            'target': ['streamflow'],
            'optimizer': {'name': 'Adadelta'},
            'lr': 1.0,
            'lr_scheduler': {
                'name': 'StepLR',
                'step_size': 10,
                'gamma': 0.5,
            },
            'loss_function': {'name': 'RmseCombLoss'},
            'batch_size': 5,
            'epochs': 2,
            'start_epoch': 0,
            'save_epoch': 1,
        },
        'test': {
            'start_time': '2000/02/01',
            'end_time': '2000/02/10',
            'batch_size': 5,
            'test_epoch': 2,
        },
        'sim': {
            'start_time': '2000/02/01',
            'end_time': '2000/02/10',
            'batch_size': 5,
        },
        'model': {
            'rho': 10,
            'warm_up': 2,
            'use_log_norm': ['prcp'],
            'phy': {
                'name': ['Hbv_2_mts'],  # Multi-timescale physics model
                'nmul': 1,
                'warm_up_states': True,
                'dy_drop': 0.0,
                'dynamic_params': {
                    'Hbv_2_mts': ['parBETA', 'parBETAET'],
                },
                'routing': True,
                'nearzero': 1e-5,
                'forcings': ['prcp', 'tmean', 'pet'],
                'attributes': [],
                'cache_states': False,
            },
            'nn': {
                'name': 'StackLstmMlpModel',  # Multi-scale NN
                'dropout': 0.5,
                'hidden_size': 32,
                'hidden_size_low_freq': 16,
                'forcings': ['prcp', 'tmean', 'pet'],
                'attributes': ['area_gages2'],
                'regional_attributes': [],
                'cache_states': False,
            },
        },
        'observations': {
            'name': 'camels_531',
            'data_path': '',
            'area_name': 'area_gages2',
            'start_time': '2000/01/01',
            'end_time': '2000/02/28',
            'all_forcings': ['prcp', 'tmean', 'pet'],
            'all_attributes': ['area_gages2'],
        },
        'output_dir': 'tests/test_output/',
        'model_dir': 'tests/test_output/model/',
        'plot_dir': 'tests/test_output/plots/',
        'sim_dir': 'tests/test_output/sim/',
        'log_dir': 'tests/test_output/log/',
    }

    config_tmp = OmegaConf.create(config_dict)
    config = initialize_config(config_tmp, write_out=False)
    return config


@pytest.fixture
def mts_mock_dataset(mts_config):
    """Mock dataset for multi-timescale models."""
    set_randomseed(mts_config['seed'])

    n_basins = 10
    n_timesteps = 30

    # High-frequency inputs (e.g., hourly)
    n_timesteps_hf = n_timesteps * 24  # 24 hours per day

    return {
        # Low-frequency inputs
        'xc_nn_norm_low_freq': torch.rand(
            n_timesteps,
            n_basins,
            len(mts_config['model']['nn']['forcings'])
            + len(mts_config['model']['nn']['attributes']),
        ),
        # High-frequency inputs
        'xc_nn_norm_high_freq': torch.rand(
            n_timesteps_hf,
            n_basins,
            len(mts_config['model']['nn']['forcings']),
        ),
        # Static attributes
        'c_nn_norm': torch.rand(
            n_basins,
            len(mts_config['model']['nn']['attributes']),
        ),
        # Regional attributes (if any)
        'rc_nn_norm': torch.rand(n_basins, 0),  # Empty for this test
        # Physics model inputs
        'x_phy': torch.rand(
            n_timesteps,
            n_basins,
            len(mts_config['model']['phy']['forcings']),
        ),
        'c_phy': torch.rand(
            n_basins,
            len(mts_config['model']['phy']['attributes']),
        ),
        # Target
        'target': torch.rand(n_timesteps, n_basins, 1),
    }


# ---------------------------------------------------------------------------
#   Tests
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason=_MTS_SKIP_REASON)
class TestMtsDplModel:
    """Test MtsDplModel (multi-timescale differentiable model)."""

    def test_mts_dpl_model_initialization(self, mts_config):
        """Test MtsDplModel initializes correctly."""
        model = MtsDplModel(config=mts_config['model'], device='cpu')

        assert model.initialized
        assert model.nn_model is not None
        assert model.phy_model is not None

    def test_mts_dpl_model_forward_pass(self, mts_config, mts_mock_dataset):
        """Test forward pass with multi-timescale data."""
        set_randomseed(mts_config['seed'])

        model = MtsDplModel(config=mts_config['model'], device='cpu')

        output = model(mts_mock_dataset)

        assert isinstance(output, torch.Tensor)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_mts_separate_parameter_paths(self, mts_config, mts_mock_dataset):
        """Test that low/high-frequency parameters are generated separately."""
        set_randomseed(mts_config['seed'])

        model = MtsDplModel(config=mts_config['model'], device='cpu')

        # Check NN architecture supports multi-scale
        if type(model.nn_model).__name__ == 'StackLstmMlpModel':
            with torch.no_grad():
                params_lf, params_hf = model.nn_model(
                    mts_mock_dataset['xc_nn_norm_low_freq'],
                    mts_mock_dataset['xc_nn_norm_high_freq'],
                    mts_mock_dataset['c_nn_norm'],
                )

            # Verify parameters exist
            assert params_lf is not None
            assert params_hf is not None
            assert not torch.isnan(params_lf).any()
            assert not torch.isnan(params_hf).any()


@pytest.mark.skip(reason=_MTS_SKIP_REASON)
class TestMtsModelHandler:
    """Test MtsModelHandler functionality."""

    def test_mts_model_handler_initialization(self, mts_config):
        """Test MtsModelHandler initializes correctly."""
        handler = MtsModelHandler(mts_config)

        assert handler.model_type == 'dm'
        assert len(handler.model_dict) > 0

    def test_mts_model_handler_forward(self, mts_config, mts_mock_dataset):
        """Test MtsModelHandler forward pass."""
        set_randomseed(mts_config['seed'])

        handler = MtsModelHandler(mts_config)

        # Train mode
        handler.train()
        output_train = handler(mts_mock_dataset)

        assert isinstance(output_train, dict)
        assert len(output_train) > 0

        # Eval mode
        output_eval = handler(mts_mock_dataset, eval=True)

        assert isinstance(output_eval, dict)


@pytest.mark.skip(reason=_MTS_SKIP_REASON)
class TestMtsGradientFlow:
    """Test gradient flow in multi-timescale models."""

    def test_mts_gradient_flow(self, mts_config, mts_mock_dataset):
        """Verify gradients flow through both LF and HF paths."""
        set_randomseed(mts_config['seed'])

        handler = MtsModelHandler(mts_config)
        model = list(handler.model_dict.values())[0]

        model.train()

        # Forward pass
        output = model(mts_mock_dataset)
        loss = output.mean()

        # Backward pass
        loss.backward()

        # Check gradients exist
        for name, param in model.nn_model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any()

    def test_mts_parameter_update(self, mts_config, mts_mock_dataset):
        """Verify MTS model parameters update during training."""
        set_randomseed(mts_config['seed'])

        handler = MtsModelHandler(mts_config)
        model = list(handler.model_dict.values())[0]

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Store initial parameters
        initial_params = {
            name: param.clone().detach()
            for name, param in model.nn_model.named_parameters()
        }

        # Training step
        model.train()
        optimizer.zero_grad()
        output = model(mts_mock_dataset)
        loss = output.mean()
        loss.backward()
        optimizer.step()

        # Check parameters changed
        for name, param in model.nn_model.named_parameters():
            assert not torch.allclose(param, initial_params[name], atol=1e-8), (
                f"Parameter {name} did not update"
            )


class TestMtsDataHandling:
    """Test multi-timescale data handling."""

    def test_mts_dataset_structure(self, mts_mock_dataset):
        """Verify MTS dataset has correct structure."""
        required_keys = [
            'xc_nn_norm_low_freq',
            'xc_nn_norm_high_freq',
            'c_nn_norm',
            'target',
        ]

        for key in required_keys:
            assert key in mts_mock_dataset, f"Missing key: {key}"

    def test_mts_timescale_ratio(self, mts_mock_dataset):
        """Verify high-frequency data has correct temporal resolution."""
        lf_timesteps = mts_mock_dataset['xc_nn_norm_low_freq'].shape[0]
        hf_timesteps = mts_mock_dataset['xc_nn_norm_high_freq'].shape[0]

        # High-frequency should have more timesteps
        assert hf_timesteps > lf_timesteps, (
            "High-frequency data should have more timesteps than low-frequency"
        )

        # For this test, ratio is 24 (hourly vs daily)
        expected_ratio = 24
        actual_ratio = hf_timesteps / lf_timesteps

        assert actual_ratio == expected_ratio, (
            f"Expected ratio {expected_ratio}, got {actual_ratio}"
        )


@pytest.mark.skip(reason=_MTS_SKIP_REASON)
def test_mts_end_to_end_training_step(mts_config, mts_mock_dataset):
    """Test complete training step with MTS model."""
    from dmg.models.criterion.rmse_loss import RmseLoss

    set_randomseed(mts_config['seed'])

    # Initialize model and optimizer
    handler = MtsModelHandler(mts_config)
    model = list(handler.model_dict.values())[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = RmseLoss(mts_config)

    # Training loop
    losses = []
    model.train()

    for _ in range(5):
        optimizer.zero_grad()

        # Forward
        output = model(mts_mock_dataset)

        # Loss
        loss = loss_func(output, mts_mock_dataset['target'])

        # Backward
        loss.backward()

        # Update
        optimizer.step()

        losses.append(loss.item())

    # Verify training occurred
    assert all(not np.isnan(loss) for loss in losses)
    assert all(not np.isinf(loss) for loss in losses)

    # Loss should generally decrease
    initial_loss = np.mean(losses[:2])
    final_loss = np.mean(losses[-2:])

    assert final_loss <= initial_loss * 1.5, (
        "Loss did not decrease or increased too much"
    )

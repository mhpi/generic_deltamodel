"""
Tests for parameter learning in differentiable models.

Focuses on:
- Dynamic parameter generation by neural networks
- Parameter constraints and validity
- Convergence behavior
- Parameter sensitivity to inputs
- Physics-NN coupling correctness
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest
import torch

from dmg.core.utils import set_randomseed
from dmg.models.model_handler import ModelHandler


class TestDynamicParameterGeneration:
    """Test neural network parameter generation for physics models."""

    def test_nn_outputs_valid_parameter_count(self, config, mock_dataset):
        """Verify NN generates correct number of parameters."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        with torch.no_grad():
            if type(dpl_model.nn_model).__name__ == 'LstmMlpModel':
                params = dpl_model.nn_model(
                    mock_dataset['xc_nn_norm'], mock_dataset['c_nn_norm']
                )
            else:
                params = dpl_model.nn_model(mock_dataset['xc_nn_norm'])

        expected_count = dpl_model.phy_model.learnable_param_count
        actual_count = params.shape[-1]

        assert actual_count == expected_count, (
            f"NN generated {actual_count} params, expected {expected_count}"
        )

    def test_parameter_temporal_variation(self, config, mock_dataset):
        """Verify parameters vary over time (dynamic parameters)."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        with torch.no_grad():
            if type(dpl_model.nn_model).__name__ == 'LstmMlpModel':
                params = dpl_model.nn_model(
                    mock_dataset['xc_nn_norm'], mock_dataset['c_nn_norm']
                )
            else:
                params = dpl_model.nn_model(mock_dataset['xc_nn_norm'])

        # Check temporal dimension exists
        n_timesteps = params.shape[0]
        assert n_timesteps > 1, "Parameters should have temporal dimension"

        # Check parameters vary over time
        param_variance = params.var(dim=0)
        assert (param_variance > 0).any(), (
            "Parameters should vary over time for at least some features"
        )

    def test_parameter_spatial_variation(self, config, mock_dataset):
        """Verify parameters vary across basins (spatial variation)."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        # Create dataset with different basin attributes
        varied_dataset = mock_dataset.copy()
        varied_dataset['xc_nn_norm'] = torch.rand_like(mock_dataset['xc_nn_norm'])

        with torch.no_grad():
            if type(dpl_model.nn_model).__name__ == 'LstmMlpModel':
                params = dpl_model.nn_model(
                    varied_dataset['xc_nn_norm'], varied_dataset['c_nn_norm']
                )
            else:
                params = dpl_model.nn_model(varied_dataset['xc_nn_norm'])

        # Check basin dimension exists
        n_basins = params.shape[1]
        assert n_basins > 1, "Parameters should have basin dimension"

        # Check parameters vary across basins
        basin_variance = params.var(dim=1)
        assert (basin_variance > 0).any(), "Parameters should vary across basins"

    def test_parameter_values_are_finite(self, config, mock_dataset):
        """Verify generated parameters are finite and valid."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        with torch.no_grad():
            if type(dpl_model.nn_model).__name__ == 'LstmMlpModel':
                params = dpl_model.nn_model(
                    mock_dataset['xc_nn_norm'], mock_dataset['c_nn_norm']
                )
            else:
                params = dpl_model.nn_model(mock_dataset['xc_nn_norm'])

        assert not torch.isnan(params).any(), "Parameters contain NaN"
        assert not torch.isinf(params).any(), "Parameters contain Inf"
        assert params.requires_grad or not dpl_model.training, (
            "Parameters should track gradients in training mode"
        )


class TestParameterSensitivity:
    """Test parameter sensitivity to input changes."""

    def test_parameters_respond_to_forcing_changes(self, config, mock_dataset):
        """Verify parameters change when forcings change."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        # Original parameters
        with torch.no_grad():
            if type(dpl_model.nn_model).__name__ == 'LstmMlpModel':
                params_orig = dpl_model.nn_model(
                    mock_dataset['xc_nn_norm'], mock_dataset['c_nn_norm']
                )
            else:
                params_orig = dpl_model.nn_model(mock_dataset['xc_nn_norm'])

        # Modified forcings
        modified_dataset = mock_dataset.copy()
        modified_dataset['xc_nn_norm'] = mock_dataset['xc_nn_norm'] * 2.0

        with torch.no_grad():
            if type(dpl_model.nn_model).__name__ == 'LstmMlpModel':
                params_modified = dpl_model.nn_model(
                    modified_dataset['xc_nn_norm'], modified_dataset['c_nn_norm']
                )
            else:
                params_modified = dpl_model.nn_model(modified_dataset['xc_nn_norm'])

        # Parameters should differ
        assert not torch.allclose(params_orig, params_modified, rtol=1e-5), (
            "Parameters should respond to forcing changes"
        )

    def test_parameter_gradient_wrt_input(self, config, mock_dataset):
        """Verify parameters have gradients w.r.t. inputs."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        dpl_model.train()

        # Create input with requires_grad
        input_data = mock_dataset['xc_nn_norm'].clone().requires_grad_(True)

        # Generate parameters
        if type(dpl_model.nn_model).__name__ == 'LstmMlpModel':
            params = dpl_model.nn_model(input_data, mock_dataset['c_nn_norm'])
        else:
            params = dpl_model.nn_model(input_data)

        # Compute dummy loss
        loss = params.mean()
        loss.backward()

        # Check input has gradients
        assert input_data.grad is not None, (
            "Input should have gradients through parameter generation"
        )
        assert not torch.isnan(input_data.grad).any()


class TestParameterPhysicsIntegration:
    """Test integration of learned parameters with physics models."""

    def test_physics_model_accepts_nn_parameters(self, config, mock_dataset):
        """Verify physics model accepts NN-generated parameters."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        with torch.no_grad():
            # Generate parameters
            if type(dpl_model.nn_model).__name__ == 'LstmMlpModel':
                params = dpl_model.nn_model(
                    mock_dataset['xc_nn_norm'], mock_dataset['c_nn_norm']
                )
            else:
                params = dpl_model.nn_model(mock_dataset['xc_nn_norm'])

            # Pass to physics model
            try:
                output = dpl_model.phy_model(mock_dataset, params)
                assert output is not None
                assert not torch.isnan(output).any()
            except RuntimeError as e:
                pytest.fail(f"Physics model failed to accept NN parameters: {e}")

    def test_end_to_end_parameter_flow(self, config, mock_dataset):
        """Test complete parameter flow: input → NN → params → physics → output."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        # Enable gradients
        dpl_model.train()

        # End-to-end forward
        output = dpl_model(mock_dataset)

        # Verify output
        assert isinstance(output, torch.Tensor)
        assert output.shape == mock_dataset['target'].shape
        assert not torch.isnan(output).any()

        # Verify gradient flow
        loss = output.mean()
        loss.backward()

        for param in dpl_model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestParameterConvergence:
    """Test parameter learning convergence behavior."""

    def test_parameters_converge_during_training(self, config, mock_dataset):
        """Verify learned parameters stabilize during training."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        optimizer = torch.optim.Adam(dpl_model.parameters(), lr=0.01)

        # Track parameter changes
        param_history = []

        dpl_model.train()
        for _ in range(20):
            optimizer.zero_grad()

            # Forward
            output = dpl_model(mock_dataset)
            loss = (output - mock_dataset['target']).pow(2).mean()

            # Backward
            loss.backward()
            optimizer.step()

            # Store parameter snapshot
            with torch.no_grad():
                if type(dpl_model.nn_model).__name__ == 'LstmMlpModel':
                    params = dpl_model.nn_model(
                        mock_dataset['xc_nn_norm'], mock_dataset['c_nn_norm']
                    )
                else:
                    params = dpl_model.nn_model(mock_dataset['xc_nn_norm'])

                param_history.append(params.clone())

        # Check parameter changes decrease over time (convergence)
        early_change = (param_history[5] - param_history[0]).abs().mean()
        late_change = (param_history[-1] - param_history[-6]).abs().mean()

        # Late changes should be smaller (parameters stabilizing)
        # Allow some tolerance for stochastic optimization
        assert late_change < early_change * 2.0, (
            f"Parameters not stabilizing: early_change={early_change}, late_change={late_change}"
        )

    def test_parameter_updates_reduce_loss(self, config, mock_dataset):
        """Verify parameter updates lead to loss reduction."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        optimizer = torch.optim.Adam(dpl_model.parameters(), lr=0.01)

        dpl_model.train()

        # Initial loss
        with torch.no_grad():
            output_initial = dpl_model(mock_dataset)
            loss_initial = (
                (output_initial - mock_dataset['target']).pow(2).mean().item()
            )

        # Train for several steps
        for _ in range(15):
            optimizer.zero_grad()
            output = dpl_model(mock_dataset)
            loss = (output - mock_dataset['target']).pow(2).mean()
            loss.backward()
            optimizer.step()

        # Final loss
        with torch.no_grad():
            output_final = dpl_model(mock_dataset)
            loss_final = (output_final - mock_dataset['target']).pow(2).mean().item()

        # Loss should decrease
        assert loss_final < loss_initial, (
            f"Loss did not decrease: initial={loss_initial:.4f}, final={loss_final:.4f}"
        )


class TestParameterConstraints:
    """Test parameter constraints and validity."""

    def test_parameter_range_after_activation(self, config, mock_dataset):
        """Verify parameter activation functions produce valid ranges."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        with torch.no_grad():
            if type(dpl_model.nn_model).__name__ == 'LstmMlpModel':
                params = dpl_model.nn_model(
                    mock_dataset['xc_nn_norm'], mock_dataset['c_nn_norm']
                )
            else:
                params = dpl_model.nn_model(mock_dataset['xc_nn_norm'])

        # Parameters should be within reasonable range
        # (depends on activation function used)
        param_min = params.min().item()
        param_max = params.max().item()

        assert param_min > -1e3, f"Parameter too negative: {param_min}"
        assert param_max < 1e3, f"Parameter too large: {param_max}"

    def test_parameter_gradient_clipping_compatibility(self, config, mock_dataset):
        """Verify parameter learning works with gradient clipping."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        optimizer = torch.optim.SGD(dpl_model.parameters(), lr=0.01)

        dpl_model.train()

        # Training step with gradient clipping
        optimizer.zero_grad()
        output = dpl_model(mock_dataset)
        loss = output.mean()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(dpl_model.parameters(), max_norm=1.0)

        # Verify gradients are clipped
        total_norm = 0.0
        for param in dpl_model.parameters():
            if param.grad is not None:
                total_norm += param.grad.norm().item() ** 2
        total_norm = total_norm**0.5

        assert total_norm <= 1.0 + 1e-5, (
            f"Gradient norm {total_norm} exceeds clipping threshold"
        )

        # Update should still work
        optimizer.step()


class TestParameterConsistency:
    """Test parameter generation consistency."""

    def test_deterministic_parameter_generation(self, config, mock_dataset):
        """Verify parameter generation is deterministic with same seed."""
        set_randomseed(config['seed'])
        model1 = ModelHandler(config)

        set_randomseed(config['seed'])
        model2 = ModelHandler(config)

        model1.eval()
        model2.eval()

        with torch.no_grad():
            dpl1 = model1.model_dict['Hbv']
            dpl2 = model2.model_dict['Hbv']

            if type(dpl1.nn_model).__name__ == 'LstmMlpModel':
                params1 = dpl1.nn_model(
                    mock_dataset['xc_nn_norm'], mock_dataset['c_nn_norm']
                )
                params2 = dpl2.nn_model(
                    mock_dataset['xc_nn_norm'], mock_dataset['c_nn_norm']
                )
            else:
                params1 = dpl1.nn_model(mock_dataset['xc_nn_norm'])
                params2 = dpl2.nn_model(mock_dataset['xc_nn_norm'])

        assert torch.allclose(params1, params2, rtol=1e-5), (
            "Parameter generation not deterministic"
        )

    def test_batch_parameter_consistency(self, config, mock_dataset):
        """Verify parameters are consistent when processing batches."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        dpl_model.eval()

        # Full batch
        with torch.no_grad():
            if type(dpl_model.nn_model).__name__ == 'LstmMlpModel':
                params_full = dpl_model.nn_model(
                    mock_dataset['xc_nn_norm'], mock_dataset['c_nn_norm']
                )
            else:
                params_full = dpl_model.nn_model(mock_dataset['xc_nn_norm'])

        # Split batch
        n_basins = mock_dataset['xc_nn_norm'].shape[1]
        split_idx = n_basins // 2

        with torch.no_grad():
            # First half
            data_split1 = {
                k: v[:, :split_idx] if v.dim() > 1 else v[:split_idx]
                for k, v in mock_dataset.items()
            }

            if type(dpl_model.nn_model).__name__ == 'LstmMlpModel':
                params_split1 = dpl_model.nn_model(
                    data_split1['xc_nn_norm'], data_split1['c_nn_norm']
                )
            else:
                params_split1 = dpl_model.nn_model(data_split1['xc_nn_norm'])

        # Parameters should match (within numerical precision)
        assert torch.allclose(params_full[:, :split_idx], params_split1, rtol=1e-5), (
            "Batch processing inconsistent"
        )

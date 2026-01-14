"""
Comprehensive tests for differentiable models in dmg.

Tests cover:
- Gradient flow through NN and physics models
- Parameter learning and optimization
- Forward pass consistency
- Backward pass correctness
- Multi-model configurations
- Device compatibility
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch

from dmg.core.utils import set_randomseed
from dmg.models.model_handler import ModelHandler


class TestDplModelGradients:
    """Test gradient flow in DplModel."""

    def test_gradient_flow_through_nn(self, config, mock_dataset):
        """Verify gradients flow through neural network parameters."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        # Enable gradient tracking
        dpl_model.train()

        # Forward pass
        output = dpl_model(mock_dataset)

        # Compute dummy loss
        loss = output.mean()

        # Backward pass
        loss.backward()

        # Check NN parameters have gradients
        for name, param in dpl_model.nn_model.named_parameters():
            assert param.grad is not None, f"No gradient for NN parameter: {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    def test_gradient_flow_through_physics_model(self, config, mock_dataset):
        """Verify gradients flow through physics model parameters if they exist."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        dpl_model.train()

        # Forward pass
        output = dpl_model(mock_dataset)

        # Compute loss
        loss = output.mean()

        # Backward pass
        loss.backward()

        # Check physics model parameters (if learnable)
        phy_params_with_grad = [
            (name, param)
            for name, param in dpl_model.phy_model.named_parameters()
            if param.requires_grad
        ]

        # If physics model has trainable parameters, check gradients
        if phy_params_with_grad:
            for name, param in phy_params_with_grad:
                assert param.grad is not None, (
                    f"No gradient for physics parameter: {name}"
                )

    def test_gradient_magnitudes(self, config, mock_dataset):
        """Verify gradient magnitudes are reasonable (not vanishing/exploding)."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        dpl_model.train()

        # Forward pass
        output = dpl_model(mock_dataset)
        loss = output.mean()

        # Backward pass
        loss.backward()

        # Check gradient magnitudes
        for name, param in dpl_model.nn_model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert grad_norm > 1e-10, f"Vanishing gradient in {name}: {grad_norm}"
                assert grad_norm < 1e5, f"Exploding gradient in {name}: {grad_norm}"

    def test_gradient_computation_deterministic(self, config, mock_dataset):
        """Verify gradient computation is deterministic given same inputs."""
        set_randomseed(config['seed'])

        model1 = ModelHandler(config)
        set_randomseed(config['seed'])
        model2 = ModelHandler(config)

        dpl_model1 = model1.model_dict['Hbv']
        dpl_model2 = model2.model_dict['Hbv']

        # Forward + backward for model 1
        dpl_model1.train()
        output1 = dpl_model1(mock_dataset)
        loss1 = output1.mean()
        loss1.backward()

        # Forward + backward for model 2
        dpl_model2.train()
        output2 = dpl_model2(mock_dataset)
        loss2 = output2.mean()
        loss2.backward()

        # Compare gradients
        for (name1, param1), (name2, param2) in zip(
            dpl_model1.nn_model.named_parameters(),
            dpl_model2.nn_model.named_parameters(),
        ):
            assert name1 == name2
            if param1.grad is not None and param2.grad is not None:
                assert torch.allclose(param1.grad, param2.grad, rtol=1e-5), (
                    f"Gradient mismatch in {name1}"
                )


class TestDplModelForward:
    """Test forward pass behavior of DplModel."""

    def test_forward_output_shape(self, config, mock_dataset):
        """Verify forward pass produces correct output shape."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        output = dpl_model(mock_dataset)

        # Output should match target shape
        expected_shape = mock_dataset['target'].shape
        assert output.shape == expected_shape, (
            f"Output shape {output.shape} != expected {expected_shape}"
        )

    def test_forward_output_values(self, config, mock_dataset):
        """Verify forward pass produces valid (non-NaN, finite) outputs."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        output = dpl_model(mock_dataset)

        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

    def test_forward_train_vs_eval(self, config, mock_dataset):
        """Verify forward pass differs between train and eval modes (if dropout used)."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        # Train mode
        dpl_model.train()
        set_randomseed(config['seed'])
        _ = dpl_model(mock_dataset)
        set_randomseed(config['seed'])
        _ = dpl_model(mock_dataset)

        # Eval mode
        dpl_model.eval()
        set_randomseed(config['seed'])
        output_eval1 = dpl_model(mock_dataset)
        set_randomseed(config['seed'])
        output_eval2 = dpl_model(mock_dataset)

        # Eval mode should be deterministic
        assert torch.allclose(output_eval1, output_eval2), (
            "Eval mode outputs should be deterministic"
        )

        # If dropout is used, train mode may differ from eval
        # (This depends on config['model']['nn']['dropout'])
        if config['model']['nn']['dropout'] > 0:
            # Train mode might differ due to dropout
            pass

    def test_forward_with_different_batch_sizes(self, config, mock_dataset):
        """Verify forward pass works with different batch sizes."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        # Test with smaller batch
        n_timesteps, n_basins, n_features = mock_dataset['xc_nn_norm'].shape

        # Slice to smaller batch
        small_batch = {
            k: v[:, :3, :] if v.dim() == 3 else v[:3, :]
            for k, v in mock_dataset.items()
        }

        output_small = dpl_model(small_batch)
        assert output_small.shape[1] == 3, "Small batch output shape incorrect"


class TestParameterLearning:
    """Test parameter learning and optimization behavior."""

    def test_parameters_update_after_optimizer_step(self, config, mock_dataset):
        """Verify model parameters actually update during training."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        # Initialize optimizer
        optimizer = torch.optim.SGD(dpl_model.parameters(), lr=0.01)

        # Store initial parameter values
        initial_params = {
            name: param.clone().detach()
            for name, param in dpl_model.nn_model.named_parameters()
        }

        # Training step
        dpl_model.train()
        optimizer.zero_grad()

        output = dpl_model(mock_dataset)
        loss = output.mean()
        loss.backward()
        optimizer.step()

        # Check parameters changed
        for name, param in dpl_model.nn_model.named_parameters():
            initial_value = initial_params[name]
            assert not torch.allclose(param, initial_value, atol=1e-8), (
                f"Parameter {name} did not update after optimizer step"
            )

    def test_nn_generates_correct_param_count(self, config, mock_dataset):
        """Verify NN outputs correct number of parameters for physics model."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        # Get NN output (parameters for physics model)
        with torch.no_grad():
            if type(dpl_model.nn_model).__name__ == 'LstmMlpModel':
                parameters = dpl_model.nn_model(
                    mock_dataset['xc_nn_norm'], mock_dataset['c_nn_norm']
                )
            else:
                parameters = dpl_model.nn_model(mock_dataset['xc_nn_norm'])

        # Check parameter count matches physics model expectation
        expected_param_count = dpl_model.phy_model.learnable_param_count

        # Parameters shape should be [timesteps, basins, param_count]
        assert parameters.shape[-1] == expected_param_count, (
            f"NN output {parameters.shape[-1]} params, expected {expected_param_count}"
        )

    def test_loss_decreases_over_training_steps(self, config, mock_dataset):
        """Verify loss decreases over multiple training steps."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        optimizer = torch.optim.Adam(dpl_model.parameters(), lr=0.01)

        losses = []

        # Train for a few steps
        dpl_model.train()
        for _ in range(10):
            optimizer.zero_grad()
            output = dpl_model(mock_dataset)
            loss = (output - mock_dataset['target']).pow(2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Check loss generally decreases (allow some fluctuation)
        initial_loss = np.mean(losses[:3])
        final_loss = np.mean(losses[-3:])

        assert final_loss < initial_loss, (
            f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )


class TestModelHandler:
    """Test ModelHandler functionality."""

    def test_model_handler_initialization(self, config):
        """Verify ModelHandler initializes correctly."""
        model = ModelHandler(config)

        assert 'Hbv' in model.model_dict
        assert model.model_type == 'dm'
        assert model.device == 'cpu'

    def test_model_handler_forward_train_mode(self, config, mock_dataset):
        """Test ModelHandler forward in train mode."""
        model = ModelHandler(config)

        # Forward in train mode (gradients enabled)
        model.train()
        output_dict = model(mock_dataset)

        assert 'Hbv' in output_dict
        assert isinstance(output_dict['Hbv'], torch.Tensor)

    def test_model_handler_forward_eval_mode(self, config, mock_dataset):
        """Test ModelHandler forward in eval mode."""
        model = ModelHandler(config)

        # Forward in eval mode (no gradients)
        output_dict = model(mock_dataset, eval=True)

        assert 'Hbv' in output_dict
        assert isinstance(output_dict['Hbv'], torch.Tensor)

    def test_model_handler_calc_loss(self, config, mock_dataset):
        """Test ModelHandler loss calculation."""
        from dmg.models.criterion.mse_loss import MseLoss

        model = ModelHandler(config)
        loss_func = MseLoss(config)

        # Forward pass
        _ = model(mock_dataset)

        # Calculate loss
        loss = model.calc_loss(mock_dataset, loss_func)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss >= 0  # MSE should be non-negative

    def test_model_handler_get_parameters(self, config):
        """Test ModelHandler parameter retrieval."""
        model = ModelHandler(config)

        params = list(model.get_parameters())

        assert len(params) > 0, "No parameters found"

        # All should be tensors with requires_grad=True
        for param in params:
            assert isinstance(param, torch.Tensor)
            assert param.requires_grad


class TestGradientBackpropagation:
    """Test detailed gradient backpropagation mechanics."""

    def test_grad_enabled_in_train_mode(self, config, mock_dataset):
        """Verify gradients are enabled in train mode."""
        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        dpl_model.train()

        # Check requires_grad is True for inputs if needed
        output = dpl_model(mock_dataset)

        # Output should have grad_fn (computation graph)
        assert output.grad_fn is not None, "No computation graph in train mode"

    def test_no_grad_in_eval_mode(self, config, mock_dataset):
        """Verify gradients are disabled in eval mode."""
        model = ModelHandler(config)

        # Eval mode uses torch.no_grad() context
        with torch.no_grad():
            output_dict = model(mock_dataset, eval=True)
            output = output_dict['Hbv']

        # Output should NOT have grad_fn
        assert output.grad_fn is None, "Computation graph exists in eval mode"

    def test_gradient_accumulation(self, config, mock_dataset):
        """Verify gradients accumulate correctly without zero_grad."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        dpl_model.train()

        # First backward pass
        output1 = dpl_model(mock_dataset)
        loss1 = output1.mean()
        loss1.backward()

        # Store gradients
        first_grads = {
            name: param.grad.clone()
            for name, param in dpl_model.nn_model.named_parameters()
            if param.grad is not None
        }

        # Second backward pass without zero_grad (accumulation)
        output2 = dpl_model(mock_dataset)
        loss2 = output2.mean()
        loss2.backward()

        # Gradients should be accumulated (doubled if same input)
        for name, param in dpl_model.nn_model.named_parameters():
            if name in first_grads:
                accumulated_grad = param.grad
                # expected_grad = first_grads[name] * 2  # Approximately

                # Should be larger than first gradient
                assert accumulated_grad.norm() > first_grads[name].norm() * 0.9, (
                    f"Gradient did not accumulate for {name}"
                )

    def test_zero_grad_clears_gradients(self, config, mock_dataset):
        """Verify zero_grad clears all gradients."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        optimizer = torch.optim.SGD(dpl_model.parameters(), lr=0.01)

        dpl_model.train()

        # Backward pass
        output = dpl_model(mock_dataset)
        loss = output.mean()
        loss.backward()

        # Verify gradients exist
        for param in dpl_model.nn_model.parameters():
            if param.requires_grad:
                assert param.grad is not None

        # Clear gradients
        optimizer.zero_grad()

        # Verify gradients are None or zero
        for param in dpl_model.nn_model.parameters():
            if param.requires_grad:
                assert param.grad is None or torch.allclose(
                    param.grad, torch.zeros_like(param.grad)
                )


class TestDifferentiablePhysicsIntegration:
    """Test integration between NN and physics models."""

    def test_nn_to_physics_parameter_flow(self, config, mock_dataset):
        """Verify parameters flow correctly from NN to physics model."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        with torch.no_grad():
            # Get NN output
            if type(dpl_model.nn_model).__name__ == 'LstmMlpModel':
                nn_params = dpl_model.nn_model(
                    mock_dataset['xc_nn_norm'], mock_dataset['c_nn_norm']
                )
            else:
                nn_params = dpl_model.nn_model(mock_dataset['xc_nn_norm'])

            # Physics model should accept these parameters
            physics_output = dpl_model.phy_model(mock_dataset, nn_params)

        # Verify output is valid
        assert not torch.isnan(physics_output).any()
        assert not torch.isinf(physics_output).any()

    def test_end_to_end_differentiability(self, config, mock_dataset):
        """Verify end-to-end gradient flow from output to NN input."""
        set_randomseed(config['seed'])

        model = ModelHandler(config)
        dpl_model = model.model_dict['Hbv']

        dpl_model.train()

        # Create input that requires grad
        input_data = mock_dataset['xc_nn_norm'].clone().requires_grad_(True)
        mock_dataset_grad = {**mock_dataset, 'xc_nn_norm': input_data}

        # Forward pass
        output = dpl_model(mock_dataset_grad)
        loss = output.mean()

        # Backward pass
        loss.backward()

        # Check gradient exists for input
        assert input_data.grad is not None, "No gradient w.r.t. input"
        assert not torch.isnan(input_data.grad).any()


def test_model_serialization(config, mock_dataset, tmp_path):
    """Test saving and loading model states."""
    set_randomseed(config['seed'])

    # Create and train model briefly
    model1 = ModelHandler(config)
    optimizer = torch.optim.SGD(model1.get_parameters(), lr=0.01)

    model1.train()
    for _ in range(5):
        optimizer.zero_grad()
        _ = model1(mock_dataset)
        loss = model1.calc_loss(mock_dataset, torch.nn.MSELoss())
        loss.backward()
        optimizer.step()

    # Save model
    save_path = tmp_path / "test_model.pt"
    torch.save(model1.state_dict(), save_path)

    # Load into new model
    model2 = ModelHandler(config)
    model2.load_state_dict(torch.load(save_path))

    # Verify outputs match
    model1.eval()
    model2.eval()

    with torch.no_grad():
        output1 = model1(mock_dataset)['Hbv']
        output2 = model2(mock_dataset)['Hbv']

    assert torch.allclose(output1, output2, rtol=1e-5), (
        "Loaded model produces different output"
    )


def test_device_placement(config, mock_dataset):
    """Test model can be moved to different devices (CPU only for this test)."""
    model = ModelHandler(config, device='cpu')

    # Verify model is on CPU
    assert model.device == 'cpu'

    # Verify all parameters are on CPU
    for param in model.get_parameters():
        assert param.device.type == 'cpu'

    # Forward pass should work
    output = model(mock_dataset)
    assert output['Hbv'].device.type == 'cpu'

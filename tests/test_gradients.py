"""
Tests for gradient flow and backpropagation in differentiable models.

Coverage:
- Gradient flow through NN and physics models
- Gradient magnitudes (vanishing/exploding detection)
- End-to-end differentiability (output -> NN input)
- Gradient computation determinism
- Gradient accumulation and zero_grad behavior
- Gradient clipping compatibility

NOTE: All tests are parametrized over Hbv, Hbv_1_1p, and Hbv_2.
"""

import torch

from dmg.core.utils import set_randomseed
from dmg.models.model_handler import ModelHandler
from tests import _skip_if_zero_streamflow, compute_mse_loss, get_phy_model_name


class TestGradientFlow:
    """Test gradient flow through the differentiable model pipeline."""

    def test_gradient_flow_through_nn(self, model_config, model_dataset):
        """Verify gradients flow through neural network parameters."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        model = ModelHandler(model_config)
        dpl_model = model.model_dict[model_name]

        dpl_model.train()
        warm_up = model_config['model']['warm_up']
        loss, _ = compute_mse_loss(dpl_model, model_dataset, warm_up)
        loss.backward()

        for name, param in dpl_model.nn_model.named_parameters():
            assert param.grad is not None, f"No gradient for NN parameter: {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    def test_gradient_flow_through_physics_model(
        self,
        model_config,
        model_dataset,
    ):
        """Verify gradients flow through physics model parameters if any."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        model = ModelHandler(model_config)
        dpl_model = model.model_dict[model_name]

        dpl_model.train()
        warm_up = model_config['model']['warm_up']
        loss, _ = compute_mse_loss(dpl_model, model_dataset, warm_up)
        loss.backward()

        phy_params_with_grad = [
            (name, param)
            for name, param in dpl_model.phy_model.named_parameters()
            if param.requires_grad
        ]
        if phy_params_with_grad:
            for name, param in phy_params_with_grad:
                assert param.grad is not None, (
                    f"No gradient for physics parameter: {name}"
                )

    def test_gradient_magnitudes(self, model_config, model_dataset):
        """Verify gradient magnitudes are reasonable (not vanishing/exploding)."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        model = ModelHandler(model_config)
        dpl_model = model.model_dict[model_name]
        _skip_if_zero_streamflow(dpl_model, model_dataset)

        dpl_model.train()
        warm_up = model_config['model']['warm_up']
        loss, _ = compute_mse_loss(dpl_model, model_dataset, warm_up)
        loss.backward()

        nonzero_count = 0
        total_count = 0
        for name, param in dpl_model.nn_model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert grad_norm < 1e5, f"Exploding gradient in {name}: {grad_norm}"
                if grad_norm > 1e-10:
                    nonzero_count += 1
                total_count += 1

        assert nonzero_count > 0, "All gradients are zero"
        assert nonzero_count / total_count > 0.5, (
            f"Too many vanishing gradients: {nonzero_count}/{total_count}"
        )

    def test_end_to_end_differentiability(self, model_config, model_dataset):
        """Verify end-to-end gradient flow from output to NN input."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        model = ModelHandler(model_config)
        dpl_model = model.model_dict[model_name]
        dpl_model.train()

        input_data = model_dataset['xc_nn_norm'].clone().requires_grad_(True)
        dataset_grad = {**model_dataset, 'xc_nn_norm': input_data}

        output = dpl_model(dataset_grad)
        loss = output['streamflow'].mean()
        loss.backward()

        assert input_data.grad is not None, "No gradient w.r.t. input"
        assert not torch.isnan(input_data.grad).any()


class TestGradientMechanics:
    """Test gradient computation mechanics (determinism, accumulation, etc.)."""

    def test_gradient_computation_deterministic(
        self,
        model_config,
        model_dataset,
    ):
        """Verify gradient computation is deterministic given same inputs."""
        model_name = get_phy_model_name(model_config)
        warm_up = model_config['model']['warm_up']

        set_randomseed(model_config['seed'])
        model1 = ModelHandler(model_config)
        set_randomseed(model_config['seed'])
        model2 = ModelHandler(model_config)

        dpl1 = model1.model_dict[model_name]
        dpl2 = model2.model_dict[model_name]

        # Use eval mode to avoid dropout non-determinism.
        dpl1.eval()
        dpl2.eval()

        # But we still need grad computation, so use enable_grad.
        with torch.enable_grad():
            loss1, _ = compute_mse_loss(dpl1, model_dataset, warm_up)
            loss1.backward()

            loss2, _ = compute_mse_loss(dpl2, model_dataset, warm_up)
            loss2.backward()

        for (name1, p1), (name2, p2) in zip(
            dpl1.nn_model.named_parameters(),
            dpl2.nn_model.named_parameters(),
        ):
            assert name1 == name2
            if p1.grad is not None and p2.grad is not None:
                assert torch.allclose(p1.grad, p2.grad, rtol=1e-4), (
                    f"Gradient mismatch in {name1}"
                )

    def test_grad_enabled_in_train_mode(self, model_config, model_dataset):
        """Verify computation graph exists in train mode."""
        model_name = get_phy_model_name(model_config)
        model = ModelHandler(model_config)
        dpl_model = model.model_dict[model_name]

        dpl_model.train()
        output = dpl_model(model_dataset)

        assert output['streamflow'].grad_fn is not None, (
            "No computation graph in train mode"
        )

    def test_no_grad_in_eval_mode(self, model_config, model_dataset):
        """Verify gradients are disabled in eval mode."""
        model_name = get_phy_model_name(model_config)
        model = ModelHandler(model_config)

        with torch.no_grad():
            output_dict = model(model_dataset, eval=True)
            output = output_dict[model_name]['streamflow']

        assert output.grad_fn is None, "Computation graph exists in eval mode"

    def test_gradient_accumulation(self, model_config, model_dataset):
        """Verify gradients accumulate correctly without zero_grad."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)
        warm_up = model_config['model']['warm_up']

        model = ModelHandler(model_config)
        dpl_model = model.model_dict[model_name]
        _skip_if_zero_streamflow(dpl_model, model_dataset)
        dpl_model.train()

        loss1, _ = compute_mse_loss(dpl_model, model_dataset, warm_up)
        loss1.backward()

        first_grads = {
            name: param.grad.clone()
            for name, param in dpl_model.nn_model.named_parameters()
            if param.grad is not None and param.grad.norm() > 1e-10
        }

        loss2, _ = compute_mse_loss(dpl_model, model_dataset, warm_up)
        loss2.backward()

        accumulated = 0
        for name, param in dpl_model.nn_model.named_parameters():
            if name in first_grads:
                if param.grad.norm() > first_grads[name].norm() * 0.9:
                    accumulated += 1

        assert len(first_grads) > 0, "No parameters had non-zero gradients"
        assert accumulated > 0, "No gradients accumulated across backward passes"
        assert accumulated / len(first_grads) > 0.5, (
            f"Too few gradients accumulated: {accumulated}/{len(first_grads)}"
        )

    def test_zero_grad_clears_gradients(self, model_config, model_dataset):
        """Verify zero_grad clears all gradients."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        model = ModelHandler(model_config)
        dpl_model = model.model_dict[model_name]
        optimizer = torch.optim.SGD(dpl_model.parameters(), lr=0.01)

        dpl_model.train()
        output = dpl_model(model_dataset)
        loss = output['streamflow'].mean()
        loss.backward()

        for param in dpl_model.nn_model.parameters():
            if param.requires_grad:
                assert param.grad is not None

        optimizer.zero_grad()

        for param in dpl_model.nn_model.parameters():
            if param.requires_grad:
                assert param.grad is None or torch.allclose(
                    param.grad,
                    torch.zeros_like(param.grad),
                )

    def test_gradient_clipping_compatibility(
        self,
        model_config,
        model_dataset,
    ):
        """Verify parameter learning works with gradient clipping."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        model = ModelHandler(model_config)
        dpl_model = model.model_dict[model_name]
        optimizer = torch.optim.SGD(dpl_model.parameters(), lr=0.01)

        dpl_model.train()
        optimizer.zero_grad()
        output = dpl_model(model_dataset)
        loss = output['streamflow'].mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(dpl_model.parameters(), max_norm=1.0)

        total_norm = 0.0
        for param in dpl_model.parameters():
            if param.grad is not None:
                total_norm += param.grad.norm().item() ** 2
        total_norm = total_norm**0.5

        assert total_norm <= 1.0 + 1e-5, (
            f"Gradient norm {total_norm} exceeds clipping threshold"
        )

        optimizer.step()

"""
Tests for training convergence and physics integration.

Coverage:
- Parameter stabilization during training
- Loss reduction through parameter updates
- Physics model acceptance of NN-generated parameters
- End-to-end parameter flow (input -> NN -> params -> physics -> output)

NOTE: All tests are parametrized over Hbv, Hbv_1_1p, and Hbv_2.
"""

import torch

from dmg.core.utils import set_randomseed
from dmg.models.model_handler import ModelHandler
from tests import _get_nn_params, _skip_if_zero_streamflow, get_phy_model_name

# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------


class TestParameterConvergence:
    """Test parameter learning convergence behavior."""

    def test_parameters_converge_during_training(
        self,
        model_config,
        model_dataset,
    ):
        """Verify learned parameters stabilize during training."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        model = ModelHandler(model_config)
        dpl_model = model.model_dict[model_name]
        _skip_if_zero_streamflow(dpl_model, model_dataset)
        optimizer = torch.optim.Adam(dpl_model.parameters(), lr=0.01)

        warm_up = model_config['model']['warm_up']
        target = model_dataset['target'][warm_up:]

        param_history = []
        dpl_model.train()
        for _ in range(20):
            optimizer.zero_grad()
            output = dpl_model(model_dataset)
            sf = output['streamflow']
            n = min(sf.shape[0], target.shape[0])
            loss = (sf[:n] - target[:n]).pow(2).mean()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                params = _get_nn_params(dpl_model, model_dataset)
                if isinstance(params, tuple):
                    params = params[0]
                param_history.append(params.clone())

        early_change = (param_history[5] - param_history[0]).abs().mean()
        late_change = (param_history[-1] - param_history[-6]).abs().mean()

        assert late_change < early_change * 2.0, (
            f"Parameters not stabilizing: early={early_change}, late={late_change}"
        )

    def test_parameter_updates_reduce_loss(
        self,
        model_config,
        model_dataset,
    ):
        """Verify parameter updates lead to loss reduction."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        model = ModelHandler(model_config)
        dpl_model = model.model_dict[model_name]
        _skip_if_zero_streamflow(dpl_model, model_dataset)
        optimizer = torch.optim.Adam(dpl_model.parameters(), lr=0.01)

        warm_up = model_config['model']['warm_up']
        target = model_dataset['target'][warm_up:]

        dpl_model.train()

        with torch.no_grad():
            out_initial = dpl_model(model_dataset)
            sf = out_initial['streamflow']
            n = min(sf.shape[0], target.shape[0])
            loss_initial = (sf[:n] - target[:n]).pow(2).mean().item()

        for _ in range(15):
            optimizer.zero_grad()
            output = dpl_model(model_dataset)
            sf = output['streamflow']
            n = min(sf.shape[0], target.shape[0])
            loss = (sf[:n] - target[:n]).pow(2).mean()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            out_final = dpl_model(model_dataset)
            sf = out_final['streamflow']
            n = min(sf.shape[0], target.shape[0])
            loss_final = (sf[:n] - target[:n]).pow(2).mean().item()

        assert loss_final < loss_initial, (
            f"Loss did not decrease: initial={loss_initial:.4f}, final={loss_final:.4f}"
        )


class TestPhysicsIntegration:
    """Test integration of NN-generated parameters with physics models."""

    def test_physics_model_accepts_nn_parameters(
        self,
        model_config,
        model_dataset,
    ):
        """Verify physics model accepts NN-generated parameters and produces
        valid output."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        model = ModelHandler(model_config)
        dpl_model = model.model_dict[model_name]

        with torch.no_grad():
            if type(dpl_model.nn_model).__name__ == 'LstmMlpModel':
                nn_params = dpl_model.nn_model(
                    model_dataset['xc_nn_norm'],
                    model_dataset['c_nn_norm'],
                )
            else:
                nn_params = dpl_model.nn_model(model_dataset['xc_nn_norm'])

            physics_output = dpl_model.phy_model(model_dataset, nn_params)
            if isinstance(physics_output, tuple):
                flux_dict = physics_output[0]
            else:
                flux_dict = physics_output

        for key, val in flux_dict.items():
            if isinstance(val, torch.Tensor):
                assert not torch.isnan(val).any(), f"NaN in '{key}'"
                assert not torch.isinf(val).any(), f"Inf in '{key}'"

    def test_end_to_end_parameter_flow(self, model_config, model_dataset):
        """Test complete flow: input -> NN -> params -> physics -> output."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        model = ModelHandler(model_config)
        dpl_model = model.model_dict[model_name]
        dpl_model.train()

        output = dpl_model(model_dataset)

        assert isinstance(output, dict)
        assert 'streamflow' in output
        assert not torch.isnan(output['streamflow']).any()

        loss = output['streamflow'].mean()
        loss.backward()

        for param in dpl_model.parameters():
            if param.requires_grad:
                assert param.grad is not None

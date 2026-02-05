"""
Tests for neural network parameter generation.

Coverage:
- Parameter count validation against physics model requirements
- Parameter shape and temporal dimension
- Temporal and spatial variation
- Sensitivity to input changes
- Value constraints and ranges
- Deterministic generation
- Batch consistency

NOTE: All tests are parametrized over Hbv, Hbv_1_1p, and Hbv_2.
"""

import torch

from dmg.core.utils import set_randomseed
from dmg.models.model_handler import ModelHandler
from tests import _get_nn_params, get_phy_model_name

# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------


class TestParameterGeneration:
    """Test neural network parameter generation for physics models."""

    def test_nn_outputs_valid_parameter_count(
        self,
        model_config,
        model_dataset,
    ):
        """Verify NN generates the parameter count the physics model expects."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        model = ModelHandler(model_config)
        dpl_model = model.model_dict[model_name]
        params = _get_nn_params(dpl_model, model_dataset)

        expected_count = dpl_model.phy_model.learnable_param_count
        if isinstance(params, tuple):
            actual_count = sum(p.shape[-1] for p in params)
        else:
            actual_count = params.shape[-1]

        assert actual_count == expected_count, (
            f"NN generated {actual_count} params, expected {expected_count}"
        )

    def test_nn_output_temporal_dimension(self, model_config, model_dataset):
        """NN output must have correct temporal and basin dimensions."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        handler = ModelHandler(model_config)
        dpl_model = handler.model_dict[model_name]
        params = _get_nn_params(dpl_model, model_dataset)

        n_timesteps = model_dataset['xc_nn_norm'].shape[0]
        n_basins = model_dataset['xc_nn_norm'].shape[1]

        if isinstance(params, tuple):
            # Dynamic params (LSTM output) should have temporal dimension.
            assert params[0].shape[0] == n_timesteps, (
                f"Dynamic param temporal dim: expected {n_timesteps}, "
                f"got {params[0].shape[0]}"
            )
            assert params[0].shape[1] == n_basins, (
                f"Dynamic param basin dim: expected {n_basins}, "
                f"got {params[0].shape[1]}"
            )
            # Static params (MLP output) should have basin dimension only.
            assert params[1].shape[0] == n_basins, (
                f"Static param basin dim: expected {n_basins}, got {params[1].shape[0]}"
            )
        else:
            assert params.shape[0] == n_timesteps, (
                f"Param temporal dim: expected {n_timesteps}, got {params.shape[0]}"
            )
            assert params.shape[1] == n_basins, (
                f"Param basin dim: expected {n_basins}, got {params.shape[1]}"
            )

    def test_parameter_temporal_variation(self, model_config, model_dataset):
        """Verify parameters vary over time (dynamic parameters)."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        model = ModelHandler(model_config)
        dpl_model = model.model_dict[model_name]
        params = _get_nn_params(dpl_model, model_dataset)

        # For LstmMlpModel, check the LSTM (temporal) component.
        if isinstance(params, tuple):
            params = params[0]

        n_timesteps = params.shape[0]
        assert n_timesteps > 1, "Parameters should have temporal dimension"

        param_variance = params.var(dim=0)
        assert (param_variance > 0).any(), (
            "Parameters should vary over time for at least some features"
        )

    def test_parameter_spatial_variation(self, model_config, model_dataset):
        """Verify parameters vary across basins (spatial variation)."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        model = ModelHandler(model_config)
        dpl_model = model.model_dict[model_name]

        # Use randomized dataset to ensure spatial variation.
        varied_dataset = model_dataset.copy()
        varied_dataset['xc_nn_norm'] = torch.rand_like(
            model_dataset['xc_nn_norm'],
        )

        params = _get_nn_params(dpl_model, varied_dataset)
        if isinstance(params, tuple):
            params = params[0]

        n_basins = params.shape[1]
        assert n_basins > 1, "Parameters should have basin dimension"

        basin_variance = params.var(dim=1)
        assert (basin_variance > 0).any(), "Parameters should vary across basins"

    def test_parameter_values_are_finite(self, model_config, model_dataset):
        """Verify generated parameters are finite and valid."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        model = ModelHandler(model_config)
        dpl_model = model.model_dict[model_name]
        params = _get_nn_params(dpl_model, model_dataset)

        if isinstance(params, tuple):
            for p in params:
                assert not torch.isnan(p).any(), "Parameters contain NaN"
                assert not torch.isinf(p).any(), "Parameters contain Inf"
        else:
            assert not torch.isnan(params).any(), "Parameters contain NaN"
            assert not torch.isinf(params).any(), "Parameters contain Inf"


class TestParameterSensitivity:
    """Test parameter sensitivity to input changes."""

    def test_parameters_respond_to_forcing_changes(
        self,
        model_config,
        model_dataset,
    ):
        """Verify parameters change when forcings change."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        model = ModelHandler(model_config)
        dpl_model = model.model_dict[model_name]

        params_orig = _get_nn_params(dpl_model, model_dataset)

        modified_dataset = model_dataset.copy()
        modified_dataset['xc_nn_norm'] = model_dataset['xc_nn_norm'] * 2.0
        params_modified = _get_nn_params(dpl_model, modified_dataset)

        if isinstance(params_orig, tuple):
            assert not torch.allclose(
                params_orig[0],
                params_modified[0],
                rtol=1e-5,
            )
        else:
            assert not torch.allclose(
                params_orig,
                params_modified,
                rtol=1e-5,
            )


class TestParameterConstraints:
    """Test parameter constraints and validity."""

    def test_parameter_range_after_activation(
        self,
        model_config,
        model_dataset,
    ):
        """Verify parameter activation functions produce valid ranges."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        model = ModelHandler(model_config)
        dpl_model = model.model_dict[model_name]
        params = _get_nn_params(dpl_model, model_dataset)

        if isinstance(params, tuple):
            for p in params:
                assert p.min().item() > -1e3, f"Parameter too negative: {p.min()}"
                assert p.max().item() < 1e3, f"Parameter too large: {p.max()}"
        else:
            assert params.min().item() > -1e3, f"Parameter too negative: {params.min()}"
            assert params.max().item() < 1e3, f"Parameter too large: {params.max()}"


class TestParameterConsistency:
    """Test parameter generation consistency."""

    def test_deterministic_parameter_generation(
        self,
        model_config,
        model_dataset,
    ):
        """Verify parameter generation is deterministic with same seed."""
        set_randomseed(model_config['seed'])
        model1 = ModelHandler(model_config)

        set_randomseed(model_config['seed'])
        model2 = ModelHandler(model_config)

        model_name = get_phy_model_name(model_config)
        # Call eval() on DplModel directly (ModelHandler.model_dict is a plain
        # dict, so Module.eval() doesn't propagate to its children).
        model1.model_dict[model_name].eval()
        model2.model_dict[model_name].eval()

        with torch.no_grad():
            p1 = _get_nn_params(model1.model_dict[model_name], model_dataset)
            p2 = _get_nn_params(model2.model_dict[model_name], model_dataset)

        if isinstance(p1, tuple):
            for a, b in zip(p1, p2):
                assert torch.allclose(a, b, rtol=1e-5), (
                    "Parameter generation not deterministic"
                )
        else:
            assert torch.allclose(p1, p2, rtol=1e-5), (
                "Parameter generation not deterministic"
            )

    def test_batch_parameter_consistency(self, model_config, model_dataset):
        """Verify parameters are consistent when processing batches."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        model = ModelHandler(model_config)
        dpl_model = model.model_dict[model_name]
        dpl_model.eval()

        with torch.no_grad():
            params_full = _get_nn_params(dpl_model, model_dataset)

        n_basins = model_dataset['xc_nn_norm'].shape[1]
        split_idx = n_basins // 2

        data_split = {}
        for k, v in model_dataset.items():
            if isinstance(v, torch.Tensor):
                if v.dim() == 3:
                    data_split[k] = v[:, :split_idx, :]
                elif v.dim() == 2:
                    data_split[k] = v[:split_idx, :]
                elif v.dim() == 1:
                    data_split[k] = v[:split_idx]
                else:
                    data_split[k] = v
            else:
                data_split[k] = v

        with torch.no_grad():
            params_split = _get_nn_params(dpl_model, data_split)

        if isinstance(params_full, tuple):
            assert torch.allclose(
                params_full[0][:, :split_idx],
                params_split[0],
                rtol=1e-5,
            ), "Batch processing inconsistent"
        else:
            assert torch.allclose(
                params_full[:, :split_idx],
                params_split,
                rtol=1e-5,
            ), "Batch processing inconsistent"

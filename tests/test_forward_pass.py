"""
Tests for DplModel forward-pass behavior.

Coverage:
- Output structure (dict type, keys, tensor shapes)
- Output validity (finite values, no NaN/Inf)
- Eval mode determinism
- Batch size flexibility
- Runtime state snapshot and restore for stepwise inference
  (get/load_runtime_state, Hbv_2 with LstmMlpModel)

NOTE: Tests other than the runtime state tests are parametrized over Hbv,
Hbv_1_1p, and Hbv_2.
"""

import pytest
import torch

from dmg.core.utils import set_randomseed
from dmg.models.model_handler import ModelHandler
from tests import get_phy_model_name

# ---------------------------------------------------------------------------- #
#  Expected output keys for each model
# ---------------------------------------------------------------------------- #

EXP_FLUX_KEYS = {
    'Hbv': [
        'AET_hydro',
        'BFI',
        'PET_hydro',
        'SWE',
        'evapfactor',
        'excs',
        'gwflow',
        'gwflow_no_rout',
        'percolation',
        'recharge',
        'srflow',
        'srflow_no_rout',
        'ssflow',
        'ssflow_no_rout',
        'streamflow',
        'streamflow_no_rout',
        'tosoil',
    ],
    'Hbv_1_1p': [
        'AET_hydro',
        'BFI',
        'PET_hydro',
        'SWE',
        'capillary',
        'evapfactor',
        'excs',
        'gwflow',
        'gwflow_no_rout',
        'percolation',
        'recharge',
        'srflow',
        'srflow_no_rout',
        'ssflow',
        'ssflow_no_rout',
        'streamflow',
        'streamflow_no_rout',
        'tosoil',
    ],
    'Hbv_2': [
        'AET_hydro',
        'BFI',
        'PET_hydro',
        'SWE',
        'capillary',
        'evapfactor',
        'excs',
        'gwflow',
        'gwflow_no_rout',
        'percolation',
        'recharge',
        'srflow',
        'srflow_no_rout',
        'ssflow',
        'ssflow_no_rout',
        'streamflow',
        'streamflow_no_rout',
        'tosoil',
    ],
}


# ---------------------------------------------------------------------------- #
#  Tests
# ---------------------------------------------------------------------------- #


class TestForwardPassOutput:
    """Verify DplModel forward-pass output structure and validity."""

    def test_output_is_dict(self, model_config, model_dataset):
        """Verify forward pass returns a dict of flux tensors."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        model = ModelHandler(model_config)
        dpl_model = model.model_dict[model_name]

        output = dpl_model(model_dataset)

        assert isinstance(output, dict), (
            f"Expected dict output, got {type(output).__name__}"
        )
        assert 'streamflow' in output, "Output missing 'streamflow' key"
        assert isinstance(output['streamflow'], torch.Tensor)
        assert output['streamflow'].dim() == 3  # [time, basins, 1]

    def test_output_keys_match_expected(self, model_config, model_dataset):
        """Forward pass output dict keys must match expected flux keys."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        handler = ModelHandler(model_config)
        dpl_model = handler.model_dict[model_name]
        dpl_model.eval()

        with torch.no_grad():
            output = dpl_model(model_dataset)

        expected_keys = set(EXP_FLUX_KEYS[model_name])
        actual_keys = set(output.keys())

        assert actual_keys == expected_keys, (
            f"Output key mismatch for {model_name}.\n"
            f"  Missing: {expected_keys - actual_keys}\n"
            f"  Extra: {actual_keys - expected_keys}"
        )

    def test_streamflow_key_always_present(self, model_config, model_dataset):
        """Streamflow must always be in the output."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        handler = ModelHandler(model_config)
        dpl_model = handler.model_dict[model_name]
        dpl_model.eval()

        with torch.no_grad():
            output = dpl_model(model_dataset)

        assert 'streamflow' in output, "streamflow key missing from output"

    def test_output_tensor_shapes(self, model_config, model_dataset):
        """Output tensors must have consistent shapes (timesteps, basins, 1)."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        handler = ModelHandler(model_config)
        dpl_model = handler.model_dict[model_name]
        dpl_model.eval()

        with torch.no_grad():
            output = dpl_model(model_dataset)

        n_basins = model_dataset['xc_nn_norm'].shape[1]

        for key, tensor in output.items():
            if key == 'BFI':
                # BFI is per-basin, not per-timestep.
                assert tensor.shape == (n_basins,), (
                    f"BFI shape mismatch: expected ({n_basins},), got {tensor.shape}"
                )
            else:
                assert tensor.dim() == 3, (
                    f"{key} should be 3D (timesteps, basins, 1), got {tensor.dim()}D"
                )
                assert tensor.shape[1] == n_basins, (
                    f"{key} basin dim mismatch: expected {n_basins}, "
                    f"got {tensor.shape[1]}"
                )
                assert tensor.shape[2] == 1, (
                    f"{key} last dim should be 1, got {tensor.shape[2]}"
                )

    def test_output_values_finite(self, model_config, model_dataset):
        """All output values must be finite (no NaN or Inf)."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        handler = ModelHandler(model_config)
        dpl_model = handler.model_dict[model_name]
        dpl_model.eval()

        with torch.no_grad():
            output = dpl_model(model_dataset)

        for key, tensor in output.items():
            assert not torch.isnan(tensor).any(), f"{key} contains NaN values"
            assert not torch.isinf(tensor).any(), f"{key} contains Inf values"

    def test_eval_mode_deterministic(self, model_config, model_dataset):
        """Verify eval mode forward is deterministic."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        model = ModelHandler(model_config)
        dpl_model = model.model_dict[model_name]

        dpl_model.eval()
        with torch.no_grad():
            out1 = dpl_model(model_dataset)
            out2 = dpl_model(model_dataset)

        assert torch.allclose(out1['streamflow'], out2['streamflow']), (
            "Eval mode outputs should be deterministic"
        )

    def test_different_batch_sizes(self, model_config, model_dataset):
        """Verify forward pass works with different batch sizes."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        model = ModelHandler(model_config)
        dpl_model = model.model_dict[model_name]

        # Slice to a smaller batch (3 basins).
        small_batch = {}
        for k, v in model_dataset.items():
            if isinstance(v, torch.Tensor):
                if v.dim() == 3:
                    small_batch[k] = v[:, :3, :]
                elif v.dim() == 2:
                    small_batch[k] = v[:3, :]
                elif v.dim() == 1:
                    small_batch[k] = v[:3]
                else:
                    small_batch[k] = v
            else:
                small_batch[k] = v

        output = dpl_model(small_batch)
        assert output['streamflow'].shape[1] == 3, "Small batch output shape incorrect"


# ---------------------------------------------------------------------------- #
#  Runtime state
# ---------------------------------------------------------------------------- #


class TestRuntimeState:
    """Test runtime state snapshot and restore for stepwise inference."""

    TIME_KEYS = ('xc_nn_norm', 'x_phy')

    @staticmethod
    def _stepwise_model(config, state_dict=None):
        """Build an Hbv_2 dPL model set up for stepwise inference."""
        set_randomseed(config['seed'])
        model = ModelHandler(config).model_dict[get_phy_model_name(config)]
        if state_dict is not None:
            model.load_state_dict(state_dict)
        model.eval()
        model.phy_model.cache_states = True
        model.nn_model.cache_states = True
        model.nn_model.lstminv.cache_states = True
        return model

    def _slice(self, dataset, start, stop):
        """Slice the time-dependent inputs of a dataset."""
        return {
            k: v[start:stop] if k in self.TIME_KEYS else v for k, v in dataset.items()
        }

    def test_resume_matches_continuous(self, config_hbv_2, mock_dataset_hbv_2):
        """A restored model continues stepwise exactly as the original."""
        source = self._stepwise_model(config_hbv_2)
        n_steps = mock_dataset_hbv_2['x_phy'].shape[0]
        split = n_steps // 2

        with torch.no_grad():
            source(self._slice(mock_dataset_hbv_2, 0, split))
            snapshot = source.get_runtime_state()
            expected = [
                source(self._slice(mock_dataset_hbv_2, t, t + 1))['streamflow']
                for t in range(split, n_steps)
            ]

            target = self._stepwise_model(config_hbv_2, source.state_dict())
            target.load_runtime_state(snapshot)
            result = [
                target(self._slice(mock_dataset_hbv_2, t, t + 1))['streamflow']
                for t in range(split, n_steps)
            ]

        for r, e in zip(result, expected):
            assert torch.equal(r, e)

    def test_restore_fills_lstm_caches(self, config_hbv_2):
        """Restored LSTM states are what 'get_states' reports and the LSTM uses."""
        model = self._stepwise_model(config_hbv_2)
        hidden = model.nn_model.hiddeninv1
        state = {
            'nn': {'hn': torch.rand(1, 3, hidden), 'cn': torch.rand(1, 3, hidden)},
            'phy': {'states': None},
        }
        model.load_runtime_state(state)

        nn_model = model.nn_model
        for hn, cn in (
            (nn_model.lstminv.hn, nn_model.lstminv.cn),
            nn_model.get_states(),
            nn_model.lstminv.get_states(),
        ):
            assert torch.equal(hn, state['nn']['hn'])
            assert torch.equal(cn, state['nn']['cn'])

    def test_cold_roundtrip(self, config_hbv_2):
        """An unused model restores to empty states."""
        source = self._stepwise_model(config_hbv_2)
        target = self._stepwise_model(config_hbv_2)
        target.nn_model.lstminv.hn = torch.rand(1, 3, 4)
        target.load_runtime_state(source.get_runtime_state())
        assert target.nn_model.lstminv.hn is None
        assert target.phy_model.states is None

    def test_rejects_unknown_keys(self, config_hbv_2):
        """Snapshots with the wrong sections or keys are rejected."""
        model = self._stepwise_model(config_hbv_2)
        with pytest.raises(ValueError, match='do not match'):
            model.load_runtime_state({'nn': {}})

        state = model.get_runtime_state()
        state['nn'] = {**state['nn'], 'extra': None}
        with pytest.raises(ValueError, match='do not match'):
            model.load_runtime_state(state)

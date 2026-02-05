"""
Regression tests for deterministic model outputs and training pipeline.

These tests verify that dmg produces the same numerical results across code
changes. If any test here fails, it means a change has altered model behavior.

Coverage:
- Forward-pass output snapshot regression for Hbv, Hbv_1_1p, and Hbv_2
- NN parameter count and split regression
- Physics model parameter bounds regression
- Full training + evaluation pipeline regression

NOTE: Expected values were generated with seed=111111 and the
mock dataset from conftest.py. If conftest.py mock data generation changes,
these values must be updated.

NOTE: LSTM on CPU is non-deterministic across hardware (e.g. local machine vs.
GitHub Actions runner). Each numeric expected value is therefore a list of known
good values — one per known environment. If a test fails on a new environment,
add the reported value in the error to the appropriate list.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from dmg.core.utils import set_randomseed
from dmg.models.model_handler import ModelHandler
from dmg.trainers.trainer import Trainer
from tests import _get_nn_params, get_phy_model_name

# Force deterministic algorithms for LSTM reproducibility on CPU.
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)


def _assert_close_to_any(actual, expected_values, stat_name, model_name, rtol=1e-5):
    """Assert actual matches any value in expected_values within tolerance.

    LSTM CPU non-determinism means different hardware produces different (but
    individually reproducible) outputs.  Each entry in expected_values is a
    known value from one environment.
    """
    for exp in expected_values:
        if torch.isclose(actual, torch.tensor(exp), rtol=rtol, atol=1e-8):
            return
    exp_str = ", ".join(f"{v:.8e}" for v in expected_values)
    pytest.fail(
        f"{stat_name} regression failed for {model_name}: "
        f"got {actual.item():.8e}, expected one of [{exp_str}].\n"
        f"If this is a new environment, add the value to EXP_SNAPSHOTS."
    )


# ---------------------------------------------------------------------------
#  Expected learnable parameter counts.
# ---------------------------------------------------------------------------

EXP_PARAM_COUNTS = {
    'Hbv': {'total': 15},
    'Hbv_1_1p': {'total': 16},
    'Hbv_2': {'total': 18, 'dynamic': 3, 'static': 15},
}


# ---------------------------------------------------------------------------
#  Expected regression snapshot statistics (seed=111111).
# ---------------------------------------------------------------------------

EXP_SNAPSHOTS = {
    'Hbv': {
        'n_timesteps': 57,
        'streamflow_mean': [6.2234971672e-02, 5.1971767100e-02],
        'streamflow_sum': [3.5473934174e01, 2.9623907247e01],
        'streamflow_first': [1.0076804756e-05, 1.5578e-05],
        'streamflow_last': [1.8451465666e-01, 1.5288e-01],
        'AET_hydro_mean': [5.8963859081e-01, 5.84160924e-01],
        'recharge_mean': [1.5003199875e-01, 1.16926119e-01],
    },
    'Hbv_1_1p': {
        'n_timesteps': 57,
        'streamflow_mean': [2.1709175780e-02, 2.65194718e-02],
        'streamflow_sum': [1.2374230385e01, 1.51160989e01],
        'streamflow_first': [3.6438275856e-05, 3.46385932e-05],
        'streamflow_last': [6.6204883158e-02, 7.83670321e-02],
        'AET_hydro_mean': [4.6455222368e-01, 5.48316360e-01],
        'recharge_mean': [1.1395389587e-01, 1.3484e-01],
        'capillary_mean': [6.5453238785e-02, 7.44503066e-02],
    },
    'Hbv_2': {
        'n_timesteps': 59,
        'streamflow_mean': [0.0],
        'streamflow_sum': [0.0],
        'AET_hydro_mean': [5.4986560345e-01, 4.86893982e-01],
        'recharge_mean': [1.4136713743e-01, 9.89129990e-02],
        'capillary_mean': [8.9376135293e-06, 7.90938429e-06],
    },
}


# ---------------------------------------------------------------------------
#  Expected parameter bounds for each model.
# ---------------------------------------------------------------------------

EXP_PARAMETER_BOUNDS = {
    'Hbv': {
        'parBETA': [1.0, 6.0],
        'parFC': [50, 1000],
        'parK0': [0.05, 0.9],
        'parK1': [0.01, 0.5],
        'parK2': [0.001, 0.2],
        'parLP': [0.2, 1],
        'parPERC': [0, 10],
        'parUZL': [0, 100],
        'parTT': [-2.5, 2.5],
        'parCFMAX': [0.5, 10],
        'parCFR': [0, 0.1],
        'parCWH': [0, 0.2],
        'parBETAET': [0.3, 5],
    },
    'Hbv_1_1p': {
        'parBETA': [1.0, 6.0],
        'parFC': [50, 1000],
        'parK0': [0.05, 0.9],
        'parK1': [0.01, 0.5],
        'parK2': [0.001, 0.2],
        'parLP': [0.2, 1],
        'parPERC': [0, 10],
        'parUZL': [0, 100],
        'parTT': [-2.5, 2.5],
        'parCFMAX': [0.5, 10],
        'parCFR': [0, 0.1],
        'parCWH': [0, 0.2],
        'parBETAET': [0.3, 5],
        'parC': [0, 1],
    },
    'Hbv_2': {
        'parBETA': [1.0, 6.0],
        'parFC': [50, 1000],
        'parK0': [0.05, 0.9],
        'parK1': [0.01, 0.5],
        'parK2': [0.001, 0.2],
        'parLP': [0.2, 1],
        'parPERC': [0, 10],
        'parUZL': [0, 100],
        'parTT': [-2.5, 2.5],
        'parCFMAX': [0.5, 10],
        'parCFR': [0, 0.1],
        'parCWH': [0, 0.2],
        'parBETAET': [0.3, 5],
        'parC': [0, 1],
        'parRT': [0, 20],
        'parAC': [0, 2500],
    },
}

EXP_ROUTING_BOUNDS = {
    'route_a': [0, 2.9],
    'route_b': [0, 6.5],
}


# ---------------------------------------------------------------------------
#  Expected training pipeline regression values.
# ---------------------------------------------------------------------------

EXP_FINAL_LOSS_VALUES = [
    24.07529079914093,  # Local machine loss
    26.94981688261032,  # GHA runner loss (current)
]
EXP_NSE_VALUES = [
    -33.58369255065918,  # Local machine NSE
    -33.43178367614746,  # GHA runner NSE
]


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------


class TestOutputSnapshotRegression:
    """Deterministic forward-pass snapshot regression tests.

    These tests catch any code change that alters model output values.
    Each stat is checked against a list of known-good values from different
    environments (see EXP_SNAPSHOTS).
    """

    def _run_model(self, model_config, model_dataset):
        """Run the model and return (model_name, output dict)."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        handler = ModelHandler(model_config)
        dpl_model = handler.model_dict[model_name]
        dpl_model.eval()

        with torch.no_grad():
            output = dpl_model(model_dataset)

        return model_name, output

    def test_streamflow_statistics(self, model_config, model_dataset):
        """Verify streamflow statistics match expected snapshot values."""
        model_name, output = self._run_model(model_config, model_dataset)
        expected = EXP_SNAPSHOTS[model_name]
        sf = output['streamflow']

        assert sf.shape[0] == expected['n_timesteps'], (
            f"Timestep count changed: expected {expected['n_timesteps']}, "
            f"got {sf.shape[0]}"
        )

        _assert_close_to_any(
            sf.mean(),
            expected['streamflow_mean'],
            'streamflow_mean',
            model_name,
        )
        _assert_close_to_any(
            sf.sum(),
            expected['streamflow_sum'],
            'streamflow_sum',
            model_name,
        )

    def test_streamflow_pointwise_regression(self, model_config, model_dataset):
        """Verify first and last streamflow values match expected snapshots."""
        model_name, output = self._run_model(model_config, model_dataset)
        expected = EXP_SNAPSHOTS[model_name]
        sf = output['streamflow']

        if 'streamflow_first' in expected:
            _assert_close_to_any(
                sf[0, 0],
                expected['streamflow_first'],
                'streamflow_first',
                model_name,
            )

        if 'streamflow_last' in expected:
            _assert_close_to_any(
                sf[-1, -1],
                expected['streamflow_last'],
                'streamflow_last',
                model_name,
            )

    def test_evapotranspiration_regression(self, model_config, model_dataset):
        """Verify AET statistics match expected snapshot values."""
        model_name, output = self._run_model(model_config, model_dataset)
        expected = EXP_SNAPSHOTS[model_name]

        _assert_close_to_any(
            output['AET_hydro'].mean(),
            expected['AET_hydro_mean'],
            'AET_hydro_mean',
            model_name,
        )

    def test_recharge_regression(self, model_config, model_dataset):
        """Verify recharge statistics match expected snapshot values."""
        model_name, output = self._run_model(model_config, model_dataset)
        expected = EXP_SNAPSHOTS[model_name]

        _assert_close_to_any(
            output['recharge'].mean(),
            expected['recharge_mean'],
            'recharge_mean',
            model_name,
        )

    def test_capillary_regression(self, model_config, model_dataset):
        """Verify capillary rise stats for models that support it."""
        model_name = get_phy_model_name(model_config)
        expected = EXP_SNAPSHOTS[model_name]

        if 'capillary_mean' not in expected:
            pytest.skip(f"{model_name} does not have capillary output")

        _, output = self._run_model(model_config, model_dataset)

        _assert_close_to_any(
            output['capillary'].mean(),
            expected['capillary_mean'],
            'capillary_mean',
            model_name,
        )


# ---------------------------------------------------------------------------
#  Parameter count and bounds regression tests
# ---------------------------------------------------------------------------


class TestParameterRegression:
    """Regression tests for NN parameter counts and physics model bounds."""

    def test_nn_parameter_count(self, model_config, model_dataset):
        """NN output dimension must match expected regression count."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)
        expected = EXP_PARAM_COUNTS[model_name]

        handler = ModelHandler(model_config)
        dpl_model = handler.model_dict[model_name]
        params = _get_nn_params(dpl_model, model_dataset)

        if isinstance(params, tuple):
            actual_count = sum(p.shape[-1] for p in params)
        else:
            actual_count = params.shape[-1]

        assert actual_count == expected['total'], (
            f"NN parameter count mismatch for {model_name}: "
            f"expected {expected['total']}, got {actual_count}"
        )

    def test_nn_parameter_count_split(self, model_config, model_dataset):
        """For LstmMlpModel (Hbv_2), verify dynamic/static parameter split."""
        model_name = get_phy_model_name(model_config)
        expected = EXP_PARAM_COUNTS[model_name]

        if 'dynamic' not in expected:
            pytest.skip(f"{model_name} does not use split parameters")

        set_randomseed(model_config['seed'])

        handler = ModelHandler(model_config)
        dpl_model = handler.model_dict[model_name]
        params = _get_nn_params(dpl_model, model_dataset)

        assert isinstance(params, tuple), (
            f"{model_name} should return tuple of (dynamic, static) params"
        )
        assert params[0].shape[-1] == expected['dynamic'], (
            f"Dynamic param count mismatch: "
            f"expected {expected['dynamic']}, got {params[0].shape[-1]}"
        )
        assert params[1].shape[-1] == expected['static'], (
            f"Static param count mismatch: "
            f"expected {expected['static']}, got {params[1].shape[-1]}"
        )

    def test_physics_model_parameter_bounds(self, model_config, model_dataset):
        """Physics model parameter bounds must match expected values."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        handler = ModelHandler(model_config)
        phy_model = handler.model_dict[model_name].phy_model

        expected_bounds = EXP_PARAMETER_BOUNDS[model_name]
        actual_bounds = phy_model.parameter_bounds

        for param_name, expected_range in expected_bounds.items():
            assert param_name in actual_bounds, (
                f"Parameter '{param_name}' missing from {model_name}.parameter_bounds"
            )
            assert actual_bounds[param_name] == expected_range, (
                f"Bounds changed for {model_name}.{param_name}: "
                f"expected {expected_range}, got {actual_bounds[param_name]}"
            )

        # Check no unexpected parameters added.
        expected_names = set(expected_bounds.keys())
        actual_names = set(actual_bounds.keys())
        assert actual_names == expected_names, (
            f"Parameter names changed for {model_name}.\n"
            f"  Missing: {expected_names - actual_names}\n"
            f"  Extra: {actual_names - expected_names}"
        )

    def test_routing_parameter_bounds(self, model_config, model_dataset):
        """Routing parameter bounds must match expected values."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        handler = ModelHandler(model_config)
        phy_model = handler.model_dict[model_name].phy_model

        actual_routing = phy_model.routing_parameter_bounds

        for param_name, expected_range in EXP_ROUTING_BOUNDS.items():
            assert param_name in actual_routing, (
                f"Routing parameter '{param_name}' missing from {model_name}"
            )
            assert actual_routing[param_name] == expected_range, (
                f"Routing bounds changed for {model_name}.{param_name}: "
                f"expected {expected_range}, got {actual_routing[param_name]}"
            )


# ---------------------------------------------------------------------------
#  Training pipeline regression test
# ---------------------------------------------------------------------------


def test_training_regression(config, mock_dataset, tmp_path):
    """Test the full training and evaluation pipeline for reproducibility."""
    set_randomseed(config['seed'])

    model = ModelHandler(config)
    trainer = Trainer(
        config,
        model,
        train_dataset=mock_dataset,
        eval_dataset=mock_dataset,
        write_out=True,
    )

    # --- Training ---
    trainer.train()

    is_loss_close = np.isclose(
        trainer.total_loss,
        EXP_FINAL_LOSS_VALUES,
        rtol=1e-4,
        atol=1e-5,
    )
    assert np.any(is_loss_close), (
        f"Training loss regression failed. "
        f"Got: {trainer.total_loss}, "
        f"Expected one of: {EXP_FINAL_LOSS_VALUES}"
    )

    # --- Evaluation ---
    config['mode'] = 'test'
    config['test']['test_epoch'] = 2
    model_eval = ModelHandler(config)

    trainer_eval = Trainer(config, model_eval, eval_dataset=mock_dataset)
    trainer_eval.evaluate()

    metrics_path = Path(config['output_dir']) / 'metrics_agg.json'
    with open(metrics_path) as f:
        metrics = json.load(f)

    actual_nse = metrics['nse']['median']
    is_nse_close = np.isclose(actual_nse, EXP_NSE_VALUES, atol=1e-3)
    assert np.any(is_nse_close), (
        f"Evaluation NSE regression failed. "
        f"Got: {actual_nse}, "
        f"Expected one of: {EXP_NSE_VALUES}"
    )

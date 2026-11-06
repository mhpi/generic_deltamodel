"""Test data loaders in dmg/core/data/loaders/."""

import pickle

import numpy as np
import pytest
import torch

from dmg.core.data.loaders.hydro_loader import HydroLoader
from dmg.core.utils.paths import clear_norm_stats

# ---------------------------------------------------------------------------- #
#  Helpers
# ---------------------------------------------------------------------------- #


def _low_flow_loader(config, mock_dataset, tmp_path):
    """Build a HydroLoader for the low-flow regime on mock data."""
    data_path = tmp_path / 'test_data.pkl'
    with open(data_path, 'wb') as f:
        forcings = mock_dataset['x_phy'].permute(1, 0, 2).numpy()
        target = mock_dataset['target'].permute(1, 0, 2).numpy()
        attributes = mock_dataset['c_nn'].numpy()
        pickle.dump((forcings, target, attributes), f)

    config['observations']['data_path'] = str(data_path)
    config['model']['flow_regime'] = 'low'

    # Isolate from norm stats cached by other tests. A pretrained_model_dir takes
    # priority over model_dir in the loader.
    config['model_dir'] = str(tmp_path)
    config['pretrained_model_dir'] = None
    return HydroLoader(config, test_split=True)


# ---------------------------------------------------------------------------- #
#  Tests
# ---------------------------------------------------------------------------- #


def test_hydro_loader_init(config, mock_dataset, tmp_path):
    """Test the initialization of the HydroLoader."""
    data_path = tmp_path / "test_data.pkl"
    with open(data_path, 'wb') as f:
        # The loader expects a tuple of (forcings, target, attributes)
        # with shapes (basins, time, features)
        forcings = mock_dataset['x_phy'].permute(1, 0, 2).numpy()
        target = mock_dataset['target'].permute(1, 0, 2).numpy()
        attributes = mock_dataset['c_nn'].numpy()
        pickle.dump((forcings, target, attributes), f)

    config['observations']['data_path'] = str(data_path)

    # Test with test_split = True
    loader_split = HydroLoader(config, test_split=True)
    assert loader_split.train_dataset is not None
    assert loader_split.eval_dataset is not None
    with pytest.raises(AttributeError):
        _ = loader_split.dataset
    assert 'xc_nn_norm' in loader_split.train_dataset
    assert 'xc_nn_norm' in loader_split.eval_dataset

    # Test with test_split = False
    config['mode'] = 'simulation'  # Change mode to avoid splitting
    loader_no_split = HydroLoader(config, test_split=False)
    with pytest.raises(AttributeError):
        _ = loader_no_split.train_dataset
    with pytest.raises(AttributeError):
        _ = loader_no_split.eval_dataset
    assert loader_no_split.dataset is not None
    assert 'xc_nn_norm' in loader_no_split.dataset


def test_low_flow_target_is_standardized(config, mock_dataset, tmp_path):
    """Log-normalized targets use log-space stats, so they standardize."""
    loader = _low_flow_loader(config, mock_dataset, tmp_path)
    target = loader.train_dataset['target']
    assert target.mean().item() == pytest.approx(0.0, abs=1e-4)
    assert target.std().item() == pytest.approx(1.0, abs=0.02)


def test_from_norm_inverts_log_norm_target(config, mock_dataset, tmp_path):
    """De-normalizing a log-normalized target recovers the original values."""
    loader = _low_flow_loader(config, mock_dataset, tmp_path)
    values = np.array([[[0.0], [0.5], [4.0]]], dtype=np.float32)
    normed = loader.to_norm(values, ['streamflow'])
    restored = loader.from_norm(normed, ['streamflow'])
    assert np.allclose(restored, values, atol=1e-4)


def test_from_norm_clamps_below_log_floor(config, mock_dataset, tmp_path):
    """Values below the log-sqrt floor invert to zero, not to positive flow."""
    loader = _low_flow_loader(config, mock_dataset, tmp_path)
    mean, std = loader.norm_stats['streamflow'][2:4]
    below_floor = (np.log10(0.1) - 1.0 - mean) / std
    data = np.full((1, 1, 1), below_floor, dtype=np.float32)
    assert loader.from_norm(data, ['streamflow']).item() == 0.0


@pytest.mark.parametrize(
    ('overwrite', 'mode', 'pretrained', 'removed'),
    [
        (True, 'train', None, True),
        (True, 'train_test', None, True),
        (True, 'test', None, False),
        (True, 'sim', None, False),
        (False, 'train_test', None, False),
        (True, 'train_test', 'pretrained/', False),
    ],
)
def test_clear_norm_stats(overwrite, mode, pretrained, removed, tmp_path):
    """Cached stats are removed only for a training run that overwrites."""
    path = tmp_path / 'normalization_statistics.json'
    path.write_text('{}')
    config = {
        'overwrite': overwrite,
        'mode': mode,
        'pretrained_model_dir': pretrained,
        'model_dir': str(tmp_path),
    }
    clear_norm_stats(config)
    assert path.exists() != removed


def test_clear_norm_stats_without_cached_file(tmp_path):
    """A first run with overwrite has nothing to remove and does not fail."""
    config = {'overwrite': True, 'mode': 'train', 'model_dir': str(tmp_path)}
    clear_norm_stats(config)


def test_to_norm_rounds_once(config, mock_dataset, tmp_path):
    """Normalization runs in double precision and rounds only the result."""
    loader = _low_flow_loader(config, mock_dataset, tmp_path)
    values = np.random.default_rng(0).uniform(0.0, 50.0, (100, 1))
    mean, std = loader.norm_stats['streamflow'][2:4]
    expected = (np.log10(np.sqrt(values) + 0.1) - mean) / std

    normed = loader.to_norm(values, ['streamflow'])
    assert normed.dtype == np.float32
    assert np.array_equal(normed, expected.astype(np.float32))

    normed = loader.to_norm(values, ['streamflow'], dtype=np.float64)
    assert np.array_equal(normed, expected)


def test_train_dataset_keeps_double_precision_target(config, mock_dataset, tmp_path):
    """Only the training data keeps an unrounded target, for loss statistics."""
    loader = _low_flow_loader(config, mock_dataset, tmp_path)
    target = loader.train_dataset['target_f64']
    assert target.dtype == torch.float64
    assert torch.equal(target.float(), loader.train_dataset['target'])
    assert 'target_f64' not in loader.eval_dataset

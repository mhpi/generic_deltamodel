"""Test data loaders in dmg/core/data/loaders/."""

import sys
from pathlib import Path
import pickle

sys.path.append(str(Path(__file__).parent.parent))
from dmg.core.data.loaders.hydro_loader import HydroLoader


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
    assert loader_split.dataset is None
    assert 'xc_nn_norm' in loader_split.train_dataset
    assert 'xc_nn_norm' in loader_split.eval_dataset

    # Test with test_split = False
    config['mode'] = 'simulation'  # Change mode to avoid splitting
    loader_no_split = HydroLoader(config, test_split=False)
    assert loader_no_split.train_dataset is None
    assert loader_no_split.eval_dataset is None
    assert loader_no_split.dataset is not None
    assert 'xc_nn_norm' in loader_no_split.dataset

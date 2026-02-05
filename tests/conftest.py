"""
Fixtures and utilities for testing differentiable models in dmg.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).parent.parent))

from dmg.core.utils import initialize_config
from tests import get_phy_model_name


def build_config(config_dict):
    """Helper function to build a mock configuration dictionary."""
    # Convert to OmegaConf dict to run Pydantic field validation.
    config_tmp = OmegaConf.create(config_dict)
    config = initialize_config(config_tmp, write_out=False)

    # Use temporary directory for outputs
    config['output_dir'] = os.path.join(os.getcwd(), config['output_dir'])
    config['model_dir'] = os.path.join(os.getcwd(), config['model_dir'])
    config['plot_dir'] = os.path.join(os.getcwd(), config['plot_dir'])
    config['sim_dir'] = os.path.join(os.getcwd(), config['sim_dir'])
    config['log_dir'] = os.path.join(os.getcwd(), config['log_dir'])

    # Create output directories
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(config['sim_dir'], exist_ok=True)

    return config


# ---------------------------------------------------------------------------
#  Shared config sections.
# ---------------------------------------------------------------------------

_SHARED_CONFIG = {
    'mode': 'train_test',
    'multimodel_type': 'none',
    'seed': 111111,
    'logging': 'tensorboard',
    'cache_states': False,
    'device': 'cpu',
    'gpu_id': 0,
    'data_loader': 'HydroLoader',
    'data_sampler': 'HydroSampler',
    'trainer': 'Trainer',
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
    'train_time': ['2000/01/01', '2000/01/31'],
    'test_time': ['2000/02/01', '2000/02/10'],
    'sim_time': ['2000/02/01', '2000/02/10'],
    'all_time': ['2000/01/01', '2000/02/10'],
}


def _hbv_config_dict():
    """Config dict for dHBV 1.0 with 2 dynamic parameters."""
    return {
        **_SHARED_CONFIG,
        'model': {
            'rho': 10,
            'warm_up': 2,
            'use_log_norm': ['prcp'],
            'phy': {
                'name': ['Hbv'],
                'nmul': 1,
                'warm_up_states': True,
                'dy_drop': 0.0,
                'dynamic_params': {'Hbv': ['parBETA', 'parBETAET']},
                'routing': True,
                'nearzero': 1e-5,
                'forcings': ['prcp', 'tmean', 'pet'],
                'attributes': [],
                'cache_states': False,
            },
            'nn': {
                'name': 'LstmModel',
                'dropout': 0.5,
                'hidden_size': 32,
                'forcings': ['prcp', 'tmean', 'pet'],
                'attributes': ['area_gages2'],
                'cache_states': False,
            },
        },
    }


def _hbv_1_1p_config_dict():
    """Config dict for dHBV 1.1p with 3 dynamic parameters."""
    return {
        **_SHARED_CONFIG,
        'model': {
            'rho': 10,
            'warm_up': 2,
            'use_log_norm': ['prcp'],
            'phy': {
                'name': ['Hbv_1_1p'],
                'nmul': 1,
                'warm_up_states': False,
                'dy_drop': 0.0,
                'dynamic_params': {
                    'Hbv_1_1p': ['parBETA', 'parK0', 'parBETAET'],
                },
                'routing': True,
                'nearzero': 1e-5,
                'forcings': ['prcp', 'tmean', 'pet'],
                'attributes': [],
                'cache_states': False,
            },
            'nn': {
                'name': 'LstmModel',
                'dropout': 0.5,
                'hidden_size': 32,
                'forcings': ['prcp', 'tmean', 'pet'],
                'attributes': ['area_gages2'],
                'cache_states': False,
            },
        },
    }


def _hbv_2_config_dict():
    """Config dict for dHBV 2.0 with LstmMlpModel NN."""
    return {
        **_SHARED_CONFIG,
        'model': {
            'rho': 10,
            'warm_up': 2,
            'use_log_norm': ['prcp'],
            'phy': {
                'name': ['Hbv_2'],
                'nmul': 1,
                'warm_up_states': True,
                'dy_drop': 0.0,
                'dynamic_params': {
                    'Hbv_2': ['parBETA', 'parK0', 'parBETAET'],
                },
                'routing': True,
                'nearzero': 1e-5,
                'forcings': ['prcp', 'tmean', 'pet'],
                'attributes': [],
                'cache_states': False,
            },
            'nn': {
                'name': 'LstmMlpModel',
                'lstm_hidden_size': 32,
                'mlp_hidden_size': 64,
                'lstm_dropout': 0.5,
                'mlp_dropout': 0.5,
                'forcings': ['prcp', 'tmean', 'pet'],
                'attributes': ['area_gages2'],
                'cache_states': False,
            },
        },
    }


def build_mock_dataset(config, n_basins=10):
    """Build a mock dataset matching the given config."""
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    start_date = config['observations']['start_time'].replace('/', '-')
    end_date = config['observations']['end_time'].replace('/', '-')
    n_timesteps = (np.datetime64(end_date) - np.datetime64(start_date)).astype(
        'timedelta64[D]',
    ).astype(int) + 1

    n_phy_forcings = len(config['model']['phy']['forcings'])
    n_phy_attrs = len(config['model']['phy']['attributes'])
    n_nn_forcings = len(config['model']['nn']['forcings'])
    n_nn_attrs = len(config['model']['nn']['attributes'])

    # Scale physics forcings to realistic ranges:
    # prcp ~ [0, 10] mm/day, tmean ~ [-5, 25] °C, pet ~ [0, 5] mm/day.
    x_phy = torch.rand(n_timesteps, n_basins, n_phy_forcings)
    if n_phy_forcings >= 1:
        x_phy[:, :, 0] *= 10.0  # prcp
    if n_phy_forcings >= 2:
        x_phy[:, :, 1] = x_phy[:, :, 1] * 30.0 - 5.0  # tmean
    if n_phy_forcings >= 3:
        x_phy[:, :, 2] *= 5.0  # pet

    dataset = {
        'x_phy': x_phy,
        'c_phy': torch.rand(n_basins, max(n_phy_attrs, 1)),
        'x_nn': torch.rand(n_timesteps, n_basins, n_nn_forcings),
        'c_nn': torch.rand(n_basins, n_nn_attrs),
        'xc_nn_norm': torch.rand(
            n_timesteps,
            n_basins,
            n_nn_forcings + n_nn_attrs,
        ),
        'target': torch.rand(
            n_timesteps,
            n_basins,
            len(config['train']['target']),
        ),
    }

    # LstmMlpModel requires c_nn_norm for its MLP head.
    if config['model']['nn']['name'] == 'LstmMlpModel':
        dataset['c_nn_norm'] = torch.rand(n_basins, n_nn_attrs)

    # HBV_2 physics model requires catchment area and elevation.
    phy_name = get_phy_model_name(config)
    if phy_name == 'Hbv_2':
        dataset['ac_all'] = torch.rand(n_basins) * 1000 + 10
        dataset['elev_all'] = torch.rand(n_basins) * 3000

    return dataset


# ---------------------------------------------------------------------------
#  Named fixtures for individual model configs.
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    """A fixture for a mock configuration dictionary.

    Looks like config for dHBV1.0 with 2 dynamic parameters.
    """
    return build_config(_hbv_config_dict())


@pytest.fixture
def mock_dataset(config):
    """A fixture for a mock dataset dictionary, consistent with the config."""
    return build_mock_dataset(config)


@pytest.fixture
def config_hbv_1_1p():
    """Config fixture for dHBV 1.1p with 3 dynamic parameters."""
    return build_config(_hbv_1_1p_config_dict())


@pytest.fixture
def mock_dataset_hbv_1_1p(config_hbv_1_1p):
    """Mock dataset fixture for dHBV 1.1p."""
    return build_mock_dataset(config_hbv_1_1p)


@pytest.fixture
def config_hbv_2():
    """Config fixture for dHBV 2.0 with LstmMlpModel NN."""
    return build_config(_hbv_2_config_dict())


@pytest.fixture
def mock_dataset_hbv_2(config_hbv_2):
    """Mock dataset fixture for dHBV 2.0."""
    return build_mock_dataset(config_hbv_2)


# ---------------------------------------------------------------------------
#  Parametrized fixtures: run the same test across all three models.
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=['hbv', 'hbv_1_1p', 'hbv_2'],
    ids=['Hbv', 'Hbv_1_1p', 'Hbv_2'],
)
def model_config(request):
    """Parametrized config fixture for all HBV model variants."""
    builders = {
        'hbv': _hbv_config_dict,
        'hbv_1_1p': _hbv_1_1p_config_dict,
        'hbv_2': _hbv_2_config_dict,
    }
    return build_config(builders[request.param]())


@pytest.fixture
def model_dataset(model_config):
    """Mock dataset matching the parametrized model_config."""
    return build_mock_dataset(model_config)

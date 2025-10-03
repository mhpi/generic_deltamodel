import pytest
import torch
import numpy as np


@pytest.fixture
def config():
    """A fixture for a mock configuration dictionary."""
    return {
        'mode': 'train_test',
        'multimodel_type': 'none',
        'seed': 111111,
        'device': 'cpu',
        'gpu_id': 0,
        'dtype': 'torch.float32',
        'data_loader': 'HydroLoader',
        'data_sampler': 'HydroSampler',
        'trainer': 'Trainer',
        'train': {
            'start_time': '2000/01/01',
            'end_time': '2000/01/30',
            'target': ['streamflow'],
            'optimizer': 'Adadelta',
            'batch_size': 5,
            'epochs': 2,
            'start_epoch': 0,
            'save_epoch': 1,
            'loss_function': {
                'name': 'RmseLossComb',
            },
            'lr': 1.0,
            'lr_scheduler': None,
        },
        'test': {
            'start_time': '2000/02/01',
            'end_time': '2000/02/10',
            'batch_size': 5,
            'test_epoch': 2,
        },
        'sim': {
            'start_time': '2000/01/01',
            'end_time': '2000/01/10',
            'batch_size': 5,
        },
        'model': {
            'rho': 10,
            'phy': {
                'name': ['Hbv'],
                'nmul': 1,
                'warm_up': 0,
                'warm_up_states': True,
                'dy_drop': 0.0,
                'dynamic_params': {
                    'Hbv': ['parBETA', 'parBETAET'],
                },
                'routing': True,
                'use_log_norm': ['prcp'],
                'forcings': ['prcp', 'tmean', 'pet'],
                'attributes': [],
            },
            'nn': {
                'name': 'LstmModel',
                'dropout': 0.5,
                'hidden_size': 32,
                'forcings': ['prcp', 'tmean', 'pet'],
                'attributes': ['p_mean'],
            },
        },
        'observations': {
            'name': 'camels_531',
            'all_forcings': ['prcp', 'tmean', 'pet'],
            'all_attributes': ['p_mean'],
            'area_name': 'p_mean',  # Using p_mean for area for simplicity
            'start_time': '2000/01/01',
            'end_time': '2000/02/28',
        },
        'output_dir': './tests/test_output/',
        'model_dir': './tests/test_output/model',
        'train_time': ['2000/01/01', '2000/01/30'],
        'eval_time': ['2000/02/01', '2000/02/10'],
        'sim_time': ['2000/02/01', '2000/02/10'],
        'all_time': ['2000/01/01', '2000/02/10'],
    }


@pytest.fixture
def mock_dataset(config):
    """A fixture for a mock dataset dictionary, consistent with the config."""
    n_basins = 10
    # Replace slashes with hyphens for numpy compatibility
    start_date = config['observations']['start_time'].replace('/', '-')
    end_date = config['observations']['end_time'].replace('/', '-')

    n_timesteps = (np.datetime64(end_date) - np.datetime64(start_date)).astype(
        'timedelta64[D]'
    ).astype(int) + 1

    return {
        'x_phy': torch.rand(
            n_timesteps, n_basins, len(config['model']['phy']['forcings'])
        ),
        'c_phy': torch.rand(n_basins, len(config['model']['phy']['attributes'])),
        'x_nn': torch.rand(
            n_timesteps, n_basins, len(config['model']['nn']['forcings'])
        ),
        'c_nn': torch.rand(n_basins, len(config['model']['nn']['attributes'])),
        'xc_nn_norm': torch.rand(
            n_timesteps,
            n_basins,
            len(config['model']['nn']['forcings'])
            + len(config['model']['nn']['attributes']),
        ),
        'target': torch.rand(n_timesteps, n_basins, len(config['train']['target'])),
    }

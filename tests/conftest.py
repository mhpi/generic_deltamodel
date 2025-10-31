import pytest
import torch
import numpy as np


@pytest.fixture
def config():
    """A fixture for a mock configuration dictionary.

    Looks like config for dHBV1.0 with 2 dynamic parameters.
    """
    return {
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
            'optimizer': {
                'name': 'Adadelta',
            },
            'lr': 1.0,
            'lr_scheduler': {
                'name': 'StepLR',
                'step_size': 10,
                'gamma': 0.5,
            },
            'loss_function': {
                'name': 'RmseCombLoss',
            },
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
        'model': {
            'rho': 10,
            'warm_up': 2,
            'use_log_norm': ['prcp'],
            'phy': {
                'name': ['Hbv'],
                'nmul': 1,
                'warm_up_states': True,
                'dy_drop': 0.0,
                'dynamic_params': {
                    'Hbv': ['parBETA', 'parBETAET'],
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


@pytest.fixture
def mock_dataset(config):
    """A fixture for a mock dataset dictionary, consistent with the config."""
    # Set random seed
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

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

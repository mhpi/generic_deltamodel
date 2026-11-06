"""
Tests that the benchmark configs follow the published source code setups.

Coverage:
- Spin-up ends exactly at the start of the source evaluation window
- dHBV models warm up states without gradients, as the source code does

NOTE: Physics parity with the source code is tested in hydrodl2
(tests/test_source_parity.py).
"""

from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from omegaconf import OmegaConf

CONF_DIR = Path(__file__).parent.parent / 'example' / 'conf'
EVAL_START = date(1989, 10, 1)


@pytest.mark.parametrize('section', ['test', 'sim'])
@pytest.mark.parametrize(
    'name',
    ['config_dhbv.yaml', 'config_dhbv_1_1p.yaml', 'config_lstm.yaml'],
)
def test_spin_up_precedes_eval_window(name, section):
    """Warmup days are taken before the source evaluation start."""
    config = OmegaConf.load(CONF_DIR / name)
    start = datetime.strptime(config[section]['start_time'], '%Y/%m/%d').date()
    assert start + timedelta(days=config['model']['warmup']) == EVAL_START


@pytest.mark.parametrize('name', ['config_dhbv.yaml', 'config_dhbv_1_1p.yaml'])
def test_dhbv_warms_up_states_without_gradients(name):
    """The source code spins up states under no_grad, before the main run."""
    config = OmegaConf.load(CONF_DIR / name)
    assert config['model']['phy']['warmup_states'] is True

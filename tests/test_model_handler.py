"""
Tests for ModelHandler functionality.

Coverage:
- Initialization
- Forward pass in train and eval modes
- Train/eval mode switching
- Parameter retrieval
- ModelHandler vs DplModel output equivalence
- Model serialization (save/load)
- Device placement

NOTE: All tests are parametrized over Hbv, Hbv_1_1p, and Hbv_2.
"""

import torch

from dmg.core.utils import set_randomseed
from dmg.models.model_handler import ModelHandler
from tests import compute_mse_loss, get_phy_model_name


class TestModelHandler:
    """Test ModelHandler functionality."""

    def test_model_handler_initialization(self, model_config):
        """Verify ModelHandler initializes correctly."""
        model_name = get_phy_model_name(model_config)
        model = ModelHandler(model_config)

        assert model_name in model.model_dict
        assert model.model_type == 'dm'
        assert model.device == 'cpu'

    def test_model_handler_forward_train_mode(
        self,
        model_config,
        model_dataset,
    ):
        """Test ModelHandler forward in train mode."""
        model_name = get_phy_model_name(model_config)
        model = ModelHandler(model_config)

        model.train()
        output_dict = model(model_dataset)

        assert model_name in output_dict
        assert isinstance(output_dict[model_name], dict)
        assert 'streamflow' in output_dict[model_name]

    def test_model_handler_forward_eval_mode(
        self,
        model_config,
        model_dataset,
    ):
        """Test ModelHandler forward in eval mode."""
        model_name = get_phy_model_name(model_config)
        model = ModelHandler(model_config)

        output_dict = model(model_dataset, eval=True)

        assert model_name in output_dict
        assert isinstance(output_dict[model_name], dict)
        assert 'streamflow' in output_dict[model_name]

    def test_model_handler_get_parameters(self, model_config):
        """Test ModelHandler parameter retrieval."""
        model = ModelHandler(model_config)
        params = list(model.get_parameters())

        assert len(params) > 0, "No parameters found"
        for param in params:
            assert isinstance(param, torch.Tensor)
            assert param.requires_grad

    def test_model_handler_train_eval_mode_switching(
        self,
        model_config,
        model_dataset,
    ):
        """Test switching between train and eval modes.

        NOTE: ModelHandler.model_dict is a plain dict, not a ModuleDict, so
        Module.eval()/train() does NOT propagate to child DplModels. The
        ModelHandler.forward() method manually sets train/eval mode on each
        model. This test verifies that forward() correctly sets the mode.
        """
        handler = ModelHandler(model_config)

        # Train mode via forward.
        handler(model_dataset, eval=False)
        for model in handler.model_dict.values():
            assert model.training

        # Eval mode via forward.
        handler(model_dataset, eval=True)
        for model in handler.model_dict.values():
            assert not model.training

    def test_model_handler_output_matches_dpl_model(
        self,
        model_config,
        model_dataset,
    ):
        """Verify ModelHandler wrapping doesn't alter DplModel outputs."""
        set_randomseed(model_config['seed'])
        model_name = get_phy_model_name(model_config)

        handler = ModelHandler(model_config)
        dpl_model = handler.model_dict[model_name]

        # Direct DplModel forward.
        dpl_model.eval()
        with torch.no_grad():
            direct_output = dpl_model(model_dataset)

        # Re-seed and recreate for ModelHandler forward.
        set_randomseed(model_config['seed'])
        handler2 = ModelHandler(model_config)
        handler2.model_dict[model_name].eval()
        with torch.no_grad():
            handler_output = handler2(model_dataset, eval=True)

        handler_model_output = handler_output[model_name]

        for key in direct_output:
            assert torch.allclose(
                direct_output[key],
                handler_model_output[key],
                rtol=1e-5,
            ), f"ModelHandler output differs from DplModel for key '{key}'"


def test_model_serialization(model_config, model_dataset, tmp_path):
    """Test saving and loading DplModel states."""
    set_randomseed(model_config['seed'])
    model_name = get_phy_model_name(model_config)
    warm_up = model_config['model']['warm_up']

    model1 = ModelHandler(model_config)
    dpl1 = model1.model_dict[model_name]
    optimizer = torch.optim.Adam(dpl1.parameters(), lr=0.01)

    dpl1.train()
    for _ in range(3):
        optimizer.zero_grad()
        loss, _ = compute_mse_loss(dpl1, model_dataset, warm_up)
        loss.backward()
        optimizer.step()

    save_path = tmp_path / "test_model.pt"
    torch.save(dpl1.state_dict(), save_path)

    set_randomseed(model_config['seed'])
    model2 = ModelHandler(model_config)
    dpl2 = model2.model_dict[model_name]
    dpl2.load_state_dict(torch.load(save_path))

    dpl1.eval()
    dpl2.eval()

    with torch.no_grad():
        output1 = dpl1(model_dataset)['streamflow']
        output2 = dpl2(model_dataset)['streamflow']

    assert torch.allclose(output1, output2, rtol=1e-4, atol=1e-7), (
        "Loaded model produces different output"
    )


def test_device_placement(model_config, model_dataset):
    """Test model can be moved to different devices (CPU)."""
    model_name = get_phy_model_name(model_config)
    model = ModelHandler(model_config, device='cpu')

    assert model.device == 'cpu'

    for param in model.get_parameters():
        assert param.device.type == 'cpu'

    output = model(model_dataset)
    assert output[model_name]['streamflow'].device.type == 'cpu'

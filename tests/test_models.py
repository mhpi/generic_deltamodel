# """Test models in dmg/models/."""

# import sys
# from pathlib import Path

# sys.path.append(str(Path(__file__).parent.parent))

# import torch

# from dmg.models.delta_models.dpl_model import DplModel
# from dmg.models.model_handler import ModelHandler
# from hydrodl2.models.hbv import Hbv
# from dmg.core.utils import load_nn_model

# def test_dpl_model(config, mock_dataset):
#     """Test the DplModel forward pass."""
#     phy_model = Hbv(config['model']['phy'])
#     nn_model = load_nn_model(phy_model, config['model'])
#     model = DplModel(phy_model=phy_model, nn_model=nn_model)

#     output = model(mock_dataset)
#     assert isinstance(output, dict)
#     assert 'streamflow' in output


# def test_model_handler(config, mock_dataset):
#     """Test the ModelHandler."""
#     # Temporarily set mode to train to allow initialization
#     config['mode'] = 'train'
#     handler = ModelHandler(config)
#     assert 'Hbv' in handler.model_dict

#     # Test forward pass in train mode
#     output = handler(mock_dataset)
#     assert 'Hbv' in output
#     assert isinstance(output['Hbv'], dict)

#     # Test forward pass in eval mode
#     output_eval = handler(mock_dataset, eval=True)
#     assert 'Hbv' in output_eval

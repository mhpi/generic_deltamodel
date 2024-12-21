from typing import Any, Dict

import torch
from models.model_handler import ModelHandler as dModel


def convert_and_load_model_state(
    state_path: str,
    save_path: str,
    config: Dict[str, Any],
    model_architecture: str = 'HBV'
) -> None:
    """
    Convert saved model state dict to a new format for a different model architecture,
    load the states into the model, and save the updated state dict and model.

    Parameters
    ----------
    state_path : str
        Path to the saved state dictionary (.pt file).
    save_path : str
        Path to save the converted state dictionary and model.
    config : dict
        Configuration dictionary for the model.
    model_architecture : str, optional
        Name of the new model architecture, by default 'HBV'.
    """
    # Load the original state dict
    state_dict = torch.load(state_path)

    # Convert the state dictionary to the new format
    new_state_dict = {
        'NN_model.linearIn.weight': state_dict['lstminv1.linearIn.weight'],
        'NN_model.linearIn.bias': state_dict['lstminv1.linearIn.bias'],
        'NN_model.lstm.w_ih': state_dict['lstminv1.lstm.w_ih'],
        'NN_model.lstm.w_hh': state_dict['lstminv1.lstm.w_hh'],
        'NN_model.lstm.b_ih': state_dict['lstminv1.lstm.b_ih'],
        'NN_model.lstm.b_hh': state_dict['lstminv1.lstm.b_hh'],
        'NN_model.linearOut.weight': state_dict['lstminv1.linearOut.weight'],
        'NN_model.linearOut.bias': state_dict['lstminv1.linearOut.bias'],
        'ANN_model.i2h.weight': state_dict['Ann.i2h.weight'],
        'ANN_model.i2h.bias': state_dict['Ann.i2h.bias'],
        'ANN_model.h2h1.weight': state_dict['Ann.h2h1.weight'],
        'ANN_model.h2h1.bias': state_dict['Ann.h2h1.bias'],
        'ANN_model.h2h2.weight': state_dict['Ann.h2h2.weight'],
        'ANN_model.h2h2.bias': state_dict['Ann.h2h2.bias'],
        'ANN_model.h2h3.weight': state_dict['Ann.h2h3.weight'],
        'ANN_model.h2h3.bias': state_dict['Ann.h2h3.bias'],
        'ANN_model.h2h4.weight': state_dict['Ann.h2h4.weight'],
        'ANN_model.h2h4.bias': state_dict['Ann.h2h4.bias'],
        'ANN_model.h2h5.weight': state_dict['Ann.h2h5.weight'],
        'ANN_model.h2h5.bias': state_dict['Ann.h2h5.bias'],
        'ANN_model.h2h6.weight': state_dict['Ann.h2h6.weight'],
        'ANN_model.h2h6.bias': state_dict['Ann.h2h6.bias'],
        'ANN_model.h2o.weight': state_dict['Ann.h2o.weight'],
        'ANN_model.h2o.bias': state_dict['Ann.h2o.bias'],
    }

    # Load the new state dict into the model
    model = dModel(config, model_architecture).to(config['device'])
    model.load_state_dict(new_state_dict)

    # Save the new state dict and model
    torch.save(new_state_dict, f"{save_path}HBV_waterLoss_states_Ep50.pt")
    torch.save(model, f"{save_path}HBV_waterLoss_model_Ep50.pt")


if __name__ == '__main__':
    state_path = '../../../results/camels_531/debugging/train_1980_1995/3_forcing/no_ensemble/LSTM_E50_R365_B100_H64_n4_111111/HBV_/NseLossBatchFlow_/dynamic_para/parBETA_parBETAET_/model_states_50.pt'
    
    save_path = '/../../../results/camels_531/debugging/train_1980_1995/3_forcing/no_ensemble/LSTM_E50_R365_B100_H64_n4_111111/HBV_/NseLossBatchFlow_/dynamic_para/parBETA_parBETAET_/'
    config = {
        'device': 'cuda:0',  # or 'cpu'
    }

    convert_and_load_model_state(state_path, save_path, config)

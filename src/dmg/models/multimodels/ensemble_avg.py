from typing import Any

import torch

from dmg.core.utils.utils import find_shared_keys


def model_average(
    model_preds_dict: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    For any number of metrics specified in the input dictionary, calculate
    composite predictions as the average of multiple models' outputs at each
    basin for each day.

    Parameters
    ----------
    model_preds_dict
        Dictionary of model predictions

    config
        Dictionary of model configuration.
        
    Returns
    -------
    dict[str, Any]
        predictions dict with attributes
        'flow_sim', 'srflow', 'ssflow', 'gwflow', 'AET_hydro', 'PET_hydro',
        'flow_sim_no_rout', 'srflow_no_rout', 'ssflow_no_rout', 'gwflow_no_rout',
        'BFI_sim'
    """
    ensemble_pred = {}

    # Get prediction shared between all models.
    mod_dicts = [model_preds_dict[mod] for mod in config['hydro_models']]
    shared_keys = find_shared_keys(*mod_dicts)

     # TODO: identify why 'flow_sim_no_rout' calculation returns shape [365,1]
    # vs [365, 100] which breaks the ensemble loop at point of matrix mul below. (weights_dict[mod]
    # takes shape [365, 100].) Look at `QSIM` comprout vs no comprout in HBVmul.py. For now, remove it.
    # NOTE: may have fixed this ^^^ need to confirm.
    shared_keys.remove('flow_sim_no_rout')

    print(shared_keys)
    for key in shared_keys:
        ensemble_pred[key] = 0
        for mod in config['hydro_models']:
            if len(model_preds_dict[mod][key].shape) > 1:
                # Cut out warmup data present when testing model from loaded mod file.
                print(key, model_preds_dict[mod][key].size())


                ensemble_pred[key] += model_preds_dict[mod][key][
                config['warm_up']:,:].squeeze()
            else:
                print(key)
                ensemble_pred[key] += model_preds_dict[mod][key]
        # Compute avg
        ensemble_pred[key] = ensemble_pred[key] / len(config['hydro_models'])

    return ensemble_pred


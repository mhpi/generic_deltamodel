from typing import Any, Dict
from pydantic import BaseModel
import os
import json
import hashlib


class PathBuilder(BaseModel):
    """Build and initialize output directories for saving models and outputs.
    
    Using Pydantic BaseModel to enforce type checking and validation.
    """
    config: Dict[str, Any] = {}

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config=config)

    def model_post_init(self, __context: Any) -> Any:
        """Post-initialization method to create output directories.
        
        This method is called after the model is initialized.
        """
        self.write_output_dir(self.config)  



    def write_output_dir(config: Dict[str, Any]) -> dict:
        """Creates directory where model and outputs will be saved.

        Creates all root directories to support the target directory for the model.

        Parameters
        ----------
        config : dict
            Configuration dictionary with paths and model settings.
        
        Returns
        -------
        dict
            The original config with path modifications.
        """
        # Dir for dataset name:
        dataset_name = f"{config['observations']['name']}"

        # Dir for training period:
        train_period = f"train{config['train']['start_time'][:4]}-{config['train']['end_time'][:4]}"

        ## TODO: Leave out until better integration (e.g., what if same count, different inputs?).
        # Dir for phy_model input count:
        # attributes = config['dpl_model']['phy_model']['attributes']
        # if attributes == []:
        #     attributes = 0
        # num_inputs = f"{config['dpl_model']['phy_model']['forcings']}dy_{attributes}st_in"

        # Dir for multimodel state:
        multimodel_state = config['multimodel_type'] or 'no_multi'

        # Dirs for model names and parameters (hashed for concision):
        models = config['dpl_model']['phy_model']['model']
        parameters = config['dpl_model']['phy_model']['dynamic_params']

        model_names = '_'.join(models)
        dynamic_param_str = '_'.join(
            param for model in models for param in parameters[model]
        )

        dynamic_param_hash = hashlib.md5(dynamic_param_str.encode()).hexdigest()[:8]
        
        # Dir for dynamic or static physical parameter state:
        dynamic_state = 'dyn' if dynamic_param_str else 'static'

        # Dir for loss function:
        loss_fn = '_'.join(
            fn for model in models for fn in config['loss_function']['model']
        )

        # Dir for hyperparemter details:
        norm = 'noLn'
        norm_list = config['dpl_model']['phy_model']['use_log_norm']
        if norm_list != []:
            vars = '_'.join(var for var in norm_list)
            norm = f"Ln_{vars}"

        params = f"{config['dpl_model']['nn_model']['model']}_ \
            E{config['train']['epochs']}_ \
            R{config['dpl_model']['rho']}_ \
            B{config['train']['batch_size']}_ \
            H{config['dpl_model']['nn_model']['hidden_size']}_ \
            n{config['dpl_model']['phy_model']['nmul']}_ \
            {norm}_ \
            {config['random_seed']}"
        
        # Full root path
        model_path = os.path.join(
            config['save_path'],
            dataset_name,
            train_period,
            multimodel_state,
            params,
            model_names,
            loss_fn,
            dynamic_state,
            dynamic_param_hash if dynamic_state == 'dyn' else ''
        )

        # Dir for test period:
        test_period = f"test{config['test']['start_time'][:4]}-{config['test']['end_time'][:4]}"
        test_path = os.path.join(model_path, test_period)

        # Create dirs
        os.makedirs(test_path, exist_ok=True)

        # Append the output directories to the config.
        config['out_path'] = model_path
        config['validation_path'] = test_path
        
        # Save config metadata (overwrite if it exists).
        config_path = os.path.join(model_path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        return config

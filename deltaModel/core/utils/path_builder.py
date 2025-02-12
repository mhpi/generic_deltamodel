import hashlib
import json
import os
from typing import Any, Dict

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PathBuilder(BaseModel):
    """Build and initialize output directories for saving models and outputs.
    
    Using Pydantic BaseModel to enforce type checking and validation.
    Scalable and flexible for diverse model configurations.

    Parameters
    ----------
    config : dict
        Configuration dictionary with experiment and model settings.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_config['protected_namespaces'] = ()

    config: Dict[str, Any] = Field(..., description="Experiment configuration dictionary")
    base_path: str = ''
    dataset_name: str = ''
    phy_model_inputs: str = ''
    train_period: str = ''
    test_period: str = ''
    multimodel_state: str = ''
    model_names: str = ''
    dynamic_parameters: str = ''
    dynamic_state: str = ''
    loss_function: str = ''
    hyperparameter_detail: str = ''

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config=config)

    def model_post_init(self, __context: Any) -> Any:
        """Post-initialization method to create output directories.
        
        This method is called after the model is initialized.
        """
        self.base_path = self.config['save_path']

        self.dataset_name = self._dataset_name(self.config)
        self.phy_model_inputs = self._phy_model_inputs(self.config)

        self.train_period = self._train_period(self.config, abbreviate=False)
        self.test_period = self._test_period(self.config, abbreviate=False)

        self.multimodel_state = self._multimodel_state(self.config)

        self.model_names = self._model_names(self.config)
        self.dynamic_parameters = self._dynamic_parameters(self.config, hash=False)
        self.dynamic_state = self._dynamic_state()

        self.loss_function = self._loss_function(self.config)
        self.hyperparameter_detail = self._hyperparameter_details(self.config)

        return super().model_post_init(__context)
    
    @field_validator('config')
    def validate_config(cls, value) -> Dict[str, Any]:
        """Validate the configuration dictionary."""
        required_keys = ['save_path', 'train', 'test', 'dpl_model']
        for key in required_keys:
            if key not in value:
                raise ValueError(f"Missing required configuration key: {key}")
        return value
    
    def build_path_model(self) -> str:
        """Build path to model object from individual root paths."""
        return os.path.join(
            self.base_path,
            self.dataset_name,
            self.train_period,
            self.multimodel_state,
            self.hyperparameter_detail,
            self.model_names,
            self.loss_function,
            self.dynamic_state,
            self.dynamic_parameters,
        )

    def build_path_out(self, model_path: str = None) -> Dict[str, Any]:
        """Build path to model outputs from individual root paths.
        
        Parameters
        ----------
        model_path : str
            Path to the model object.
        
        Returns
        -------
        str
            Path to the output directory.
        """
        if model_path:
            return os.path.join(
                model_path,
                self.test_period,
            )
        else:
            return os.path.join(
                self.build_path_model(),
                self.test_period,
            )

    def write_path (self, config: Dict[str, Any]) -> dict:
        """Create directory where model and outputs will be saved.

        Creates all root directories to support the target directory.
        
        Returns
        -------
        dict
            The original config with path modifications.
        """
        # Check base path
        self.validate_base_path(config['save_path'])

        # Build paths
        if os.path.exists(config.get('trained_model', '')):
            # Use user defined model path if it exists
            model_path = os.path.dirname(config['trained_model'])
            out_path = self.build_path_out(model_path)
        else:
            model_path = self.build_path_model()
            out_path = self.build_path_out(model_path)
        
        # Create dirs
        if config['mode'] not in ['test', 'predict']:
            os.makedirs(model_path, exist_ok=True)
            os.makedirs(out_path, exist_ok=True)
        elif os.path.exists(model_path):
            os.makedirs(out_path, exist_ok=True)
        else:
            raise ValueError(f"No model to validate at path {model_path}")

        # Append the output paths to the config.
        config['model_path'] = model_path
        config['out_path'] = out_path
        
        # Save config
        serializable_config = self.make_json_serializable(config)
        self.save_config(model_path, serializable_config)
        
        return config

    def make_json_serializable(self, obj: Any) -> Any:
        """
        Recursively converts objects in a dictionary to JSON-serializable
        formats.
        """
        if isinstance(obj, dict):
            return {k: self.make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_json_serializable(v) for v in obj]
        elif isinstance(obj, torch.dtype):  # Handle torch.dtype specifically
            return str(obj)  # Convert dtype to a string
        elif hasattr(obj, '__dict__'):  # Handle objects with attributes
            return self.make_json_serializable(vars(obj))
        else:
            return obj  # Return as is for natively serializable types
        
    @staticmethod
    def save_config(path: str, config: Dict[str, Any]) -> None:
        """Save the configuration metadata to the output directory.
        
        Overwrite if it already exists.
        """
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
    @staticmethod
    def validate_base_path(base_path: Any) -> Any:
        """Check that the base path exists. If not, attempt to create it.
        
        Parameters
        ----------
        base_path : Any
            Base path for saving models and outputs.
        """
        base_path = os.path.abspath(base_path)

        if not os.path.exists(base_path):
            try:
                os.makedirs(base_path)
            except Exception as e:
                raise ValueError(f"Error creating base save path from config: {e}")
    
    @staticmethod
    def _dataset_name(config: Dict[str, Any]) -> str:
        """Name of the dataset used."""
        return f"{config['observations']['name']}"

    @staticmethod
    def _phy_model_inputs(config: Dict[str, Any]) -> str:
        """Number of physical model input variables.
        
        TODO: needs more thought (e.g. what is same count, different inputs?)
        ...maybe use hash.
        """
        attributes = config['dpl_model']['phy_model'].get('attributes', '')
        if attributes == []:
            attributes = 0
        return f"{config['dpl_model']['phy_model']['forcings']}dy_{attributes}st_in"
    
    @staticmethod
    def _train_period(config: Dict[str, Any], abbreviate: bool = False) -> str:
        """Training period for an experiment.

        Format is 
            'trainYYYY-YYYY' or 'trainYY-YY' if abbreviated.
        """
        start = config['train']['start_time'][:4]
        end = config['train']['end_time'][:4]

        if abbreviate:
            return f"train{start[-2:]}-{end[-2:]}"
        else: 
            return f"train{start}-{end}"

    @staticmethod
    def _test_period(config: Dict[str, Any], abbreviate: bool = False) -> str:
        """Testing period for an experiment.

        Format is
            'testYYYY-YYYY' or 'testYY-YY' if abbreviate.
        """
        start = config['test']['start_time'][:4]
        end = config['test']['end_time'][:4]
        test_epoch = config['test'].get('test_epoch', '')

        if test_epoch:
            test_epoch = f"_Ep{test_epoch}"
        else:
            test_epoch = ''

        if abbreviate:
            return f"test{start[-2:]}-{end[-2:]}" + test_epoch
        else: 
            return f"test{start}-{end}" + test_epoch

    @staticmethod
    def _multimodel_state(config: Dict[str, Any]) -> str:
        """Name multimodel state for an experiment."""
        return config['multimodel_type'] or 'no_multi'

    @staticmethod
    def _model_names(config: Dict[str, Any]) -> str:
        """Names of the models used in an experiment."""
        models = config['dpl_model']['phy_model']['model']
        return '_'.join(models)

    @staticmethod
    def _dynamic_parameters(config: Dict[str, Any], hash: bool = False) -> str:
        """Dynamic parameters used in the model(s).
        
        Parameters
        ----------
        hash : bool
            If True, returns a short hash of the dynamic parameters.

        Format is
            'p1_p2_q1_q2' etc. for model1 parameters p1, p2,... 
            and model2 parameters q1, q2,...

            or

            'abc12345' if hash.
        """
        models = config['dpl_model']['phy_model']['model']
        parameters = config['dpl_model']['phy_model']['dynamic_params']

        param_str = '_'.join(
            param for model in models for param in parameters.get(model, [])
        )
        if not param_str:
            return ''

        if hash:
            return hashlib.md5(param_str.encode()).hexdigest()[:8]
        return param_str
        
    def _dynamic_state(self) -> str:
        """Identify if any physical model parameters are dynamic.
        
        Parameters
        ----------
        dynamic_parameters : str
            String of dynamic parameters used in the model(s).
        """
        param_count = len(self.dynamic_parameters.split('_'))
        if self.dynamic_parameters == '':
            return 'stat'
        else:
            return f"{param_count}dyn"

    @staticmethod
    def _loss_function(config: Dict[str, Any]) -> str:
        """Loss function(s) used in the model(s)."""
        models = config['dpl_model']['phy_model']['model']
        loss_fn = config['loss_function']['model']
        loss_fn_str = '_'.join(
            loss_fn for model in models
        )          
        return loss_fn_str

    @staticmethod
    def _hyperparameter_details(config: Dict[str, Any]) -> str:
        """Details of hyperparameters used in the model(s)."""
        norm = 'noLn'
        norm_list = config['dpl_model']['phy_model']['use_log_norm']
        if norm_list:
            vars = '_'.join(norm_list)
            norm = f"Ln_{vars}"
        
        warmup = 'noWU'
        if config['dpl_model']['phy_model']['warm_up_states']:
            warmup = 'WU'

        # Set hiddensize for single or multi-NN setups.
        if config['dpl_model']['nn_model']['model'] == 'LSTMMLP':
            hidden_size = f"{config['dpl_model']['nn_model']['lstm_hidden_size']}" \
                            f"_{config['dpl_model']['nn_model']['mlp_hidden_size']}"
        else:
            hidden_size = config['dpl_model']['nn_model']['hidden_size']

        return (
            f"{config['dpl_model']['nn_model']['model']}_"
            f"E{config['train']['epochs']}_"
            f"R{config['dpl_model']['rho']}_"
            f"B{config['train']['batch_size']}_"
            f"H{hidden_size}_"
            f"n{config['dpl_model']['phy_model']['nmul']}_"
            f"{norm}_"
            f"{warmup}_"
            f"{config['random_seed']}"
        )

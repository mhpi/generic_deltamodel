import logging
import os
from typing import Any, Dict, List, Optional

import torch.nn
from core.utils import save_model
from models.differentiable_model import DeltaModel

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class ModelHandler(torch.nn.Module):
    """Streamlines handling of differentiable models and multimodel ensembles.

    This interface additionally acts as a link to the CSDMS BMI, enabling
    compatibility with the NOAA-OWP NextGen framework.

    Features
    - Model initialization (new or from a checkpoint)
    - Loss calculation
    - Forwarding for single/multi-model setups
    - (Planned) Multimodel ensembles/loss and multi-GPU compute

    Parameters
    ----------
    config : dict
        Configuration settings for the model.
    device : str, optional
        Device to run the model on. Default is None.
    verbose : bool, optional
        Whether to print verbose output. Default is False.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[str] = None,
        verbose=False
    ) -> None:
        super().__init__()
        self.config = config
        self.name = 'Differentiable Model Handler'
        self.save_path = config['out_path']
        self.verbose = verbose

        if device is None:
            self.device = config['device']
        else:
            self.device = device

        self.model_dict = {}
        self._init_models()
        self.loss_func = None
        self.loss_dict = {key: 0 for key in self.models_to_initialize}
        if verbose:
            log.info(f"Initialized {self.name} with {self.models_to_initialize}.")

    @property
    def models_to_initialize(self) -> List[str]:
        """List of models specified in the configuration."""
        return self.config['dpl_model']['phy_model']['model']
    
    def _init_models(self) -> None:
        """Initialize and store models, multimodels, and checkpoints."""
        if self.config['multimodel_type'] == 'none' and len(self.models_to_initialize) > 1:
            raise ValueError(
                "Multiple models specified, but ensemble type is 'none'. Check configuration."
            )
        if self.config['mode'] == 'train':
            load_epoch = self.config['train']['start_epoch']
        elif self.config['mode'] == 'test':
            load_epoch = self.config['test']['test_epoch']
        else:
            load_epoch = self.config.get('load_epoch', 0)
        
        # Initialize new models or load from a saved checkpoint
        try:
            self.load_model(load_epoch)
        except Exception as e:
            raise e

    def load_model(self, epoch: int = 0) -> None:
        """Load a specific model from a checkpoint.
        
        Parameters
        ----------
        epoch : int 
            Epoch to load the model from. Default is 0.
        """
        for name in self.models_to_initialize:
            # Created new model
            self.model_dict[name] = DeltaModel(
                phy_model_name=name,
                config=self.config['dpl_model'],
                device=self.device
            )

            if epoch == 0:
                # Leave model uninitialized
                if self.verbose:
                    log.info(f"Created new model: {name}")
                continue 
            
            else:
                # Initialize model from checkpoint state dict.
                path = os.path.join(self.save_path, f"{name}_model_Ep{epoch}.pt")
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"{path} not found for model {name}."
                    )
                self.model_dict[name].load_state_dict(torch.load(path))
                self.model_dict[name].to(self.device)
                
                # Overwrite internal config if there is discontinuity:
                if self.model_dict[name].config:
                    self.model_dict[name].config = self.config

                if self.verbose:
                    log.info(f"Loaded model: {name}, Ep {epoch}")

    def _new_model_instance(self, model_name: str) -> DeltaModel:
        """Create a new instance of a differentiable model.
        
        Multi-GPU support is planned here.

        Parameters
        ----------
        model_name : str
            Name of the model to initialize.
        """
        return DeltaModel(
            phy_model_name=model_name,
            config=self.config['dpl_model'],
            device=self.device
        )

    def get_parameters(self) -> List[torch.Tensor]:
        """Return all model parameters."""
        self.parameters = []
        for model in self.model_dict.values():
            self.parameters += list(model.parameters())
        return self.parameters
        
    def forward(
        self,
        dataset_dict: Dict[str, torch.Tensor],
        eval: bool = False
    ) -> Dict[str, torch.Tensor]:        
        """Sequentially forward one or more differentiable models.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of model outputs. Each key corresponds to a model name.
        """
        self.output_dict = {}
        for name, model in self.model_dict.items():
            if eval:
                ## Inference mode
                model.eval()
                with torch.no_grad():
                    self.output_dict[name] = model(dataset_dict)
            else:
                ## Training mode
                model.train()
                self.output_dict[name] = model(dataset_dict)
        return self.output_dict

    def calc_loss(
        self,
        dataset: Dict[str, torch.Tensor],
        loss_func: Optional[torch.nn.Module] = None
    ) -> torch.Tensor:
        """Calculate combined loss across all models.
        
        Parameters
        ----------
        dataset : dict
            Dataset dictionary containing observation data.
        loss_func : nn.Module, optional
            Loss function to use. Default is None.
        """
        if not self.loss_func and not loss_func:
            raise ValueError("No loss function defined.")
        loss_func = loss_func or self.loss_func

        loss_combined = 0.0

        target_name = self.config['train']['target'][0]

        for name, output in self.output_dict.items():
            if target_name not in output.keys():
                raise ValueError(f"Target variable '{target_name}' not in model outputs.")
            output = output[target_name]

            loss = loss_func(
                output,
                dataset['target'],
                n_samples=dataset['batch_sample']
            )
            loss_combined += loss
            self.loss_dict[name] += loss.item()
        return loss_combined

    def save_model(self, epoch: int) -> None:
        """Save state dictionary of trained models to disk.
        
        Parameters
        ----------
        epoch : int
            Epoch number model will be saved at.
        """
        for name, model in self.model_dict.items():
            save_model(self.config, model, name, epoch)
            if self.verbose:
                log.info(f"Saved model: {name}, Ep {epoch}")

from math import log
from typing import Any, Dict, List, Optional, Tuple

import torch
from core.data import numpy_to_torch_dict
from core.utils.utils import find_shared_keys
from models.neural_networks.lstm_models import CudnnLstmModel
from models.neural_networks.mlp_models import MLPmul


class EnsembleGenerator(torch.nn.Module):
    """Default class for instantiating a multimodel ensemble weights generator.

    Default modality:
        NN trained in series or parallel with multiple differentiable models
        to learn weights for spatiotemporal ensembling.
    
    Parameters
    ----------
    model_list : list
        List of names of differentiable models to ensemble.
    config : dict
        The configuration dictionary.
    wnn_model : torch.nn.Module
        The neural network model to learn weights for multimodel ensembling.
        The default is None.
    device : torch.device, optional
        The device to run the model on. The default is None.
    """
    def __init__(
            self,
            model_list: List[str],
            config: Dict[str, Any],
            wnn_model: torch.nn.Module = None,
            device: Optional[torch.device] = None
        ) -> None:
        super().__init__()
        self.name = "Multimodel Ensemble Weights Generator"
        self.config = config
        self.model_list = model_list
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if wnn_model:
            self.wnn_model = wnn_model
        elif config:
            self.wnn_model = self._init_wnn_model()
        else:
            raise ValueError("A (1) neural network or (2) configuration dictionary is required.")
        
        self.weights = {}
        self.ensemble_predictions = {}
        self.initialized = True

    def _init_wnn_model(self) -> torch.nn.Module:
        """Initialize a wNN model.
        
        wNN to learn weights for multimodel ensembling.
            Inputs: forcings/attributes/observed variables.
            Outputs: weights for each hydrology model.
        
        TODO: Set this up as dynamic module import instead.
        """
        n_forcings = len(self.config['forcings'])
        n_attributes = len(self.config['attributes'])
        
        # Number of inputs 'x' and outputs 'y' for wnn
        self.nx = n_forcings + n_attributes
        self.ny = len(self.model_list)
        
        model_name = self.config['model']

        # Initialize the nn
        if model_name == 'LSTM':
            model = CudnnLstmModel(
                nx=self.nx,
                ny=self.ny,
                hiddenSize=self.config['hidden_size'],
                dr=self.config['dropout']
            )
        elif model_name == 'MLP':
            model = MLPmul(
                self.config,
                nx=self.nx,
                ny=self.ny
            )
        else:
            raise ValueError(f"{model_name} is not a supported neural network model type.")
        return model.to(self.device)

    def forward(
            self,
            dataset_dict: Dict[str, torch.Tensor],
            predictions: Dict[str, torch.Tensor],
        ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass for the model.
        
        Generate ensemble weights and ensemble model predictions.

        Parameters
        ----------
        dataset_dict : dict
            Dictionary containing input data.
        predictions : dict
            Dictionary containing predictions from individual models.
        """

        if self.config['mosaic'] == False:
            # Ensure input data is in the correct format and device.
            dataset_dict = numpy_to_torch_dict(dataset_dict, device=self.device)
            
            # Generate ensemble weights
            self._raw_weights = self.wnn_model(dataset_dict['x_nn_scaled'])
            self._scale_weights()

            # Map weights to individual models
            diff = dataset_dict['x_nn_scaled'].shape[0] - dataset_dict['target'].shape[0]
            self.weights = {
                model: self.weights_scaled[diff:, :, i]
                for i, model in enumerate(self.model_list)
            }

            # Linearly combine individual model predictions.
            predictions_list = [predictions[model] for model in self.model_list]
            shared_keys = find_shared_keys(*predictions_list)
            
            for key in shared_keys:
                self.ensemble_predictions[key] = sum(
                    self.weights[model] * predictions[model][key].squeeze()
                    for model in self.model_list
                )
        else:
            print("Mosaic mode is enabled.")
            # Generate ensemble weights
            self._raw_weights = self.wnn_model(dataset_dict['x_nn_scaled'])
            self._scale_weights()

            # Map weights to individual models
            diff = dataset_dict['x_nn_scaled'].shape[0] - dataset_dict['target'].shape[0]
            self.weights = {
                model: self.weights_scaled[diff:, :, i]
                for i, model in enumerate(self.model_list)
            }

            # Convert weights to a tensor for easier manipulation
            weights_tensor = torch.stack(list(self.weights.values()), dim=0)  # Shape: [num_models, num_timesteps, num_basins]

            # Step 1: Find the index of the model with the highest weight
            best_model_idx = weights_tensor.argmax(dim=0)  # Shape: [num_timesteps, num_basins]

            # Step 2: Create a mask to select the model with the highest weight
            # Create a one-hot mask
            mask = torch.zeros_like(weights_tensor, dtype=torch.bool)
            mask.scatter_(0, best_model_idx.unsqueeze(0), True)  # Shape: [num_models, num_timesteps, num_basins]
            
            # Step 3: Use the mask to select predictions
            # Linearly combine individual model predictions.
            predictions_list = [predictions[model] for model in self.model_list]
            shared_keys = find_shared_keys(*predictions_list)
            
            for key in shared_keys:
                predictions_tensor = torch.stack([predictions[model][key].squeeze() for model in self.model_list], dim=0)  
                # Shape: [num_models, num_timesteps, num_basins]

                if predictions_tensor.ndim == 2:
                    predictions_tensor = predictions_tensor.unsqueeze(0)
                elif predictions_tensor.ndim == 3:
                    pass
                else:
                    # Skip BFI key with shape 1.
                    continue
            
                final_predictions = torch.gather(predictions_tensor, 0, best_model_idx.unsqueeze(0)).squeeze(0)

                self.ensemble_predictions[key] = final_predictions

                
                # # Mask out all but the best model's predictions
                # selected_predictions = predictions_tensor * mask  # Shape: [num_models, num_timesteps, num_basins]

                # # Step 4: Aggregate over the model dimension
                # final_predictions = selected_predictions.sum(dim=0)  # Shape: [num_timesteps, num_basins]

                # # Store the ensemble predictions
                # self.ensemble_predictions[key] = final_predictions

        ### Note: Potentially more efficient calculation with tensors.
        # # Stack all model predictions for shared keys into a single tensor
        # predictions_stack = {
        #     key: torch.stack([predictions[mod][key].squeeze(dim=-1) for mod in self.model_list], dim=-1)
        #     for key in shared_keys
        # }

        # # Convert weights dictionary to a single tensor for batch operations
        # weights_tensor = torch.stack([self.weights[mod] for mod in self.model_list], dim=-1)

        # # Compute ensemble predictions for all shared keys in one step
        # self.ensemble_predictions = {
        #     key: torch.sum(predictions_stack[key] * weights_tensor, dim=-1)
        #     for key in shared_keys
        # }

        return self.ensemble_predictions, self.weights

    def _scale_weights(self) -> None:
        """Scale weights with an activation function."""
        method = self.config['scaling_function']
        if method == 'sigmoid':
            self.weights_scaled = torch.sigmoid(self._raw_weights)
        elif method == 'softmax':
            self.weights_scaled = torch.softmax(self._raw_weights, dim=1)
        else:
            raise ValueError(f"Invalid weighting method: {method}")

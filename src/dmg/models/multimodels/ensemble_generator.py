from typing import Any, Optional

import torch

from dmg.core.utils.factory import load_nn_model
from dmg.core.utils.utils import find_shared_keys


class EnsembleGenerator(torch.nn.Module):
    """Default class for instantiating a multimodel ensemble weights generator.

    Default modality:
        NN trained in series or parallel with multiple differentiable models
        to learn weights for spatiotemporal ensembling.
    
    Parameters
    ----------
    model_list
        List of names of differentiable models to ensemble.
    config
        Configuration dictionary.
    nn_model
        The neural network model to learn weights for multimodel ensembling.
        Default is None.
    device
        The device to run the model on. Default is None.
    """
    def __init__(
        self,
        model_list: list[str],
        config: dict[str, Any],
        nn_model: torch.nn.Module = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.name = "Multimodel Ensemble Weights Generator"
        self.config = config
        self.model_list = model_list
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if nn_model:
            self.nn_model = nn_model
        elif config:
            self.nn_model = self._init_nn_model()
        else:
            raise ValueError("A (1) neural network or (2) configuration dictionary is required.")

        self.weights = {}
        self.ensemble_predictions = {}
        self.initialized = True

    def _init_nn_model(self) -> torch.nn.Module:
        """Initialize a neural network model.
        
        Returns
        -------
        nn_model
            The neural network model.
        """
        return load_nn_model(
            None,
            self.config,
            ensemble_list=self.model_list,
            device=self.device,
        )

    def forward(
        self,
        dataset_dict: dict[str, torch.Tensor],
        predictions: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Forward pass for the model.
        
        Generate ensemble weights and ensemble model predictions.

        Parameters
        ----------
        dataset_dict
            Dictionary containing input data.
        predictions
            Dictionary containing predictions from individual models.
        
        Returns
        -------
        ensemble_predictions
            Dictionary of ensemble predictions and model weights.
        """
        if not self.config['mosaic']:
            # Ensure input data is in the correct format and device.
            # dataset_dict = numpy_to_torch_dict(dataset_dict, device=self.device)

            # Generate ensemble weights
            self._raw_weights = self.nn_model(dataset_dict['xc_nn_norm'])
            self._scale_weights()

            # Map weights to individual models
            diff = dataset_dict['xc_nn_norm'].shape[0] - predictions[self.model_list[0]]['streamflow'].shape[0]
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
            self._raw_weights = self.nn_model(dataset_dict['xc_nn_norm'])
            self._scale_weights()

            # Map weights to individual models
            diff = dataset_dict['xc_nn_norm'].shape[0] - dataset_dict['target'].shape[0]
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

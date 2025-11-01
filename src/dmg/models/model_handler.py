import logging
import os
from typing import Any, Optional

import torch
import tqdm

from dmg.core.utils import save_model
from dmg.models.criterion.range_bound_loss import RangeBoundLoss
from dmg.models.delta_models.dpl_model import DplModel
from dmg.models.multimodels.ensemble_generator import EnsembleGenerator
from dmg.models.wrappers.nn_model import NnModel

log = logging.getLogger('model_handler')


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
    config
        Configuration settings for the model.
    device
        Device to run the model on.
    verbose
        Whether to print verbose output.
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = None,
        verbose=False,
    ) -> None:
        super().__init__()
        self.config = config
        self.name = 'Differentiable Model Handler'
        self.model_type = None
        self.model_path = config['model_dir']
        self.verbose = verbose

        if device is None:
            self.device = config['device']
        else:
            self.device = device

        self.multimodel_type = config['multimodel_type']
        self.model_dict = {}
        # TODO: add proper support for multiple targets...
        self.target_names = config['train']['target']
        self.models = self.list_models()
        self._init_models()

        self.epoch = None
        self.loss_func = None
        self.loss_dict = dict.fromkeys(self.models, 0)

        if self.multimodel_type in ['nn_parallel']:
            self.is_ensemble = True
            self.weights = {}
            self.loss_func_wnn = None
            self.range_bound_loss = RangeBoundLoss(config, device=self.device)
        self.is_ensemble = False

    def list_models(self) -> list[str]:
        """List of models specified in the configuration.

        TODO: Support physics-only forward.

        Returns
        -------
        list[str]
            List of model names.
        """
        if self.config['model']['phy']:
            models = self.config['model']['phy']['name']
            self.model_type = 'dm'
        elif self.config['model']['nn']:
            models = self.config['model']['nn']['name']
            self.model_type = 'nn'
        else:
            raise ValueError("No models specified in configuration.")

        if self.multimodel_type in ['nn_parallel']:
            # Add ensemble weighting NN to the list.
            models.append('wNN')

        if not isinstance(models, list):
            models = [models]

        return models

    def _init_models(self) -> None:
        """Initialize and store models, multimodels, and checkpoints."""
        if (self.multimodel_type is None) and (len(self.models) > 1):
            raise ValueError(
                "Multiple models specified, but ensemble type is 'none'. Check configuration.",
            )

        # Epoch to load
        if self.config['mode'] == 'train':
            load_epoch = self.config['train']['start_epoch']
        elif self.config['mode'] in ['test', 'sim']:
            load_epoch = self.config['test']['test_epoch']
        else:
            load_epoch = self.config.get('load_epoch', 0)

        # Load models
        try:
            self.load_model(load_epoch)
        except Exception as e:
            raise e

    def load_model(self, epoch: int = 0) -> None:
        """Load a specific model from a checkpoint.

        Parameters
        ----------
        epoch
            Epoch to load the model from.
        """
        for name in self.models:
            # Created new model
            if name == 'wNN':
                # Ensemble weighting NN
                self.ensemble_generator = EnsembleGenerator(
                    config=self.config['multimodel'],
                    model_list=self.models[:-1],
                    device=self.device,
                )

            elif self.model_type == 'nn':
                # Standalone neural network model
                self.model_dict[name] = NnModel(
                    target_names=self.target_names,
                    config=self.config['model'],
                    device=self.device,
                )
            else:
                # Differentiable model (dPL modality)
                # TODO: make dynamic import for other modalities.
                self.model_dict[name] = DplModel(
                    phy_model_name=name,
                    config=self.config['model'],
                    device=self.device,
                )

            if epoch == 0:
                self.epoch = 0

                # Leave model uninitialized for training.
                if self.verbose:
                    log.info(f"Created new model: {name}")
                continue
            else:
                self.epoch = epoch

                # Initialize model from checkpoint state dict.
                path = self.model_path
                if f"{name.lower()}_ep" not in path:
                    path = os.path.join(path, f"{name.lower()}_ep{epoch}.pt")
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"{path} not found for model {name}.",
                    )
                if name == 'wNN':
                    self.ensemble_generator.load_state_dict(
                        torch.load(
                            path,
                            weights_only=False,
                            map_location=self.device,
                        )
                    )
                    self.ensemble_generator.to(self.device)
                else:
                    self.model_dict[name].load_state_dict(
                        torch.load(
                            path,
                            weights_only=True,
                            map_location=self.device,
                        ),
                    )
                    self.model_dict[name].to(self.device)

                    # Overwrite internal config if there is discontinuity:
                    if (self.model_type == 'dm') and self.model_dict[name].config:
                        self.model_dict[name].config = self.config['model']

                if self.verbose:
                    log.info(f"Loaded model: {name}, Ep {epoch}")

    def get_parameters(self) -> list[torch.Tensor]:
        """Return all model parameters.

        Returns
        -------
        list[torch.Tensor]
            List of model parameters.
        """
        self.parameters = []
        for model in self.model_dict.values():
            # Differentiable model parameters
            self.parameters += list(model.parameters())

        if self.multimodel_type in ['nn_parallel']:
            # Ensemble weighting NN parameters if trained in parallel.
            self.parameters += list(self.ensemble_generator.parameters())
        return self.parameters

    def forward(
        self,
        dataset_dict: dict[str, torch.Tensor],
        eval: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Sequentially forward one or more models with an optional weighting NN
        for multimodel ensembles trained in parallel or series (model
        parameterization NNs frozen).

        Parameters
        ----------
        dataset_dict
            Dictionary containing input data.
        eval
            Whether to run the model in evaluation mode with gradients
            disabled.

        Returns
        -------
        dict[str, torch.Tensor]
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

        if self.multimodel_type in ['nn_parallel']:
            self._forward_multimodel(dataset_dict, eval)
            return {list(self.model_dict.keys())[0]: self.ensemble_output_dict}
        else:
            return self.output_dict

    def _forward_multimodel(
        self,
        dataset_dict: dict[str, torch.Tensor],
        eval: bool = False,
    ) -> None:
        """
        Augment model outputs: Forward wNN and combine model outputs for
        multimodel ensemble predictions.

        Parameters
        ----------
        dataset_dict
            Dictionary containing input data.
        eval
            Whether to run the model in evaluation mode with gradients
            disabled.
        """
        if eval:
            ## Inference mode
            self.ensemble_generator.eval()
            with torch.no_grad():
                self.ensemble_output_dict, self.weights = self.ensemble_generator(
                    dataset_dict,
                    self.output_dict,
                )
        else:
            if self.multimodel_type in ['nn_parallel']:
                ## Training mode for parallel-trained ensemble.
                self.ensemble_generator.train()
                self.ensemble_output_dict, self.weights = self.ensemble_generator(
                    dataset_dict,
                    self.output_dict,
                )

    def calc_loss(
        self,
        dataset_dict: dict[str, torch.Tensor],
        loss_func: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        """Calculate combined loss across all models.

        Parameters
        ----------
        dataset_dict
            Dictionary containing input data.
        loss_func
            Loss function to use.

        Returns
        -------
        torch.Tensor
            Combined loss across all models.

        TODO: Support different loss functions for each model in ensemble.
        """
        if not self.loss_func and not loss_func:
            raise ValueError("No loss function defined.")
        loss_func = loss_func or self.loss_func

        loss_combined = 0.0

        # Loss calculation for each model
        for name, output in self.output_dict.items():
            if self.target_names[0] not in output.keys():
                raise ValueError(
                    f"Target variable '{self.target_names[0]}' not in model outputs."
                )
            output = output[self.target_names[0]]

            output, target = self._trim(output, dataset_dict['target'])
            loss = loss_func(
                output.squeeze(),
                target.squeeze(),
                sample_ids=dataset_dict['batch_sample'],
            )
            loss_combined += loss
            self.loss_dict[name] += loss.item()

        # Add ensemble loss if applicable (wNN trained in parallel)
        if self.multimodel_type in ['nn_parallel']:
            loss_combined += self.calc_loss_multimodel(dataset_dict, loss_func)

        return loss_combined

    def calc_loss_multimodel(
        self,
        dataset_dict: dict[str, torch.Tensor],
        loss_func: torch.nn.Module,
    ) -> torch.Tensor:
        """
        Calculate loss for multimodel ensemble wNN trained in parallel with
        differentiable models.

        Combine loss from ensemble predictions and range bound loss from
        weights.

        Parameters
        ----------
        dataset_dict
            Dictionary containing input data.
        loss_func
            Loss function to use.

        Returns
        -------
        torch.Tensor
            Combined loss for the multimodel ensemble.
        """
        if not self.loss_func_wnn and not loss_func:
            raise ValueError("No loss function defined.")
        self.loss_func_wnn = loss_func or self.loss_func_wnn

        # Sum of weights for each model
        weights_sum = torch.sum(
            torch.stack(
                [self.weights[model] for model in self.model_dict.keys()],
                dim=2,
            ),
            dim=2,
        )

        # Range bound loss
        if self.config['multimodel']['use_rb_loss']:
            rb_loss = self.range_bound_loss(
                weights_sum.clone().detach().requires_grad_(True),
            )
        else:
            rb_loss = 0.0

        output = self.ensemble_output_dict[self.target_names[0]]

        # Ensemble predictions loss
        ensemble_loss = self.loss_func_wnn(
            output.squeeze(),
            dataset_dict['target'][:, :, 0],
            sample_ids=dataset_dict['batch_sample'],
        )

        if self.verbose:
            if self.config['multimodel']['use_rb_loss']:
                tqdm.tqdm.write(
                    f"Ensemble loss: {ensemble_loss.item()}, "
                    f"Range bound loss: {rb_loss.item()}"
                )
            else:
                tqdm.tqdm.write(f"-- Ensemble loss: {ensemble_loss.item()}")

        loss_combined = ensemble_loss + rb_loss
        self.loss_dict['wNN'] += loss_combined.item()

        return loss_combined

    def save_model(self, epoch: int) -> None:
        """Save model state dicts.

        Parameters
        ----------
        epoch
            Epoch number to save model at.
        """
        for name, model in self.model_dict.items():
            save_model(self.config['model_dir'], model, name, epoch)
        if self.is_ensemble:
            save_model(self.config['model_dir'], self.ensemble_generator, 'wNN', epoch)

        if self.verbose:
            log.info(f"All states saved for ep:{epoch}")

    def get_states(self) -> None:
        """
        Helper function to expose physical and hidden (non-trainable) nn model
        states (e.g., for sequential simulations).
        """
        if len(self.model_dict) == 1:
            name = list(self.model_dict.keys())[0]
            nn_states = self.model_dict[name].nn_model.get_states()
            try:
                phy_states = self.model_dict[name].phy_model.get_states()
            except AttributeError:
                phy_states = None

            return nn_states, phy_states
        else:
            raise NotImplementedError(
                "Operations on hidden states for multimodel ensembles is not yet supported.",
            )

    def load_states(
        self,
        *,
        path: Optional[str] = None,
        nn_states: Optional[tuple[torch.Tensor, ...]] = None,
        phy_states: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> None:
        """
        Helper function to load physical and hidden (non-trainable) nn model
        states (e.g., for sequential simulations).
        """
        if path:
            if path and nn_states and phy_states:
                raise ValueError(
                    "Provide either `path` or `nn_states` and `phy_states`, not both."
                )
            if not os.path.exists(path):
                raise FileNotFoundError(f"State path {path} not found.")

            state_dict = torch.load(path, map_location=self.device)
            nn_states = state_dict.get('nn_states', None)
            phy_states = state_dict.get('phy_states', None)
            print(
                f"Loaded states from file | "
                f"epoch: {state_dict.get('epoch', 'N/A')} | "
                f"Resume from timestep: {state_dict.get('last_timestep', 'N/A')}"
            )
        elif nn_states:
            if not isinstance(nn_states, tuple):
                raise ValueError("`nn_states` must be a tuple of tensors.")
        elif phy_states:
            if not isinstance(phy_states, tuple):
                raise ValueError("`phy_states` must be a tuple of tensors.")
        else:
            raise ValueError(
                "Either `path` or `nn_states` and `phy_states` must be provided."
            )

        if len(self.model_dict) == 1:
            name = list(self.model_dict.keys())[0]
            self.model_dict[name].nn_model.load_states(nn_states)

            if phy_states is not None:
                try:
                    self.model_dict[name].phy_model.load_states(phy_states)
                except AttributeError:
                    pass
        else:
            raise NotImplementedError(
                "Operations on hidden states for multimodel ensembles is not yet supported.",
            )

    def save_states(self) -> None:
        """
        Helper function to save physical and nn model states (trainable and
        non-trainable) to disk.
        """
        if 'test' in self.config['mode']:
            mode = 'test'
        else:
            mode = 'sim'
        time = self.config[mode]['end_time']

        if len(self.model_dict) == 1:
            name = list(self.model_dict.keys())[0]

            nn_states, phy_states = self.get_states()

            state_dict = {
                'nn_states': nn_states,
                'nn_trainable': self.model_dict[
                    name
                ].state_dict(),  # weights and biases
                'phy_states': phy_states,
                'epoch': self.epoch,
                'last_timestep': time if time else 'N/A',
            }
            torch.save(state_dict, self.config['model_dir'] + "model_states.pt")
        else:
            raise NotImplementedError(
                "Operations on hidden states for multimodel ensembles is not yet supported.",
            )
        torch.save(state_dict, self.config['model_dir'] + "model_states.pt")

    def _trim(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Trim the output and target tensors to the same shape.

        Really, we need to trim warmup at the model interface itself, but this
        will have to do for now to avoid errors errant.

        Parameters
        ----------
        output
            The model output tensor.
        target
            The target tensor.

        Returns
        -------
        tuple
            The trimmed output and target tensors.
        """
        output = output.squeeze()
        target = target.squeeze()

        warm_up = self.config['model'].get('warm_up', 0)

        # Remove warmup timesteps
        target = target[warm_up:]

        if output.shape != target.shape:
            if output.shape[0] > target.shape[0]:
                output = output[warm_up:]
            elif target.shape[0] > output.shape[0]:
                target = target[warm_up:]
        return output, target

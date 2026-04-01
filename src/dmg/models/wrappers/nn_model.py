from typing import Any, Optional

import torch

from dmg.core.utils.factory import load_nn_model


class NnModel(torch.nn.Module):
    """Wrapper for running a neural network standalone.

    NOTE: this wrapper immitates implementation of DplModel for compatibility in
    the ModelHandler. However, we would ideally have a cleaner way of doing this
    in the future (i.e., this approach is a bit too derived).

    Parameters
    ----------
    model
        An initialized neural network model.
    config
        Configuration settings for the model.
    device
        Device to run the model on.
    """

    def __init__(
        self,
        *,
        target_names: list[str],
        model: Optional[torch.nn.Module] = None,
        config: Optional[dict[str, Any]] = None,
        device: Optional[torch.device] = 'cpu',
    ) -> None:
        super().__init__()
        self.name = 'Neural Network Model'
        self.target_names = target_names
        self.config = config
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu',
        )

        if model:
            self.nn_model = model.to(self.device)
        elif config:
            self.nn_model = self._init_model()
        else:
            raise ValueError(
                "A (1) initialized neural network or (2)"
                / " configuration dictionary is required.",
            )

        if len(self.target_names) != config['nn']['out_size']:
            raise ValueError(
                f"Number of target names ({len(self.target_names)}) does not"
                f" match model output size ({config['nn']['out_size']}).",
            )

        self.initialized = True

    def _init_model(self) -> torch.nn.Module:
        """Initialize a neural network model.

        Returns
        -------
        torch.nn.Module
            The neural network.
        """
        return load_nn_model(
            self.config,
            device=self.device,
        )

    def forward(self, data_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass, maps outputs to target names.

        Parameters
        ----------
        data_dict
            The input data dictionary.

        Returns
        -------
        torch.Tensor
            The output predictions.
        """
        prediction = self.nn_model(data_dict['xc_nn_norm'])

        out_dict = {}
        for name in self.target_names:
            out_dict[name] = prediction[self.config['warm_up'] :, ...]

        return out_dict

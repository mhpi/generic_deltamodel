
from dmg.core.utils.factory import import_phy_model, load_nn_model

import torch
from typing import Optional, Any


class DplModel(torch.nn.Module):

    def __init__(
            self,
            phy_model_name: Optional[str] = None,
            phy_model: Optional[torch.nn.Module] = None,
            nn_model: Optional[torch.nn.Module] = None,
            config: Optional[dict[str, Any]] = None,
            device: Optional[torch.device] = 'cpu',
    ) -> None:
        super().__init__()
        self.name = 'dPL Model'
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if nn_model and phy_model:
            self.phy_model = phy_model.to(self.device)
            self.nn_model = nn_model.to(self.device)
        elif config:
            # Initialize new models.
            self.phy_model = self._init_phy_model(phy_model_name)
            self.nn_model = self._init_nn_model()
        else:
            raise ValueError("A (1) neural network and physics model or (2)" /
                             " configuration dictionary is required.")

        self.initialized = True

        # # Compile models
        # self.nn_model = torch.compile(self.nn_model, mode="reduce-overhead", fullgraph=False)
        # self.phy_model = torch.compile(self.phy_model, mode="reduce-overhead", fullgraph=False)

    def _init_phy_model(self, phy_model_name) -> torch.nn.Module:
        # TODO: add Hbv_2h and Hbv_2_mts to hydrodl2, add Hbv_2_mts support to import_phy_model and load_component
        """Initialize a physics model.

        Parameters
        ----------
        phy_model_name
            The name of the physics model.

        Returns
        -------
        torch.nn.Module
            The physics model.
        """
        if phy_model_name:
            model_name = phy_model_name
        elif self.config['phy_model']:
            model_name = self.config['phy_model']['model'][0]
        else:
            raise ValueError("A (1) physics model name or (2) model spec in" /
                             " a configuration dictionary is required.")

        model = import_phy_model(model_name)
        return model(self.config['phy_model'], device=self.device)

    def _init_nn_model(self) -> torch.nn.Module:
        # TODO: add LstmMlp2Model and StackLstmMlpModel to load_nn_model
        """Initialize a neural network model.

        Returns
        -------
        torch.nn.Module
            The neural network.
        """
        return load_nn_model(
            self.phy_model,
            self.config,
            device=self.device,
        )

    def forward(self, data_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        data_dict
            The input data dictionary.

        Returns
        -------
        torch.Tensor
            The output predictions.
        """
        # Neural network
        ##################### New: Multi-timescale #################



        if type(self.nn_model).__name__ == 'StackLstmMlpModel' or (hasattr(self.nn_model, '_orig_mod') and type(self.nn_model._orig_mod).__name__ == 'StackLstmMlpModel'):
            parameters = self.nn_model(
                (data_dict['xc_nn_norm_low_freq'], data_dict['c_nn_norm']),
                (data_dict['xc_nn_norm_high_freq'], data_dict['c_nn_norm'], data_dict['rc_nn_norm'])
            )
        ############################################################
        elif type(self.nn_model).__name__ == 'LstmMlpModel':
            parameters = self.nn_model(data_dict['xc_nn_norm'], data_dict['c_nn_norm'])
        else:
            parameters = self.nn_model(data_dict['xc_nn_norm'])

        # Physics model
        predictions = self.phy_model(
            data_dict,
            parameters,
        )

        return predictions








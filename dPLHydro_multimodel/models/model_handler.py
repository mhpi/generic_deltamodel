import os

import torch.nn
from models.differentiable_model import dPLHydroModel
from models.loss_functions import get_loss_function


class ModelHandler(torch.nn.Module):
    """
    Streamlines instantiation and handling of differentiable models &
    multimodel ensembles.

    Basic functions include managing: 
    - high-level model init
    - optimizer
    - loss function(s)
    - high-level forwarding

    NOTE: In addition to interfacing with experiments, this handler is a plugin
    for BMI. All PMI-interfaced models must ultimately use this handler if they
    are to be BMI compatible.

    NOTE: Optimizer must be initialized within handler, not externally, so that
    it can be wrapped by the BMI (NextGen compatibility).
    """
    def __init__(self, config):
        super(ModelHandler, self).__init__()
        self.config = config
        self.name = 'Differentiable Model Handler'
        self._init_models()
        
    def _init_models(self):
        """
        Initialize and store each differentiable hydro model and optimizer in
        the multimodel.
        """
        self.model_dict = {}

        if (self.config['ensemble_type'] == 'none') and (len(self.config['hydro_models']) > 1):
            raise ValueError("Multiple hydro models given, but ensemble type not specified. Check config.")
        
        elif self.config['mode'] == 'train_wnn':
            # Reinitialize trained models for wNN training.
            for mod in self.config['hydro_models']:
                load_path = self.config['checkpoint'][mod]
                self.model_dict[mod] = torch.load(load_path).to(self.config['device'])

                # Overwrite internal config if there is discontinuity:
                if self.model_dict[mod].config:
                    self.model_dict[mod].config = self.config

                self.model_dict[mod].zero_grad()

        elif self.config['use_checkpoint']:
            # Reinitialize trained model(s).
            self.all_model_params = []
            for mod in self.config['hydro_models']:
                load_path = self.config['checkpoint'][mod]
                self.model_dict[mod] = torch.load(load_path).to(self.config['device'])
                self.all_model_params += list(self.model_dict[mod].parameters())

                self.model_dict[mod].zero_grad()
                self.model_dict[mod].train()
            self.init_optimizer()

        elif self.config['mode'] in ['test', 'test_conus']:
            for mod in self.config['hydro_models']:
                self.load_model(mod)

        else:
            # Initialize differentiable hydrology model(s) and bulk optimizer.
            self.all_model_params = []
            for mod in self.config['hydro_models']:

                ### TODO: change which models are set to which devices here: ###
                self.model_dict[mod] = dPLHydroModel(self.config, mod).to(self.config['device'])
                self.all_model_params += list(self.model_dict[mod].parameters())

                self.model_dict[mod].zero_grad()
                self.model_dict[mod].train()
            self.init_optimizer()
    
    def load_model(self, model) -> None:
        model_name = str(model) + '_model_Ep' + str(self.config['epochs']) + '.pt'
        model_path = os.path.join(self.config['output_dir'], model_name)
        try:
            self.model_dict[model] = torch.load(model_path).to(self.config['device'])

            # Overwrite internal config if there is discontinuity:
            if self.model_dict[model].config:
                self.model_dict[model].config = self.config
        except:
            raise FileNotFoundError(f"Model file {model_path} was not found. Check configurations.")

    def init_loss_func(self, obs) -> None:
        self.loss_func = get_loss_function(self.config, obs)
        self.loss_func = self.loss_func.to(self.config['device'])

    def init_optimizer(self) -> None:
        self.optim = torch.optim.Adadelta(self.all_model_params, lr=self.config['learning_rate'])

    def forward(self, dataset_dict_sample, eval=False):        
        """
        Batch forward one or more differentiable hydrology models.
        """
        self.flow_out_dict = dict()
        self.dataset_dict_sample = dataset_dict_sample

        # Forward
        for mod in self.model_dict:
            ## Test/Validation
            if eval:
                self.model_dict[mod].eval()
                # torch.set_grad_enabled(False)
                self.flow_out_dict[mod] = self.model_dict[mod](dataset_dict_sample)
                # torch.set_grad_enabled(True)
            ## Train
            else:
                self.flow_out_dict[mod] = self.model_dict[mod](dataset_dict_sample)
        return self.flow_out_dict

    def calc_loss(self, loss_dict) -> None:
        total_loss = 0
        for mod in self.model_dict:
            # if self.flow_out_dict[mod] == 'HBV_capillary':
            #     # Capillary HBV requires all sample observations without warm-up trimmed.
            #     obs = self.dataset_dict_sample['obs']
            # else:
            #     obs = self.dataset_dict_sample['obs'][config['warm_up']:]

            loss = self.loss_func(self.config,
                                  self.flow_out_dict[mod],
                                  self.dataset_dict_sample['obs'],
                                  igrid=self.dataset_dict_sample['iGrid']
                                  )
            # self.model_dict[mod].zero_grad()

            total_loss += loss
            loss_dict[mod] += loss.item()
        
        # total_loss.backward()
        # self.optim.step()

        return total_loss, loss_dict
    
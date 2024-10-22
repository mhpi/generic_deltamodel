import torch.nn

from MODELS.hydro_models.marrmot_PRMS.prms_marrmot import prms_marrmot
from MODELS.hydro_models.marrmot_PRMS_refreeze.prms_marrmot_refreeze import prms_marrmot_refreeze
from MODELS.hydro_models.HBV.HBVmul import HBVMul
from MODELS.hydro_models.HBV_capillary.HBVmultdET import HBVMulTDET
from MODELS.hydro_models.SACSMA.SACSMAmul import SACSMAMul
from MODELS.hydro_models.SACSMA_with_snowpack.SACSMA_snow_mul import SACSMA_snow_Mul

from MODELS.NN_models.LSTM_models import CudnnLstmModel
from MODELS.NN_models.MLP_models import MLPmul

from core.utils.small_codes import source_flow_calculation


# import MODELS
class diff_gDM(torch.nn.Module):
    def __init__(self, args):
        super(diff_gDM, self).__init__()
        self.args = args
        self.get_model()

    def get_NN_model_dim(self) -> None:
        self.nx = len(self.args["varT_NN"] + self.args["varC_NN"])

        # output size of NN
        if self.args["hydro_model_name"] != "None":
            if self.args["routing_hydro_model"] == True:  # needs a and b for routing with conv method
                self.ny_hydro = self.args["nmul"] * (len(self.hydro_model.parameters_bound)) + len(
                    self.hydro_model.conv_routing_hydro_model_bound)
            else:
                self.ny_hydro = self.args["nmul"] * len(self.hydro_model.parameters_bound)
        else:
            self.ny_hydro = 0

        self.ny = self.ny_hydro

    def get_model(self) -> None:
        # hydro_model_initialization
        if self.args["hydro_model_name"] != "None":
            if self.args["hydro_model_name"] == "marrmot_PRMS":
                self.hydro_model = prms_marrmot()
            elif self.args["hydro_model_name"] == "marrmot_PRMS_refreeze":
                self.hydro_model = prms_marrmot_refreeze()
            elif self.args["hydro_model_name"] == "HBV":
                self.hydro_model = HBVMul(self.args)
            elif self.args["hydro_model_name"] == "HBV_capillary":
                self.hydro_model = HBVMulTDET(self.args)
            elif self.args["hydro_model_name"] == "SACSMA":
                self.hydro_model = SACSMAMul()
            elif self.args["hydro_model_name"] == "SACSMA_with_snow":
                self.hydro_model = SACSMA_snow_Mul()
            else:
                print("hydrology (streamflow) model type has not been defined")
                exit()

        # get the dimensions of NN model based on hydro modela and temp model
        self.get_NN_model_dim()
        # NN_model_initialization
        if self.args["NN_model_name"] == "LSTM":
            self.NN_model = CudnnLstmModel(nx=self.nx,
                                           ny=self.ny,
                                           hiddenSize=self.args["hidden_size"],
                                           dr=self.args["dropout"])
        elif self.args["NN_model_name"] == "MLP":
            self.NN_model = MLPmul(self.args, nx=self.nx, ny=self.ny)
        else:
            print("NN model type has not been defined")
            exit()

    def breakdown_params(self, params_all):
        params_dict = dict()
        params_hydro_model = params_all[:, :, :self.ny_hydro]

        if self.args['hydro_model_name'] != "None":
            # hydro params
            params_dict["hydro_params_raw"] = torch.sigmoid(
                params_hydro_model[:, :, :len(self.hydro_model.parameters_bound) * self.args["nmul"]]).view(
                params_hydro_model.shape[0], params_hydro_model.shape[1], len(self.hydro_model.parameters_bound),
                self.args["nmul"])
            # routing params
            if self.args["routing_hydro_model"] == True:
                params_dict["conv_params_hydro"] = torch.sigmoid(
                    params_hydro_model[-1, :, len(self.hydro_model.parameters_bound) * self.args["nmul"]:])
            else:
                params_dict["conv_params_hydro"] = None

        return params_dict

    def forward(self, dataset_dictionary_sample):
        params_all = self.NN_model(dataset_dictionary_sample["inputs_NN_scaled"])  # [self.args["warm_up"]:, :, :]
        # breaking down the parameters to different pieces for different models (PET, hydro, temp)
        params_dict = self.breakdown_params(params_all)
        # hydro model
        flow_out = self.hydro_model(
            dataset_dictionary_sample["x_hydro_model"],
            dataset_dictionary_sample["c_hydro_model"],
            params_dict['hydro_params_raw'],
            self.args,
            # PET_param=params_dict["params_PET_model"],  # PET is in both temp and flow model
            warm_up=self.args["warm_up"],
            routing=self.args["routing_hydro_model"],
            conv_params_hydro=params_dict["conv_params_hydro"]
        )
        # to remove the warm_up part --> consistent with removing the torch.no_grad() in hydro_model
        for key in flow_out.keys():
            flow_out[key] = flow_out[key][self.args["warm_up"]:]

        return flow_out

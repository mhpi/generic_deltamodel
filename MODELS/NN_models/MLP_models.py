import numpy as np
import torch
import torch.nn as nn
import datetime
import torch.nn.functional as F
import math
# from core.read_configurations import config

# from rnn import CudnnLstmModel


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        # self.seq_lin_layers = nn.Sequential(
        #     nn.Linear(
        #         len(args["optData"]["varC"]), args["seq_lin_layers"]["hidden_size"]
        #     ),
        #     # nn.ReLU(),
        #     nn.Linear(
        #         args["seq_lin_layers"]["hidden_size"],
        #         args["seq_lin_layers"]["hidden_size"],
        #     ),
        #     # nn.ReLU(),
        #     nn.Linear(
        #         args["seq_lin_layers"]["hidden_size"],
        #         args["seq_lin_layers"]["hidden_size"],
        #     ),
        #     nn.Linear(
        #         args["seq_lin_layers"]["hidden_size"],
        #         args["seq_lin_layers"]["hidden_size"],
        #     ),
        #     # nn.ReLU(),
        #     nn.Linear(
        #         args["seq_lin_layers"]["hidden_size"],
        #         args["res_time_params"]["lenF_srflow"]
        #         + args["res_time_params"]["lenF_ssflow"]
        #         + args["res_time_params"]["lenF_gwflow"],
        #     ),
        #     # nn.ReLU()
        # )
        self.L1 = nn.Linear(
            len(args["optData"]["varC"]), args["seq_lin_layers"]["hidden_size"]
        )
        self.L2 = nn.Linear(
            args["seq_lin_layers"]["hidden_size"], args["seq_lin_layers"]["hidden_size"]
        )
        self.L3 = nn.Linear(
            args["seq_lin_layers"]["hidden_size"], args["seq_lin_layers"]["hidden_size"]
        )

        self.L4 = nn.Linear(args["seq_lin_layers"]["hidden_size"], 23)

        # 6 for alpha and beta of surface/subsurface/groundwater flow
        # 3 for conv bias,
        # 2 for scaling and bias of final answer,
        # 1 for shade_factor_riparian
        # 3 for surface/subsurface/groundwater flow percentage
        # 1 for albedo
        # 1 for solar shade factor
        # 4 for width coefficient nominator, width coefficient denominator, width A coefficient, and width exponent
        # 2 for p & q

        # self.lstm = CudnnLstmModel(
        #     nx=input.shape[2],
        #     ny=len(args['params_target']),
        #     hiddenSize=args['hyperprameters']['hidden_size'],
        #     dr=args['hyperprameters']['dropout'])
        #
        # self.stream_temp_eq = stream_temp_eq(args)
        self.activation_sigmoid = torch.nn.Sigmoid()
        self.activation_tanh = torch.nn.Tanh()

    def forward(self, x):
        # out = self.seq_lin_layers(x)
        out = self.L1(x)
        out = self.L2(out)
        out = self.L3(out)
        out = self.L4(out)
        # out1 = torch.abs(out)
        out = self.activation_sigmoid(out)
        return out


class MLPmul(nn.Module):
    def __init__(self, args, nx, ny):
        super(MLPmul, self).__init__()
        self.args = args
        self.L1 = nn.Linear(
            nx,  self.args["hidden_size"],
        )
        self.L2 = nn.Linear(
            self.args["hidden_size"], self.args["hidden_size"]
        )
        self.L3 = nn.Linear(
            self.args["hidden_size"], self.args["hidden_size"]
        )

        self.L4 = nn.Linear(self.args["hidden_size"], ny)
        self.activation_sigmoid = torch.nn.Sigmoid()
        self.activation_tanh = torch.nn.Tanh()

    def forward(self, x):
        # out = self.seq_lin_layers(x)
        out = self.L1(x)
        out = self.L2(out)
        out = self.L3(out)
        out = self.L4(out)
        # out1 = torch.abs(out)
        out = self.activation_sigmoid(out)
        return out

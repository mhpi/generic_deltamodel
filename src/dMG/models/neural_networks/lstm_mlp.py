import torch

from dMG.models.neural_networks.ann import AnnModel
from dMG.models.neural_networks.cudnn_lstm import CudnnLstmModel
from dMG.models.neural_networks.lstm import LstmModel


class LstmMlpModel(torch.nn.Module):
    """LSTM-MLP model for multi-scale learning."""
    def __init__(
        self,
        *,
        nx1,
        ny1,
        hiddeninv1,
        nx2,
        ny2,
        hiddeninv2,
        dr1=0.5,
        dr2=0.5,
        device='cuda',
    ):
        super().__init__()
        self.name = 'LstmMlpModel'
        
        # if device == 'cpu':
        
        # CPU-compatible LSTM model.
        self.lstminv = LstmModel(
            nx=nx1, ny=ny1, hiddenSize=hiddeninv1, dr=dr1
        )
        



        # else:
        # self.lstminv = CudnnLstmModel(
        #     nx=nx1, ny=ny1, hiddenSize=hiddeninv1, dr=dr1
        # )
    
        self.Ann = AnnModel(
            nx=nx2, ny=ny2, hiddenSize=hiddeninv2, dropout_rate=dr2)

    def forward(self, z1,z2):
        Lstm_para = self.lstminv(z1) # dim: Time, Gage, Para

        Ann_para = self.Ann(z2)

        return [ torch.sigmoid(Lstm_para),Ann_para]
